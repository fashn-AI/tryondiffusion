from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from tryondiffusion.common.python_helpers import (
    cast_tuple,
    default,
    exists,
    identity,
    maybe,
    pad_tuple_to_length,
)
from tryondiffusion.common.torch_helpers import (
    eval_decorator,
    module_device,
    zero_init_,
)
from tryondiffusion.common.torch_transforms import (
    cast_uint8_images_to_float,
    normalize_neg_one_to_one,
    unnormalize_zero_to_one,
)
from tryondiffusion.modules.general import apply_conditional_dropout
from tryondiffusion.modules.imagen import (
    CrossEmbedLayer,
    GaussianDiffusionContinuousTimes,
    LearnedSinusoidalPosEmb,
    PerceiverResampler,
    ResnetBlock,
    UpsampleCombiner,
    prob_mask_like,
    right_pad_dims_to,
)
from tryondiffusion.modules.tryon import (
    GarmentDownBlock,
    GarmentUpBlock,
    PersonDownBlock,
    PersonUpBlock,
    resize_image_to,
)

__all__ = ["ParallelUNet", "TryOnImagen", "BaseParallelUnet", "SRParallelUnet", "NullUnet", "get_unet_by_name"]


def build_time_cond_layers(emb_dim, time_cond_dim, context_dim, num_tokens):
    """
    Build layers for handling noise-time conditioning inputs,
    preparing them for both attention-based and non-attention based conditioning.

    Args:
        emb_dim (int): The dimension of the input embeddings (will usually simply be 1 for time step or noise level).
        time_cond_dim (int): The dimension for non-attention conditioning.
        context_dim (int): The dimension of the attention-based conditioning.
        num_tokens (int): The number of tokens (for attention-based conditioning).

    Returns:
        nn.ModuleDict: The module dictionary containing the layers.
    """
    layers = nn.ModuleDict(
        {
            "to_hiddens": nn.Sequential(
                LearnedSinusoidalPosEmb(emb_dim), nn.Linear(emb_dim + 1, time_cond_dim), nn.SiLU()
            ),
            "to_cond": nn.Sequential(nn.Linear(time_cond_dim, time_cond_dim)),
        }
    )
    if num_tokens > 0:
        layers["to_tokens"] = nn.Sequential(
            nn.Linear(time_cond_dim, context_dim * num_tokens),
            Rearrange("b (r d) -> b r d", r=num_tokens),
        )
    return layers


def process_time_cond_layers(time_layers, input_emb, mask=None):
    """
    Helper method to process time conditioning layers.
    """
    hiddens = time_layers["to_hiddens"](input_emb)
    cond = time_layers["to_cond"](hiddens)
    if exists(mask):
        cond = apply_conditional_dropout(cond, mask=mask)

    tokens = None
    if "to_tokens" in time_layers:
        tokens = time_layers["to_tokens"](hiddens)
        if exists(mask):
            tokens = apply_conditional_dropout(tokens, mask=mask)

    return cond, tokens


class ParallelUNet(nn.Module):
    def __init__(
        self,
        *,
        image_size: Tuple[int] = (144, 112),
        dim: int = 128,
        num_blocks: Union[int, Tuple[int]] = (3, 4, 6, 7),
        context_dim: int = 512,
        time_cond_dim: Optional[int] = None,
        num_time_tokens: int = 0,
        learned_sinu_pos_emb_dim: int = 16,
        feature_channels: Tuple[int] = (128, 256, 512, 1024),
        channels: int = 3,
        channels_out: Optional[int] = None,
        ca_image_channels: int = 3,
        garment_image_channels: int = 3,
        dropout_probs: Union[int, Tuple[int]] = 0.0,
        pose_depth: int = 2,
        pose_num_latents: int = 18,
        pose_num_latents_mean_pooled: int = 4,
        pose_attn_heads: int = 8,
        pose_attn_dim_head: int = 64,
        max_keypoints_len: int = 18,
        lowres_cond: bool = False,  # for cascading diffusion - https://cascaded-diffusion.github.io/
        layer_attns: Union[bool, Tuple[bool]] = (False, False, True, True),
        layer_cross_attns: Union[bool, Tuple[bool]] = (False, False, True, True),
        attn_dim_head: Tuple[Optional[int]] = (None, None, 4, 8),
        attn_heads: Tuple[Optional[int]] = (None, None, 128, 64),
        init_dim: int = None,
        init_conv_kernel_size: int = 3,  # kernel size of initial conv, if not using cross embed
        init_cross_embed: bool = False,
        init_cross_embed_kernel_sizes: Tuple[int] = (3, 7, 15),
        cross_embed_downsample: bool = False,
        cross_embed_downsample_kernel_sizes: Tuple[int] = (2, 4),
        init_conv_to_final_conv_residual: bool = False,
        use_global_context_attn: bool = True,
        scale_skip_connection: bool = True,
        use_zero_init: bool = True,
        final_resnet_block: bool = True,
        final_conv_kernel_size: int = 3,
        combine_upsample_fmaps: bool = False,  # combine feature maps from all upsample blocks, used in unet squared successfully
        pixel_shuffle_upsample: bool = True,  # may address checkboard artifacts
    ):
        super().__init__()

        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        self._locals.pop("self", None)
        self._locals.pop("__class__", None)

        # determine resolutions
        self.image_size = image_size
        self.resolutions = self._compute_resolutions(feature_channels)

        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        # in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        init_channels = channels * (1 + int(lowres_cond))
        init_channels += ca_image_channels
        init_dim = default(init_dim, dim)

        # initial convolution

        self.init_conv = (
            CrossEmbedLayer(init_channels, dim_out=init_dim, kernel_sizes=init_cross_embed_kernel_sizes, stride=1)
            if init_cross_embed
            else nn.Conv2d(init_channels, init_dim, init_conv_kernel_size, padding=init_conv_kernel_size // 2)
        )
        self.garment_init_conv = (
            CrossEmbedLayer(
                garment_image_channels, dim_out=init_dim, kernel_sizes=init_cross_embed_kernel_sizes, stride=1
            )
            if init_cross_embed
            else nn.Conv2d(garment_image_channels, init_dim, init_conv_kernel_size, padding=init_conv_kernel_size // 2)
        )

        # Prepare the feature dimensions structure for the down and up blocks
        down_in_out = []
        for i in range(len(feature_channels) - 1):
            block_in_out = (feature_channels[i], feature_channels[i + 1])
            down_in_out.append(block_in_out)
        down_in_out.append((feature_channels[-1], feature_channels[-1]))

        up_in_out = []
        for i in reversed(range(1, len(feature_channels))):
            block_in_out = (feature_channels[i], feature_channels[i - 1])
            up_in_out.append(block_in_out)
        up_in_out.append((feature_channels[0], feature_channels[0]))

        # dimension for attention-based conditioning (self and cross)
        context_dim = default(context_dim, dim)
        # time conditioning (or more generally - non-attention based conditioning)
        time_cond_dim = default(time_cond_dim, dim * 4 * (1 + int(lowres_cond)))

        # time conditioning layers
        self.time_layers = build_time_cond_layers(learned_sinu_pos_emb_dim, time_cond_dim, context_dim, num_time_tokens)

        # low res aug noise conditioning
        self.lowres_cond = lowres_cond
        if lowres_cond:
            self.lowres_layers = build_time_cond_layers(
                learned_sinu_pos_emb_dim, time_cond_dim, context_dim, num_time_tokens
            )
        # clothing-agnostic noise conditioning
        self.ca_noise_aug_layers = build_time_cond_layers(
            learned_sinu_pos_emb_dim, time_cond_dim, context_dim, num_time_tokens
        )
        # garment noise conditioning
        self.garment_noise_aug_layers = build_time_cond_layers(
            learned_sinu_pos_emb_dim, time_cond_dim, context_dim, num_time_tokens
        )

        # pose keypoints conditioning for both person and garment
        self.pose_to_cond = nn.ModuleList([nn.Linear(2, context_dim) for _ in range(2)])

        # attention pooling for pose keypoints
        self.pose_attn_pool = nn.ModuleList(
            [
                PerceiverResampler(
                    dim=context_dim,
                    depth=pose_depth,
                    dim_head=pose_attn_dim_head,
                    heads=pose_attn_heads,
                    num_latents=pose_num_latents,
                    num_latents_mean_pooled=pose_num_latents_mean_pooled,
                    max_seq_len=max_keypoints_len,
                )
                for _ in range(2)
            ]
        )

        # normalizations
        self.norm_cond = nn.LayerNorm(context_dim)

        # for non-attention based pose conditioning at all points in the network where time is also conditioned
        self.to_pose_non_attn_cond = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(context_dim),
                    nn.Linear(context_dim, time_cond_dim),
                    nn.SiLU(),
                    nn.Linear(time_cond_dim, time_cond_dim),
                )
                for _ in range(2)
            ]
        )

        # for classifier free guidance
        self.null_pose_hidden = nn.ParameterList([nn.Parameter(torch.randn(1, time_cond_dim)) for _ in range(2)])
        self.null_pose_tokens = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, pose_num_latents + pose_num_latents_mean_pooled, context_dim))
                for _ in range(2)
            ]
        )

        num_layers = len(down_in_out)

        # resnet block klass

        num_blocks = cast_tuple(num_blocks, num_layers)
        layer_attns = cast_tuple(layer_attns, num_layers)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_layers)
        dropout_probs = cast_tuple(dropout_probs, num_layers)

        assert all([layers == num_layers for layers in list(map(len, (layer_attns, layer_cross_attns)))])

        # scale for resnet skip connections

        self.skip_connect_scale = 1.0 if not scale_skip_connection else (2**-0.5)

        # layers

        self.person_downs = nn.ModuleList([])
        self.person_ups = nn.ModuleList([])
        self.garment_downs = nn.ModuleList([])
        self.garment_ups = nn.ModuleList([])

        num_resolutions = len(self.resolutions)

        layer_params = [
            self.resolutions,
            num_blocks,
            layer_attns,
            layer_cross_attns,
            attn_heads,
            attn_dim_head,
            dropout_probs,
        ]
        reversed_layer_params = list(map(reversed, layer_params))

        # downsampling layers

        skip_connect_dims = []  # keep track of skip connection dimensions

        for ind, (
            (dim_in, dim_out),
            layer_resolution,
            layer_num_blocks,
            layer_attn,
            layer_cross_attn,
            layer_attn_heads,
            layer_attn_dim_head,
            dropout_prob,
        ) in enumerate(zip(down_in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)

            current_dim = dim_in
            skip_connect_dims.append(current_dim)

            self.person_downs.append(
                PersonDownBlock(
                    dim_in=current_dim,
                    dim_out=dim_out,
                    time_cond_dim=time_cond_dim,
                    num_blocks=layer_num_blocks,
                    context_dim=context_dim,
                    use_gca=use_global_context_attn,
                    zero_init=use_zero_init,
                    is_last=is_last,
                    cross_embed_downsample=cross_embed_downsample,
                    cross_embed_downsample_kernel_sizes=cross_embed_downsample_kernel_sizes,
                    do_attn=layer_attn,
                    do_cross_attn=layer_cross_attn,
                    patch_size=max(1, round(max(layer_resolution) / 16)),
                    dropout_prob=dropout_prob,
                    attn_dim_head=layer_attn_dim_head,
                    attn_num_heads=layer_attn_heads,
                )
            )
            self.garment_downs.append(
                GarmentDownBlock(
                    dim_in=current_dim,
                    dim_out=dim_out,
                    time_cond_dim=time_cond_dim,
                    num_blocks=layer_num_blocks,
                    output_hiddens=layer_cross_attn,
                    use_gca=use_global_context_attn,
                    zero_init=use_zero_init,
                    is_last=is_last,
                    cross_embed_downsample=cross_embed_downsample,
                    cross_embed_downsample_kernel_sizes=cross_embed_downsample_kernel_sizes,
                    dropout_prob=dropout_prob,
                )
            )

        # upsampling layers

        upsample_fmap_dims = []

        for ind, (
            (dim_in, dim_out),
            layer_resolution,
            layer_num_blocks,
            layer_attn,
            layer_cross_attn,
            layer_attn_heads,
            layer_attn_dim_head,
            dropout_prob,
        ) in enumerate((zip(up_in_out, *reversed_layer_params))):
            is_last = ind == (len(up_in_out) - 1)

            skip_connect_dim = skip_connect_dims.pop()

            upsample_fmap_dims.append(dim_out)

            self.person_ups.append(
                PersonUpBlock(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    skip_dim=skip_connect_dim,
                    num_blocks=layer_num_blocks,
                    time_cond_dim=time_cond_dim,
                    context_dim=context_dim,
                    use_gca=use_global_context_attn,
                    zero_init=use_zero_init,
                    is_last=is_last,
                    pixel_shuffle_upsample=pixel_shuffle_upsample,
                    scale_skip_connection=scale_skip_connection,
                    do_attn=layer_attn,
                    do_cross_attn=layer_cross_attn,
                    patch_size=max(1, round(max(layer_resolution) / 16)),
                    dropout_prob=dropout_prob,
                    attn_dim_head=layer_attn_dim_head,
                    attn_num_heads=layer_attn_heads,
                )
            )
            self.garment_ups.append(
                GarmentUpBlock(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    skip_dim=skip_connect_dim,
                    num_blocks=layer_num_blocks,
                    time_cond_dim=time_cond_dim,
                    use_gca=use_global_context_attn,
                    zero_init=use_zero_init,
                    is_last=is_last,
                    pixel_shuffle_upsample=pixel_shuffle_upsample,
                    scale_skip_connection=scale_skip_connection,
                    dropout_prob=dropout_prob,
                )
                if layer_cross_attn
                else None
            )

        # whether to combine feature maps from all upsample blocks before final resnet block out
        self.upsample_combiner = UpsampleCombiner(
            dim=dim, enabled=combine_upsample_fmaps, dim_ins=upsample_fmap_dims, dim_outs=dim
        )

        # whether to do a final residual from initial conv to the final resnet block out

        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = self.upsample_combiner.dim_out + (dim if init_conv_to_final_conv_residual else 0)

        # final optional resnet block and convolution out

        self.final_res_block = (
            ResnetBlock(final_conv_dim, dim, time_cond_dim=time_cond_dim, use_gca=True) if final_resnet_block else None
        )

        final_conv_dim_in = dim if final_resnet_block else final_conv_dim
        final_conv_dim_in += channels if lowres_cond else 0

        self.final_conv = nn.Conv2d(
            final_conv_dim_in, self.channels_out, final_conv_kernel_size, padding=final_conv_kernel_size // 2
        )

        zero_init_(self.final_conv)

    def _compute_resolutions(self, dim_mults) -> List[Tuple[int, int]]:
        current_height, current_width = self.image_size
        resolutions = [(current_height, current_width)]

        for _ in dim_mults[:-1]:
            current_width //= 2
            current_height //= 2
            resolutions.append((current_height, current_width))

        return resolutions

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(self, *, lowres_cond, channels, channels_out):
        if lowres_cond == self.lowres_cond and channels == self.channels and channels_out == self.channels_out:
            return self

        updated_kwargs = dict(
            lowres_cond=lowres_cond,
            channels=channels,
            channels_out=channels_out,
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    # methods for returning the full unet config as well as its parameter state

    def to_config_and_state_dict(self):
        return self._locals, self.state_dict()

    # class method for rehydrating the unet from its config and state dict

    @classmethod
    def from_config_and_state_dict(klass, config, state_dict):
        unet = klass(**config)
        unet.load_state_dict(state_dict)
        return unet

    # methods for persisting unet to disk

    def persist_to_file(self, path):
        path = Path(path)
        path.parents[0].mkdir(exist_ok=True, parents=True)

        config, state_dict = self.to_config_and_state_dict()
        pkg = dict(config=config, state_dict=state_dict)
        torch.save(pkg, str(path))

    # class method for rehydrating the unet from file saved with `persist_to_file`

    @classmethod
    def hydrate_from_file(klass, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))

        assert "config" in pkg and "state_dict" in pkg
        config, state_dict = pkg["config"], pkg["state_dict"]

        return ParallelUNet.from_config_and_state_dict(config, state_dict)

    # forward with classifier free guidance

    def forward_with_cond_scale(self, *args, cond_scale=2.0, **kwargs):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        noisy_images,
        time,
        *,
        ca_images,
        ca_noise_times,
        garment_images,
        garment_noise_times,
        person_poses,
        garment_poses,
        lowres_cond_img=None,
        lowres_noise_times=None,
        cond_drop_prob=0.0,
    ):
        batch_size, device = noisy_images.shape[0], noisy_images.device

        # add low resolution conditioning, if present
        assert not (
            self.lowres_cond and not exists(lowres_cond_img)
        ), "low resolution conditioning image must be present"
        assert not (
            self.lowres_cond and not exists(lowres_noise_times)
        ), "low resolution conditioning noise time must be present"

        if exists(lowres_cond_img):
            noisy_images = torch.cat((noisy_images, lowres_cond_img), dim=1)

        # Prepare dropout masks for all conditional inputs
        ca_images_keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        garment_images_keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        poses_keep_mask = [prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device) for _ in range(2)]

        # condition on clothing-agnostic image
        ca_images = apply_conditional_dropout(ca_images, ca_images_keep_mask)
        noisy_images = torch.cat((ca_images, noisy_images), dim=1)

        # initial convolution

        noisy_images = self.init_conv(noisy_images)
        garment_images = self.garment_init_conv(garment_images)

        # init conv residual

        if self.init_conv_to_final_conv_residual:
            init_conv_residual = noisy_images.clone()

        # Process standard time conditioning
        t, time_tokens = process_time_cond_layers(self.time_layers, time)

        # Process lowres time conditioning
        if self.lowres_cond:
            lowres_t, lowres_time_tokens = process_time_cond_layers(self.lowres_layers, lowres_noise_times)
            t = t + lowres_t
            if exists(time_tokens):
                time_tokens = torch.cat((time_tokens, lowres_time_tokens), dim=-2)

        # Process clothing agnostic noise augmentation conditioning
        ca_t, ca_time_tokens = process_time_cond_layers(
            self.ca_noise_aug_layers, ca_noise_times, mask=ca_images_keep_mask
        )
        t = t + ca_t
        if exists(time_tokens):
            time_tokens = torch.cat((time_tokens, ca_time_tokens), dim=-2)

        # Process garment noise augmentation conditioning
        garment_t, garment_time_tokens = process_time_cond_layers(
            self.garment_noise_aug_layers, garment_noise_times, mask=garment_images_keep_mask
        )
        t = t + garment_t
        if exists(time_tokens):
            time_tokens = torch.cat((time_tokens, garment_time_tokens), dim=-2)

        # Pose conditioning for person and garment in sequence
        c = []
        for i, pose in enumerate([person_poses, garment_poses]):
            pose_tokens = self.pose_to_cond[i](pose)
            pose_tokens = self.pose_attn_pool[i](pose_tokens)

            # Create non-attention pose conditioning
            mean_pooled_pose_tokens = pose_tokens.mean(dim=-2)
            pose_hiddens = self.to_pose_non_attn_cond[i](mean_pooled_pose_tokens)

            # Apply dropout to pose hiddens and tokens
            pose_hiddens = apply_conditional_dropout(
                pose_hiddens, poses_keep_mask[i], null_tensor=self.null_pose_hidden[i]
            )
            pose_tokens = apply_conditional_dropout(
                pose_tokens, poses_keep_mask[i], null_tensor=self.null_pose_tokens[i]
            )

            t = t + pose_hiddens
            c.append(pose_tokens)

        # Concatenate pose tokens for self-attention
        c = torch.cat(c, dim=-2)

        # add time tokens to c
        if exists(time_tokens):
            c = torch.cat((c, time_tokens), dim=-2)

        # normalize conditioning tokens
        c = self.norm_cond(c)

        # DOWN
        garment_down_hiddens = []
        person_down_hiddens = []

        for garment_down_block, person_down_block in zip(self.garment_downs, self.person_downs):
            # garment
            garment_images, garment_hiddens = garment_down_block(garment_images, time_emb=t)
            garment_down_hiddens.extend(garment_hiddens)
            garment_hiddens = [g_hidden.clone() for g_hidden in garment_hiddens] if len(garment_hiddens) > 0 else None

            # person
            noisy_images, person_hiddens = person_down_block(
                noisy_images,
                time_emb=t,
                self_context=c,
                cross_context=garment_hiddens,
                cross_context_mask=garment_images_keep_mask,
            )
            person_down_hiddens.extend(person_hiddens)

        up_hiddens = []

        # UP
        for garment_up_block, person_up_block in zip(self.garment_ups, self.person_ups):
            # garment
            garment_hiddens = None
            if exists(garment_up_block):
                garment_images, garment_hiddens = garment_up_block(garment_images, garment_down_hiddens, time_emb=t)
                garment_hiddens = [g_hidden.clone() for g_hidden in garment_hiddens]

            # person

            noisy_images = person_up_block(
                noisy_images,
                person_down_hiddens,
                time_emb=t,
                self_context=c,
                cross_context=garment_hiddens,
                cross_context_mask=garment_images_keep_mask,
            )
            up_hiddens.append(noisy_images.contiguous())

        # whether to combine all feature maps from upsample blocks

        noisy_images = self.upsample_combiner(noisy_images, up_hiddens)

        # final top-most residual if needed

        if self.init_conv_to_final_conv_residual:
            noisy_images = torch.cat((noisy_images, init_conv_residual), dim=1)

        if exists(self.final_res_block):
            noisy_images = self.final_res_block(noisy_images, t)

        if exists(lowres_cond_img):
            noisy_images = torch.cat((noisy_images, lowres_cond_img), dim=1)

        return self.final_conv(noisy_images)


class NullUnet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lowres_cond = False
        self.dummy_parameter = nn.Parameter(torch.tensor([0.0]))

    def cast_model_parameters(self, *args, **kwargs):
        return self

    def forward(self, x, *args, **kwargs):
        return x


class TryOnImagen(nn.Module):
    def __init__(
        self,
        unets,
        *,
        image_sizes: Tuple[Tuple[int]] = ((144, 112), (288, 224)),  # (height, width)
        channels=3,
        timesteps=1000,
        cond_drop_prob=0.1,
        loss_type="l2",
        noise_schedules="cosine",
        pred_objectives: Literal["x_start", "noise", "v"] = ("noise", "noise"),
        aug_noise_schedule="linear",
        aug_sample_noise_level=0.2,  # in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
        per_sample_random_aug_noise_level=False,  # unclear when conditioning on augmentation noise level, whether each batch element receives a random aug noise value - turning off due to @marunine's find
        auto_normalize_img=False,  # whether to take care of normalizing the image from [0, 1] to [-1, 1]
        auto_unnormalize_img=True,  # whether to take care of unnormalizing the image from [-1, 1] to [0, 1] when sampling
        dynamic_thresholding=True,
        dynamic_thresholding_percentile=0.95,  # unsure what this was based on perusal of paper
        only_train_unet_number=None,
        resize_mode="nearest",
        min_snr_loss_weight=True,  # https://arxiv.org/abs/2303.09556
        min_snr_gamma=5,
    ):
        super().__init__()

        # loss

        if loss_type == "l1":
            loss_fn = F.l1_loss
        elif loss_type == "l2":
            loss_fn = F.mse_loss
        elif loss_type == "huber":
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # channels
        self.channels = channels

        # number of unets
        unets = cast_tuple(unets)
        num_unets = len(unets)

        # determine noise schedules per unet

        timesteps = cast_tuple(timesteps, num_unets)

        # make sure noise schedule defaults to 'cosine', 'cosine', and then 'linear' for rest of super-resoluting unets

        noise_schedules = cast_tuple(noise_schedules)
        noise_schedules = pad_tuple_to_length(noise_schedules, 2, "cosine")
        noise_schedules = pad_tuple_to_length(noise_schedules, num_unets, "linear")

        # construct noise schedulers

        noise_scheduler_klass = GaussianDiffusionContinuousTimes
        self.noise_schedulers = nn.ModuleList([])

        for timestep, noise_schedule in zip(timesteps, noise_schedules):
            noise_scheduler = noise_scheduler_klass(noise_schedule=noise_schedule, timesteps=timestep)
            self.noise_schedulers.append(noise_scheduler)

        # conditioning images augmentation noise schedule
        self.aug_noise_schedule = GaussianDiffusionContinuousTimes(noise_schedule=aug_noise_schedule)

        # ddpm objectives - predicting noise by default

        self.pred_objectives = cast_tuple(pred_objectives, num_unets)

        # construct unets

        self.unets = nn.ModuleList([])

        self.unet_being_trained_index = -1  # keeps track of which unet is being trained at the moment
        self.only_train_unet_number = only_train_unet_number

        for ind, one_unet in enumerate(unets):
            assert isinstance(one_unet, (ParallelUNet, NullUnet))
            is_first = ind == 0

            one_unet = one_unet.cast_model_parameters(
                lowres_cond=not is_first,
                channels=self.channels,
                channels_out=self.channels,
            )

            self.unets.append(one_unet)

        # unet image sizes
        self.image_sizes = image_sizes

        assert num_unets == len(
            image_sizes
        ), f"you did not supply the correct number of u-nets ({len(unets)}) for resolutions {image_sizes}"

        self.sample_channels = cast_tuple(self.channels, num_unets)

        self.right_pad_dims_to_datatype = partial(rearrange, pattern=("b -> b 1 1 1"))

        self.resize_to = resize_image_to
        self.resize_to = partial(self.resize_to, mode=resize_mode)

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (
            False,
            *((True,) * (num_unets - 1)),
        ), "the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True"

        self.aug_sample_noise_level = aug_sample_noise_level
        self.per_sample_random_aug_noise_level = per_sample_random_aug_noise_level

        # classifier free guidance

        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.0

        # normalize and unnormalize image functions

        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_unnormalize_img else identity
        self.input_image_range = (0.0 if auto_normalize_img else -1.0, 1.0)

        # dynamic thresholding

        self.dynamic_thresholding = cast_tuple(dynamic_thresholding, num_unets)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        # min snr loss weight

        min_snr_loss_weight = cast_tuple(min_snr_loss_weight, num_unets)
        min_snr_gamma = cast_tuple(min_snr_gamma, num_unets)

        assert len(min_snr_loss_weight) == len(min_snr_gamma) == num_unets
        self.min_snr_gamma = tuple(
            (gamma if use_min_snr else None) for use_min_snr, gamma in zip(min_snr_loss_weight, min_snr_gamma)
        )

        # one temp parameter for keeping track of device

        self.register_buffer("_temp", torch.tensor([0.0]), persistent=False)

        # default to device of unets passed in

        self.to(next(self.unets.parameters()).device)

    @property
    def device(self):
        return self._temp.device

    def get_unet(self, unet_number):
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.unets]
            delattr(self, "unets")
            self.unets = unets_list

        if index != self.unet_being_trained_index:
            for unet_index, unet in enumerate(self.unets):
                unet.to(self.device if unet_index == index else "cpu")

        self.unet_being_trained_index = index
        return self.unets[index]

    def reset_unets_all_one_device(self, device=None):
        device = default(device, self.device)
        self.unets = nn.ModuleList([*self.unets])
        self.unets.to(device)

        self.unet_being_trained_index = -1

    @contextmanager
    def one_unet_in_gpu(self, unet_number=None, unet=None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.unets[unet_number - 1]

        cpu = torch.device("cpu")

        devices = [module_device(unet) for unet in self.unets]

        self.unets.to(cpu)
        unet.to(self.device)

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    # overriding state dict functions

    def state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    # gaussian diffusion methods

    def p_mean_variance(
        self,
        unet,
        x,
        t,
        *,
        noise_scheduler,
        ca_images,
        garment_images,
        person_poses,
        garment_poses,
        ca_noise_times,
        garment_noise_times,
        lowres_cond_img=None,
        lowres_noise_times=None,
        cond_scale=2.0,
        model_output=None,
        t_next=None,
        pred_objective="noise",
        dynamic_threshold=True,
    ):
        # Assert statement to ensure correct usage of classifier guidance if applicable
        assert not (
            cond_scale != 1.0 and not self.can_classifier_guidance
        ), "imagen was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)"

        # Get model predictions (noise or start image) based on inputs and condition
        pred = default(
            model_output,
            lambda: unet.forward_with_cond_scale(
                x,
                noise_scheduler.get_condition(t),
                ca_images=ca_images,
                garment_images=garment_images,
                person_poses=person_poses,
                garment_poses=garment_poses,
                ca_noise_times=self.aug_noise_schedule.get_condition(ca_noise_times),
                garment_noise_times=self.aug_noise_schedule.get_condition(garment_noise_times),
                cond_scale=cond_scale,
                lowres_cond_img=lowres_cond_img,
                lowres_noise_times=self.aug_noise_schedule.get_condition(lowres_noise_times),
            ),
        )

        # Determine the start image (denoised image) based on the prediction objective
        if pred_objective == "noise":
            x_start = noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)
        elif pred_objective == "x_start":
            x_start = pred
        elif pred_objective == "v":
            x_start = noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        else:
            raise ValueError(f"unknown objective {pred_objective}")

        # Apply dynamic thresholding if enabled, to limit the range of pixel values
        if dynamic_threshold:
            s = torch.quantile(
                rearrange(x_start, "b ... -> b (...)").abs(), self.dynamic_thresholding_percentile, dim=-1
            )

            s.clamp_(min=1.0)
            s = right_pad_dims_to(x_start, s)
            x_start = x_start.clamp(-s, s) / s
        else:
            x_start.clamp_(-1.0, 1.0)

        # Calculate the mean and variance for the q-posterior distribution
        mean_and_variance = noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t, t_next=t_next)
        return mean_and_variance

    @torch.no_grad()
    def p_sample(
        self,
        unet,
        x,
        t,
        *,
        noise_scheduler,
        ca_images,
        garment_images,
        person_poses,
        garment_poses,
        ca_noise_times,
        garment_noise_times,
        t_next=None,
        cond_scale=2.0,
        lowres_cond_img=None,
        lowres_noise_times=None,
        pred_objective="noise",
        dynamic_threshold=True,
    ):
        b, *_, device = *x.shape, x.device

        # Calculate mean, variance from p_mean_variance function
        (model_mean, _, model_log_variance) = self.p_mean_variance(
            unet,
            x=x,
            t=t,
            t_next=t_next,
            noise_scheduler=noise_scheduler,
            ca_images=ca_images,
            garment_images=garment_images,
            person_poses=person_poses,
            garment_poses=garment_poses,
            ca_noise_times=ca_noise_times,
            garment_noise_times=garment_noise_times,
            cond_scale=cond_scale,
            lowres_cond_img=lowres_cond_img,
            lowres_noise_times=lowres_noise_times,
            pred_objective=pred_objective,
            dynamic_threshold=dynamic_threshold,
        )
        # Sample random noise with same shape as input image
        noise = torch.randn_like(x)

        # Check if current timestep is the last one; no noise added at t=0
        is_last_sampling_timestep = (
            (t_next == 0) if isinstance(noise_scheduler, GaussianDiffusionContinuousTimes) else (t == 0)
        )
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        # Add noise to the model mean, scaled by the model's variance
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred

    @torch.no_grad()
    def p_sample_loop(
        self,
        unet,
        shape,
        *,
        noise_scheduler,
        ca_images,
        garment_images,
        person_poses,
        garment_poses,
        ca_noise_times,
        garment_noise_times,
        lowres_cond_img,
        lowres_noise_times,
        cond_scale=2.0,
        pred_objective="noise",
        dynamic_threshold=True,
        use_tqdm=True,
    ):
        """The secondary sampling method which handles the sampling loop over a single unet in the cascade."""

        device = self.device

        batch = shape[0]
        images = torch.randn(shape, device=device)

        # time

        timesteps = noise_scheduler.get_sampling_timesteps(batch, device=device)

        for times, times_next in tqdm(
            timesteps, desc="sampling loop time step", total=len(timesteps), disable=not use_tqdm
        ):
            images = self.p_sample(
                unet,
                images,
                times,
                t_next=times_next,
                ca_images=ca_images,
                garment_images=garment_images,
                person_poses=person_poses,
                garment_poses=garment_poses,
                ca_noise_times=ca_noise_times,
                garment_noise_times=garment_noise_times,
                cond_scale=cond_scale,
                lowres_cond_img=lowres_cond_img,
                lowres_noise_times=lowres_noise_times,
                noise_scheduler=noise_scheduler,
                pred_objective=pred_objective,
                dynamic_threshold=dynamic_threshold,
            )

        images.clamp_(-1.0, 1.0)

        unnormalize_images = self.unnormalize_img(images)
        return unnormalize_images

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        *,
        ca_images,
        garment_images,
        person_poses,
        garment_poses,
        batch_size=1,
        cond_scale=2.0,
        aug_sample_noise_level=None,
        start_at_unet_number=1,
        start_image_or_video=None,
        stop_at_unet_number=None,
        return_all_unet_outputs=False,
        return_pil_images=False,
        device=None,
        use_tqdm=True,
        use_one_unet_in_gpu=True,
    ):
        """The primary sampling method which handles the sampling loop over the cascade of unets."""

        device = default(device, self.device)
        self.reset_unets_all_one_device(device=device)

        outputs = []

        is_cuda = next(self.parameters()).is_cuda
        device = next(self.parameters()).device

        aug_sample_noise_level = default(aug_sample_noise_level, self.aug_sample_noise_level)

        num_unets = len(self.unets)

        # condition scaling

        cond_scale = cast_tuple(cond_scale, num_unets)

        # handle starting at a unet greater than 1, for training only-upscaler training

        if start_at_unet_number > 1:
            assert start_at_unet_number <= num_unets, "must start a unet that is less than the total number of unets"
            assert not exists(stop_at_unet_number) or start_at_unet_number <= stop_at_unet_number
            assert exists(start_image_or_video), "starting image or video must be supplied if only doing upscaling"

        # go through each unet in cascade

        for (
            unet_number,
            unet,
            channel,
            image_size,
            noise_scheduler,
            pred_objective,
            dynamic_threshold,
            unet_cond_scale,
        ) in tqdm(
            zip(
                range(1, num_unets + 1),
                self.unets,
                self.sample_channels,
                self.image_sizes,
                self.noise_schedulers,
                self.pred_objectives,
                self.dynamic_thresholding,
                cond_scale,
            ),
            disable=not use_tqdm,
        ):
            if unet_number < start_at_unet_number:
                continue

            assert not isinstance(unet, NullUnet), "one cannot sample from null / placeholder unets"

            context = self.one_unet_in_gpu(unet=unet) if is_cuda and use_one_unet_in_gpu else nullcontext()

            with context:
                # low resolution conditioning

                lowres_cond_img = lowres_noise_times = None
                shape = (batch_size, channel, *image_size)

                if unet.lowres_cond:
                    lowres_noise_times = self.aug_noise_schedule.get_times(
                        batch_size, aug_sample_noise_level, device=device
                    )

                    lowres_cond_img = self.resize_to(img, image_size)  # noqa: F821 - img is defined in the loop below

                    lowres_cond_img = self.normalize_img(lowres_cond_img)
                    lowres_cond_img, *_ = self.aug_noise_schedule.q_sample(
                        x_start=lowres_cond_img, t=lowres_noise_times, noise=torch.randn_like(lowres_cond_img)
                    )

                # prepare the clothing-agnostic image shape noise augmentation conditioning
                ca_images_resized = self.resize_to(ca_images, image_size)
                ca_noise_times = self.aug_noise_schedule.get_times(batch_size, aug_sample_noise_level, device=device)
                ca_images_resized, *_ = self.aug_noise_schedule.q_sample(
                    x_start=ca_images_resized, t=ca_noise_times, noise=torch.randn_like(ca_images_resized)
                )
                # prepare the garment image shape and noise augmentation conditioning
                garment_images_resized = self.resize_to(garment_images, image_size)
                garment_noise_times = self.aug_noise_schedule.get_times(
                    batch_size, aug_sample_noise_level, device=device
                )
                garment_images_resized, *_ = self.aug_noise_schedule.q_sample(
                    x_start=garment_images_resized,
                    t=garment_noise_times,
                    noise=torch.randn_like(garment_images_resized),
                )

                # shape of stage

                shape = (batch_size, self.channels, *image_size)

                img = self.p_sample_loop(
                    unet,
                    shape,
                    ca_images=ca_images_resized,
                    garment_images=garment_images_resized,
                    person_poses=person_poses,
                    garment_poses=garment_poses,
                    ca_noise_times=ca_noise_times,
                    garment_noise_times=garment_noise_times,
                    cond_scale=unet_cond_scale,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_times=lowres_noise_times,
                    noise_scheduler=noise_scheduler,
                    pred_objective=pred_objective,
                    dynamic_threshold=dynamic_threshold,
                    use_tqdm=use_tqdm,
                )

                outputs.append(img)

            if exists(stop_at_unet_number) and stop_at_unet_number == unet_number:
                break

        output_index = (
            -1 if not return_all_unet_outputs else slice(None)
        )  # either return last unet output or all unet outputs

        if not return_pil_images:
            return outputs[output_index]

        if not return_all_unet_outputs:
            outputs = outputs[-1:]

        pil_images = list(map(lambda img: list(map(T.ToPILImage(), img.unbind(dim=0))), outputs))

        return pil_images[
            output_index
        ]  # now you have a bunch of pillow images you can just .save(/where/ever/you/want.png)

    def p_losses(
        self,
        unet: Union[ParallelUNet, NullUnet, DistributedDataParallel],
        x_start,
        times,
        *,
        noise_scheduler,
        lowres_cond_img=None,
        lowres_aug_times=None,
        ca_images=None,
        noise=None,
        person_poses=None,
        garment_poses=None,
        garment_images=None,
        ca_aug_times=None,
        garment_aug_times=None,
        pred_objective="noise",
        min_snr_gamma=None,
        **kwargs,
    ):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # normalize to [-1, 1]

        x_start = self.normalize_img(x_start)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # get x_t

        x_noisy, log_snr, alpha, sigma = noise_scheduler.q_sample(x_start=x_start, t=times, noise=noise)

        # also noise the lowres conditioning image
        # at sample time, they then fix the noise level of 0.1 - 0.3

        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_aug_times = default(lowres_aug_times, times)
            lowres_cond_img_noisy, *_ = self.aug_noise_schedule.q_sample(
                x_start=lowres_cond_img, t=lowres_aug_times, noise=torch.randn_like(lowres_cond_img)
            )

        # Noise the clothing-agnostic and garment images
        ca_images, *_ = self.aug_noise_schedule.q_sample(
            x_start=ca_images, t=ca_aug_times, noise=torch.randn_like(ca_images)
        )
        garment_images, *_ = self.aug_noise_schedule.q_sample(
            x_start=garment_images, t=garment_aug_times, noise=torch.randn_like(garment_images)
        )

        # time condition

        noise_cond = noise_scheduler.get_condition(times)

        # unet kwargs
        unet_kwargs = dict(
            person_poses=person_poses,
            garment_poses=garment_poses,
            #
            ca_images=ca_images,
            ca_noise_times=self.aug_noise_schedule.get_condition(ca_aug_times),
            #
            lowres_cond_img=lowres_cond_img_noisy,
            lowres_noise_times=self.aug_noise_schedule.get_condition(lowres_aug_times),
            #
            garment_images=garment_images,
            garment_noise_times=self.aug_noise_schedule.get_condition(garment_aug_times),
            #
            cond_drop_prob=self.cond_drop_prob,
            **kwargs,
        )

        # get prediction

        pred = unet.forward(x_noisy, noise_cond, **unet_kwargs)

        # prediction objective

        if pred_objective == "noise":
            target = noise
        elif pred_objective == "x_start":
            target = x_start
        elif pred_objective == "v":
            # derivation detailed in Appendix D of Progressive Distillation paper
            # https://arxiv.org/abs/2202.00512
            # this makes distillation viable as well as solve an issue with color shifting in upresoluting unets, noted in imagen-video
            target = alpha * noise - sigma * x_start
        else:
            raise ValueError(f"unknown objective {pred_objective}")

        # losses

        losses = self.loss_fn(pred, target, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")

        # min snr loss reweighting

        snr = log_snr.exp()
        maybe_clipped_snr = snr.clone()

        if exists(min_snr_gamma):
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if pred_objective == "noise":
            loss_weight = maybe_clipped_snr / snr
        elif pred_objective == "x_start":
            loss_weight = maybe_clipped_snr
        elif pred_objective == "v":
            loss_weight = maybe_clipped_snr / (snr + 1)

        losses = losses * loss_weight
        return losses.mean()

    def forward(
        self,
        unet: Union[ParallelUNet, NullUnet, DistributedDataParallel] = None,
        *,
        person_images,
        ca_images,
        garment_images,
        person_poses,
        garment_poses,
        unet_number=None,
        **kwargs,
    ):
        assert not (
            len(self.unets) > 1 and not exists(unet_number)
        ), f"you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)"
        unet_number = default(unet_number, 1)
        assert (
            not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number
        ), "you can only train on unet #{self.only_train_unet_number}"

        person_images = cast_uint8_images_to_float(person_images)
        ca_images = cast_uint8_images_to_float(ca_images)
        garment_images = cast_uint8_images_to_float(garment_images)

        assert (
            person_images.dtype == torch.float or person_images.dtype == torch.half
        ), f"images tensor needs to be floats but {person_images.dtype} dtype found instead"

        unet_index = unet_number - 1

        unet = default(unet, lambda: self.get_unet(unet_number))

        assert not isinstance(unet, NullUnet), "null unet cannot and should not be trained"

        noise_scheduler = self.noise_schedulers[unet_index]
        min_snr_gamma = self.min_snr_gamma[unet_index]
        pred_objective = self.pred_objectives[unet_index]
        target_image_size = self.image_sizes[unet_index]
        prev_image_size = self.image_sizes[unet_index - 1] if unet_index > 0 else None

        b, c, *_, h, w, device = *person_images.shape, person_images.device

        assert person_images.shape[1] == self.channels
        assert h >= target_image_size[0] and w >= target_image_size[1]

        times = noise_scheduler.sample_random_times(b, device=device)

        # handle image resolutions in case there are multiple unets
        lowres_cond_img = lowres_aug_times = None
        if exists(prev_image_size):
            lowres_cond_img = self.resize_to(person_images, prev_image_size, clamp_range=self.input_image_range)
            lowres_cond_img = self.resize_to(
                lowres_cond_img,
                target_image_size,
                clamp_range=self.input_image_range,
            )

        person_images = self.resize_to(person_images, target_image_size)
        ca_images = self.resize_to(ca_images, target_image_size)
        garment_images = self.resize_to(garment_images, target_image_size)

        # handle lowres, clothing-agnostic and garment images noise augmentation conditioning
        lowres_aug_times = self.sample_random_aug_times(self.aug_noise_schedule, b, device)
        ca_aug_times = self.sample_random_aug_times(self.aug_noise_schedule, b, device)
        garment_aug_times = self.sample_random_aug_times(self.aug_noise_schedule, b, device)

        return self.p_losses(
            unet,
            person_images,
            times,
            person_poses=person_poses,
            garment_poses=garment_poses,
            ca_images=ca_images,
            garment_images=garment_images,
            noise_scheduler=noise_scheduler,
            lowres_cond_img=lowres_cond_img,
            lowres_aug_times=lowres_aug_times,
            ca_aug_times=ca_aug_times,
            garment_aug_times=garment_aug_times,
            pred_objective=pred_objective,
            min_snr_gamma=min_snr_gamma,
            **kwargs,
        )

    def sample_random_aug_times(self, schedule, batch_size, device):
        if self.per_sample_random_aug_noise_level:
            return schedule.sample_random_times(batch_size, device=device)
        else:
            single_time = schedule.sample_random_times(1, device=device)
            return repeat(single_time, "1 -> b", b=batch_size)


class BaseParallelUnet(ParallelUNet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            image_size=(128, 128),
            dim=128,
            context_dim=512,
            time_cond_dim=512,
            pose_depth=2,
            feature_channels=(128, 256, 512, 1024),
            num_blocks=(3, 4, 6, 7),
            layer_attns=(False, False, True, True),
            layer_cross_attns=(False, False, True, True),
            attn_heads=(None, None, 4, 8),
            attn_dim_head=(None, None, 128, 64),
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})


class SRParallelUnet(ParallelUNet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            image_size=(256, 256),
            dim=128,
            context_dim=512,
            time_cond_dim=768,
            lowres_cond=True,
            pose_depth=2,
            feature_channels=(128, 128, 256, 512, 1024),
            num_blocks=(2, 3, 4, 7, 7),
            layer_attns=(False, False, False, False, True),
            layer_cross_attns=(False, False, False, False, True),
            attn_heads=(None, None, None, None, 8),
            attn_dim_head=(None, None, None, None, 64),
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})


def get_unet_by_name(name: str, **model_kwargs):
    if name == "base":
        return BaseParallelUnet(**model_kwargs)
    elif name == "sr":
        return SRParallelUnet(**model_kwargs)
    else:
        raise ValueError(f"Invalid unet name: {name}")
