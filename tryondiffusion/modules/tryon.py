from functools import partial
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tryondiffusion.common.python_helpers import exists
from tryondiffusion.modules.conditional import (
    ConditionedCrossAttention2D,
    ConditionedResnetBlock,
    ConditionedSelfAttention2D,
)
from tryondiffusion.modules.general import Identity
from tryondiffusion.modules.imagen import (
    CrossEmbedLayer,
    Downsample,
    Parallel,
    PixelShuffleUpsample,
    Upsample,
)


class PersonDownBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        time_cond_dim,
        num_blocks,
        context_dim=None,
        use_gca=False,
        is_last=False,
        zero_init=False,
        cross_embed_downsample=False,
        cross_embed_downsample_kernel_sizes=(2, 4),
        do_attn=False,
        do_cross_attn=False,
        patch_size=1,
        dropout_prob=0.0,
        attn_dim_head=None,
        attn_num_heads=None,
    ):
        super().__init__()

        if do_attn:
            assert exists(context_dim), "Context dim must be specified if attention is enabled."

        self.layers = nn.ModuleList([])

        for _ in range(num_blocks):
            self.layers.append(
                nn.ModuleList(
                    [
                        # Main block FiLM + ResBlk + self-attn + cross-attn
                        ConditionedResnetBlock(
                            dim_in=dim_in,
                            dim_out=dim_in,
                            time_cond_dim=time_cond_dim,
                            use_gca=use_gca,
                            zero_init=zero_init,
                            dropout_prob=dropout_prob,
                        ),
                        (
                            ConditionedSelfAttention2D(
                                dim=dim_in,
                                patch_size=patch_size,
                                time_cond_dim=time_cond_dim,
                                context_dim=context_dim,
                                zero_init=zero_init,
                                dim_head=attn_dim_head,
                                heads=attn_num_heads,
                            )
                            if do_attn
                            else Identity()
                        ),
                        (
                            ConditionedCrossAttention2D(
                                dim=dim_in,
                                patch_size=patch_size,
                                time_cond_dim=time_cond_dim,
                                zero_init=zero_init,
                                dim_head=attn_dim_head,
                                heads=attn_num_heads,
                            )
                            if do_cross_attn
                            else Identity()
                        ),
                    ]
                )
            )
        # Downsample
        downsample_klass = Downsample
        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes=cross_embed_downsample_kernel_sizes)

        if is_last:
            self.downsample = Parallel(nn.Conv2d(dim_in, dim_out, 3, padding=1), nn.Conv2d(dim_in, dim_out, 1))
        else:
            self.downsample = downsample_klass(dim=dim_in, dim_out=dim_out)

    def forward(
        self,
        x,
        time_emb=None,
        self_context=None,
        cross_context: Optional[List[torch.Tensor]] = None,
        cross_context_mask=None,
    ):
        hiddens = []
        for resnet_block, self_attn, cross_attn in self.layers:
            x = resnet_block(x, time_emb=time_emb)

            x = self_attn(x, context=self_context, time_emb=time_emb)

            cross_c = cross_context.pop(0) if exists(cross_context) else None
            x = cross_attn(x, context=cross_c, mask=cross_context_mask, time_emb=time_emb)

            hiddens.append(x)

        x = self.downsample(x)
        return x, hiddens


class PersonUpBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        skip_dim,
        time_cond_dim,
        num_blocks,
        context_dim=None,
        use_gca=False,
        zero_init=False,
        is_last=False,
        pixel_shuffle_upsample=True,
        scale_skip_connection=True,
        do_attn=False,
        do_cross_attn=False,
        patch_size=1,
        dropout_prob=0.0,
        attn_dim_head=None,
        attn_num_heads=None,
    ):
        super().__init__()

        if do_attn:
            assert exists(context_dim), "Context dim must be specified if attention is enabled."

        self.skip_connect_scale = 1.0 if not scale_skip_connection else (2**-0.5)

        self.layers = nn.ModuleList([])

        for _ in range(num_blocks):
            self.layers.append(
                nn.ModuleList(
                    [
                        # Main block FiLM + ResBlk + self-attn + cross-attn
                        ConditionedResnetBlock(
                            dim_in=dim_in + skip_dim,
                            dim_out=dim_in,
                            time_cond_dim=time_cond_dim,
                            use_gca=use_gca,
                            zero_init=zero_init,
                            dropout_prob=dropout_prob,
                        ),
                        (
                            ConditionedSelfAttention2D(
                                dim=dim_in,
                                patch_size=patch_size,
                                time_cond_dim=time_cond_dim,
                                context_dim=context_dim,
                                zero_init=zero_init,
                                dim_head=attn_dim_head,
                                heads=attn_num_heads,
                            )
                            if do_attn
                            else Identity()
                        ),
                        (
                            ConditionedCrossAttention2D(
                                dim=dim_in,
                                patch_size=patch_size,
                                time_cond_dim=time_cond_dim,
                                zero_init=zero_init,
                                dim_head=attn_dim_head,
                                heads=attn_num_heads,
                            )
                            if do_cross_attn
                            else Identity()
                        ),
                    ]
                )
            )

        # Upsample
        upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

        self.upsample = upsample_klass(dim_in, dim_out) if not is_last else Identity()

    def forward(
        self,
        x,
        hiddens,
        time_emb=None,
        self_context=None,
        cross_context: Optional[List[torch.Tensor]] = None,
        cross_context_mask=None,
    ):
        for resnet_block, self_attn, cross_attn in self.layers:
            x = torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim=1)
            x = resnet_block(x, time_emb=time_emb)

            x = self_attn(x, context=self_context, time_emb=time_emb)

            cross_c = cross_context.pop(0) if exists(cross_context) else None
            x = cross_attn(x, context=cross_c, mask=cross_context_mask, time_emb=time_emb)

        x = self.upsample(x)
        return x


class GarmentDownBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        time_cond_dim,
        num_blocks,
        output_hiddens=False,
        use_gca=False,
        zero_init=False,
        is_last=False,
        cross_embed_downsample=False,
        cross_embed_downsample_kernel_sizes=(2, 4),
        dropout_prob=0.0,
    ):
        super().__init__()

        # Output hiddens for both garment skip connections and person cross-attention
        self.output_hiddens = output_hiddens

        self.layers = nn.ModuleList([])

        for _ in range(num_blocks):
            self.layers.append(
                # Main block FiLM + ResBlk
                ConditionedResnetBlock(
                    dim_in=dim_in,
                    dim_out=dim_in,
                    time_cond_dim=time_cond_dim,
                    use_gca=use_gca,
                    zero_init=zero_init,
                    dropout_prob=dropout_prob,
                )
            )

        # Downsample
        downsample_klass = Downsample

        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes=cross_embed_downsample_kernel_sizes)

        if is_last:
            self.downsample = Parallel(nn.Conv2d(dim_in, dim_out, 3, padding=1), nn.Conv2d(dim_in, dim_out, 1))
        else:
            self.downsample = downsample_klass(dim=dim_in, dim_out=dim_out)

    def forward(self, x, time_emb=None):
        hiddens = []
        for resnet_block in self.layers:
            x = resnet_block(x, time_emb=time_emb)
            if self.output_hiddens:
                hiddens.append(x)

        x = self.downsample(x)
        return x, hiddens


class GarmentUpBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        skip_dim,
        time_cond_dim,
        num_blocks,
        output_hiddens=True,
        use_gca=False,
        zero_init=False,
        is_last=False,
        pixel_shuffle_upsample=True,
        scale_skip_connection=True,
        dropout_prob=0.0,
    ):
        super().__init__()

        # Output hiddens for both garment skip connections and person cross-attention
        self.output_hiddens = output_hiddens

        self.skip_connect_scale = 1.0 if not scale_skip_connection else (2**-0.5)

        # Main block FiLM + ResBlk
        self.layers = nn.ModuleList([])

        for _ in range(num_blocks):
            self.layers.append(
                # Main block FiLM + ResBlk
                ConditionedResnetBlock(
                    dim_in=dim_in + skip_dim,
                    dim_out=dim_in,
                    time_cond_dim=time_cond_dim,
                    use_gca=use_gca,
                    zero_init=zero_init,
                    dropout_prob=dropout_prob,
                )
            )

        # Upsample
        upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

        self.upsample = upsample_klass(dim_in, dim_out) if not is_last else Identity()

    def forward(self, x, hiddens, time_emb=None):
        up_hiddens = []
        for resnet_block in self.layers:
            x = torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim=1)
            x = resnet_block(x, time_emb=time_emb)
            if self.output_hiddens:
                up_hiddens.append(x)

        x = self.upsample(x)
        return x, up_hiddens


def resize_image_to(image, target_image_size, clamp_range=None, mode="nearest"):
    # Assuming image is in format [batch_size, channels, height, width]
    orig_height, orig_width = image.shape[-2], image.shape[-1]
    target_height, target_width = target_image_size

    # Check if the original size equals the target size
    if (orig_height, orig_width) == (target_height, target_width):
        return image

    # F.interpolate expects size as (height, width) for 2D images
    out = F.interpolate(image, size=(target_height, target_width), mode=mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out
