from typing import Optional

from einops import rearrange
from torch import nn

from tryondiffusion.common.python_helpers import exists
from tryondiffusion.modules.attention import MultiHeadAttention, MultiHeadCrossAttention
from tryondiffusion.modules.general import (
    Always,
    GlobalContext,
    Identity,
    apply_conditional_dropout,
)


class AdaGN(nn.Module):
    def __init__(self, dim_in, time_cond_dim: Optional[int] = None, groups: Optional[int] = None, zero_init=True):
        super().__init__()
        if not exists(groups):
            groups = min(32, dim_in // 4)

        self.groupnorm = nn.GroupNorm(groups, dim_in) if groups is not None else Identity()

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_cond_dim, dim_in * 2))

        # Optionally zero the weight and bias of the time_mlp
        if zero_init:
            nn.init.zeros_(self.time_mlp[1].weight)
            nn.init.zeros_(self.time_mlp[1].bias)

    def forward(self, x, time_emb=None):
        x = self.groupnorm(x)

        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale, shift = time_emb.chunk(2, dim=1)
            x = x * (scale + 1) + shift

        return x


class ConditionedResnetBlock(nn.Module):
    def __init__(
        self, dim_in, dim_out, *, time_cond_dim=None, use_gca=False, zero_init=False, dropout_prob: float = 0.0
    ):
        super().__init__()

        self.ada_gn1 = AdaGN(dim_in, time_cond_dim=time_cond_dim, zero_init=zero_init)
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.ada_gn2 = AdaGN(dim_out, time_cond_dim=time_cond_dim, zero_init=zero_init)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        self.dropout2 = nn.Dropout(dropout_prob)

        if zero_init:
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)

        self.activation = nn.SiLU()
        self.gca = GlobalContext(dim_in=dim_out, dim_out=dim_out) if use_gca else Always(1)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else Identity()

    def forward(self, x, time_emb=None):
        res = self.res_conv(x)

        h = self.ada_gn1(x, time_emb=time_emb)
        h = self.activation(h)
        h = self.conv1(h)
        h = self.dropout1(h)

        h = self.ada_gn2(h, time_emb=time_emb)
        h = self.activation(h)
        h = self.conv2(h)
        h = self.dropout2(h)

        h = h * self.gca(h)

        return h + res


class ConditionedSelfAttention2D(nn.Module):
    def __init__(
        self,
        dim,
        *,
        time_cond_dim=None,
        groups=None,
        heads=8,
        dim_head=64,
        context_dim=None,
        patch_size=1,
        zero_init=True,
    ):
        super().__init__()

        # AdaGN Layer
        self.adagn = AdaGN(dim, time_cond_dim=time_cond_dim, groups=groups, zero_init=zero_init)
        self.patch_size = patch_size
        # Adjust dimension if patching is used
        patch_dim = dim * (patch_size**2)

        # Attention Layer with support for null key values and additional context
        self.attention = MultiHeadAttention(
            patch_dim, dim_head=dim_head, heads=heads, context_dim=context_dim, zero_init=zero_init
        )

    def forward(self, x, time_emb=None, context=None):
        # IMPORTANT: context should already be normalized at this stage
        x = self.adagn(x, time_emb=time_emb)

        b, c, h, w = x.shape

        # Patching and rearranging
        if self.patch_size > 1:
            x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.patch_size, p2=self.patch_size)
        else:
            x = rearrange(x, "b c h w -> b (h w) c")

        # Apply Attention
        x = self.attention(x, context=context) + x

        # Unpatch and Rearrange x back to image format
        if self.patch_size > 1:
            x = rearrange(
                x,
                "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                p1=self.patch_size,
                p2=self.patch_size,
                h=h // self.patch_size,
                w=w // self.patch_size,
            )
        else:
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return x


class ConditionedCrossAttention2D(nn.Module):
    def __init__(self, dim, *, time_cond_dim=None, groups=None, heads=8, dim_head=64, patch_size=1, zero_init=False):
        super().__init__()

        # AdaGN Layers for x and context
        self.adagn_x = AdaGN(dim, time_cond_dim=time_cond_dim, groups=groups, zero_init=zero_init)
        self.adagn_context = AdaGN(dim, time_cond_dim=time_cond_dim, groups=groups, zero_init=zero_init)

        self.patch_size = patch_size
        # Adjust dimension if patching is used
        patch_dim = dim * (patch_size**2)

        # Cross-Attention Layer
        self.cross_attention = MultiHeadCrossAttention(
            patch_dim, heads=heads, dim_head=dim_head, context_dim=patch_dim, zero_init=zero_init
        )

    def forward(self, x, context, time_emb=None, mask=None):
        # Apply AdaGN to both inputs
        x = self.adagn_x(x, time_emb=time_emb)
        context = self.adagn_context(context, time_emb=time_emb)

        b, c, h, w = x.shape

        # Patching and rearranging
        if self.patch_size > 1:
            x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.patch_size, p2=self.patch_size)
            context = rearrange(
                context, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.patch_size, p2=self.patch_size
            )
        else:
            x = rearrange(x, "b c h w -> b (h w) c")
            context = rearrange(context, "b c h w -> b (h w) c")

        # Apply Cross-Attention
        if exists(mask):
            context = apply_conditional_dropout(context, mask)
        x = self.cross_attention(x, context=context) + x

        # Unpatch and Rearrange x back to image format
        if self.patch_size > 1:
            x = rearrange(
                x,
                "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                p1=self.patch_size,
                p2=self.patch_size,
                h=h // self.patch_size,
                w=w // self.patch_size,
            )
        else:
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return x
