import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from tryondiffusion.common.python_helpers import default, exists


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, context_dim=None, zero_init=False):
        super().__init__()
        self.heads = heads

        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.context_proj = nn.Linear(context_dim, inner_dim * 2) if exists(context_dim) else None

        # Initialize null key and value for each head
        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_out = nn.Linear(inner_dim, dim)
        if zero_init:
            nn.init.zeros_(self.to_out.weight)
            nn.init.zeros_(self.to_out.bias)

    def forward(self, x, context=None):
        # IMPORTANT: x and context are expected to be normalized
        b, n, _ = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # Null key/value for classifier free guidance
        nk, nv = self.null_kv.unbind(dim=0)
        nk, nv = repeat(nk, "h d -> b h () d", b=b), repeat(nv, "h d -> b h () d", b=b)
        k, v = torch.cat((nk, k), dim=-2), torch.cat((nv, v), dim=-2)

        if exists(context) and exists(self.context_proj):
            ck, cv = self.context_proj(context).chunk(2, dim=-1)
            ck, cv = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (ck, cv))
            k, v = torch.cat((ck, k), dim=-2), torch.cat((cv, v), dim=-2)

        # Scaled dot product attention
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = rearrange(attn, "b h n d -> b n (h d)")

        return self.to_out(attn)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, zero_init=False):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        context_dim = default(context_dim, dim)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_out = nn.Linear(inner_dim, dim)
        if zero_init:
            nn.init.zeros_(self.to_out.weight)
            nn.init.zeros_(self.to_out.bias)

    def forward(self, x, context):
        b, n, _ = x.shape
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)

        k, v = self.to_kv(context).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (k, v))

        # Null key/value for classifier free guidance
        nk, nv = self.null_kv.unbind(dim=0)
        nk, nv = repeat(nk, "h d -> b h () d", b=b), repeat(nv, "h d -> b h () d", b=b)
        k, v = torch.cat((nk, k), dim=-2), torch.cat((nv, v), dim=-2)

        # Scaled dot product attention
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = rearrange(attn, "b h n d -> b n (h d)")

        return self.to_out(attn)
