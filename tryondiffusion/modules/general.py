import torch
from einops import rearrange
from torch import einsum, nn

from tryondiffusion.common.python_helpers import exists


def FeedForward(dim, mult=2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias=False),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias=False),
    )


class StableLayerNorm(nn.Module):
    def __init__(self, feats, stable=False, dim=-1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(dim=dim, keepdim=True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=dim, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=dim, keepdim=True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)


# helper classes


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class Always:
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class GlobalContext(nn.Module):
    """basically a superior form of squeeze-excitation that is attention-esque"""

    def __init__(self, *, dim_in, dim_out):
        super().__init__()
        self.to_k = nn.Conv2d(dim_in, 1, 1)
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, 1), nn.SiLU(), nn.Conv2d(hidden_dim, dim_out, 1), nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x)
        x, context = map(lambda t: rearrange(t, "b n ... -> b n (...)"), (x, context))
        out = einsum("b i n, b c n -> b c i", context.softmax(dim=-1), x)
        out = rearrange(out, "... -> ... 1")
        return self.net(out)


def apply_conditional_dropout(tensor, mask, null_tensor=None):
    device, dtype = tensor.device, tensor.dtype
    mask_shape = [mask.shape[0]] + [1] * (tensor.dim() - 1)
    keep_mask = mask.view(*mask_shape)

    if exists(null_tensor):
        null_tensor = null_tensor.to(device=device, dtype=dtype)
        null_tensor = null_tensor.expand_as(tensor)
    else:
        null_tensor = torch.zeros_like(tensor)

    return torch.where(keep_mask, tensor, null_tensor)
