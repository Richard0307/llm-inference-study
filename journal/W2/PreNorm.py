import torch
import torch.nn as nn
from LayerNorm import LayerNorm


class PreNorm(nn.Module):
    """Generic Pre-Norm wrapper, agnostic to fn's interface"""
    def __init__(self, hidden_size, fn):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(self.norm(x), **kwargs)


if __name__ == "__main__":
    hidden_size = 512
    batch, seq_len = 2, 10
    x = torch.randn(batch, seq_len, hidden_size)

    fn = nn.Linear(hidden_size, hidden_size)

    # Official Pre-Norm
    ref_norm = nn.LayerNorm(hidden_size, eps=1e-12)
    out_ref = x + fn(ref_norm(x))

    # Custom Pre-Norm, align weights
    custom = PreNorm(hidden_size, fn=fn)
    custom.norm.weight = ref_norm.weight
    custom.norm.bias = ref_norm.bias
    out_custom = custom(x)

    max_diff = (out_ref - out_custom).abs().max().item()
    print(f"PreNorm max diff: {max_diff:.2e}")
    print(f"Values match: {torch.allclose(out_ref, out_custom, atol=1e-5)}")
