import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x + self.bias


if __name__ == "__main__":
    hidden_size = 512
    batch, seq_len = 2, 10
    x = torch.randn(batch, seq_len, hidden_size)

    # Official implementation for reference
    ref = nn.LayerNorm(hidden_size, eps=1e-12)

    # Custom implementation
    custom = LayerNorm(hidden_size, eps=1e-12)
    custom.weight = ref.weight
    custom.bias = ref.bias

    out_ref    = ref(x)
    out_custom = custom(x)

    max_diff = (out_ref - out_custom).abs().max().item()
    print(f"Maximum difference: {max_diff:.2e}")
    print(f"Values are close: {torch.allclose(out_ref, out_custom, atol=1e-5)}")

