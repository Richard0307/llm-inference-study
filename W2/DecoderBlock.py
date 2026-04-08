import torch
import torch.nn as nn
from PreNorm import PreNorm


class CausalSelfAttention(nn.Module):
    """Self-Attention with causal mask, wraps q=k=v and mask logic"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(self, x):
        seq_len = x.size(1)
        # Causal mask: prevent attending to future tokens
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out


class FeedForward(nn.Module):
    """GPT-2 style FFN: Linear -> GELU -> Linear"""
    def __init__(self, hidden_size, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * expansion),
            nn.GELU(),
            nn.Linear(hidden_size * expansion, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    """Single GPT-style Decoder Block: PreNorm(Attn) + PreNorm(FFN)"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn_block = PreNorm(hidden_size, fn=CausalSelfAttention(hidden_size, num_heads))
        self.ffn_block = PreNorm(hidden_size, fn=FeedForward(hidden_size))

    def forward(self, x):
        x = self.attn_block(x)
        x = self.ffn_block(x)
        return x


if __name__ == "__main__":
    hidden_size = 512
    num_heads = 8
    N = 4
    batch, seq_len = 2, 10
    x = torch.randn(batch, seq_len, hidden_size)

    # N-layer Decoder
    blocks = nn.ModuleList([DecoderBlock(hidden_size, num_heads) for _ in range(N)])

    out = x
    for i, block in enumerate(blocks):
        out = block(out)
        print(f"Block {i+1} | shape: {out.shape}, mean: {out.mean():.4f}, std: {out.std():.4f}")

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Shape match:  {x.shape == out.shape}")

    # Verify causal mask: position 0 output should not be affected by position 1+
    x_test = torch.randn(1, 5, hidden_size)
    block = DecoderBlock(hidden_size, num_heads)
    block.eval()

    out_full = block(x_test)               # all 5 tokens
    out_short = block(x_test[:, :1, :])     # only first token

    diff = (out_full[:, 0, :] - out_short[:, 0, :]).abs().max().item()
    print(f"\nCausal mask verification:")
    print(f"  Position 0 diff: {diff:.2e}")
    print(f"  Causal mask effective: {diff < 1e-5}")

# 为什么Decoder Only 会胜出呢？
# Decoder Only 模型在自然语言处理任务中表现出色，主要原因有以下几点：
