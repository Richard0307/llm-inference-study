"""
Roofline analysis of GPT-2 inference on RTX 4090 Laptop.
Explains W2 profiling data: why is decode slow?
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# ── RTX 4090 Laptop GPU specs ──
PEAK_FLOPS = 33e12        # 33 TFLOPs FP32
PEAK_BW    = 576e9        # 576 GB/s HBM
RIDGE      = PEAK_FLOPS / PEAK_BW   # FLOPs/byte breakpoint

# ── GPT-2 model ──
PARAMS = 124e6
WEIGHT_BYTES = PARAMS * 4  # FP32

def compute_ai_decode():
    """Decode step: 1 token through all weights"""
    flops = 2 * PARAMS               # 1 MAC per param = 2 FLOPs
    bytes_loaded = WEIGHT_BYTES      # load all weights once
    ai = flops / bytes_loaded
    return flops, bytes_loaded, ai

def compute_ai_prefill(seq_len):
    """Prefill step: seq_len tokens through all weights"""
    flops = 2 * PARAMS * seq_len
    bytes_loaded = WEIGHT_BYTES      # weights loaded once, reused across seq
    ai = flops / bytes_loaded
    return flops, bytes_loaded, ai

def roofline_peak(ai):
    """Maximum achievable FLOPs at given arithmetic intensity"""
    return min(PEAK_FLOPS, ai * PEAK_BW)


def main():
    # Load W2 profiling results
    with open("W2/gpt2_profiling_results.json") as f:
        results = json.load(f)

    print("=" * 78)
    print("ROOFLINE ANALYSIS: GPT-2 on RTX 4090 Laptop")
    print("=" * 78)
    print(f"Peak FP32 compute: {PEAK_FLOPS/1e12:.1f} TFLOPs")
    print(f"Peak HBM bandwidth: {PEAK_BW/1e9:.0f} GB/s")
    print(f"Ridge point (break-even AI): {RIDGE:.1f} FLOPs/byte")
    print(f"  -> AI < {RIDGE:.0f}: memory-bound")
    print(f"  -> AI > {RIDGE:.0f}: compute-bound")
    print()

    # Decode analysis
    flops_d, bytes_d, ai_d = compute_ai_decode()
    peak_d = roofline_peak(ai_d)
    measured_decode_tok_s = results[3]["decode_tokens_per_sec"]  # 256 prompt
    measured_flops_d = measured_decode_tok_s * flops_d
    efficiency_d = measured_flops_d / peak_d * 100

    print("DECODE step (1 token):")
    print(f"  FLOPs/step: {flops_d/1e6:.0f} M")
    print(f"  Bytes loaded: {bytes_d/1e6:.0f} MB")
    print(f"  Arithmetic Intensity (AI): {ai_d:.2f} FLOPs/byte  (<<{RIDGE:.0f} -> memory-bound)")
    print(f"  Roofline peak @ this AI: {peak_d/1e9:.1f} GFLOPs/s  ({peak_d/PEAK_FLOPS*100:.2f}% of compute peak)")
    print(f"  Measured throughput: {measured_decode_tok_s:.0f} tok/s = {measured_flops_d/1e9:.1f} GFLOPs/s")
    print(f"  Efficiency vs roofline: {efficiency_d:.1f}%")
    print(f"  Time per token (theory): {1000*flops_d/peak_d:.2f} ms")
    print(f"  Time per token (measured): {1000/measured_decode_tok_s:.2f} ms")
    print()

    # Prefill analysis for each prompt length
    print("PREFILL (per prompt length):")
    print(f"  {'len':>4} {'FLOPs':>10} {'AI':>8} {'bound':>12} {'peak(G)':>10} {'meas(G)':>10} {'eff%':>7}")
    prefill_data = []
    for r in results:
        L = r["prompt_len"]
        flops_p, bytes_p, ai_p = compute_ai_prefill(L)
        peak_p = roofline_peak(ai_p)
        meas_tok_s = r["prefill_tokens_per_sec"]
        meas_flops_p = meas_tok_s * 2 * PARAMS
        eff = meas_flops_p / peak_p * 100
        bound = "compute" if ai_p > RIDGE else "memory"
        print(f"  {L:>4} {flops_p/1e9:>9.1f}G {ai_p:>7.1f}  {bound:>11} {peak_p/1e9:>9.0f} {meas_flops_p/1e9:>9.0f} {eff:>6.1f}%")
        prefill_data.append((L, ai_p, meas_flops_p, peak_p))
    print()

    # ── Plot Roofline chart ──
    fig, ax = plt.subplots(figsize=(10, 7))

    ai_range = np.logspace(-2, 4, 500)
    roof = np.minimum(PEAK_FLOPS, ai_range * PEAK_BW) / 1e9   # GFLOPs/s

    ax.loglog(ai_range, roof, "k-", lw=2.5, label="Roofline")
    ax.axhline(PEAK_FLOPS / 1e9, color="#999", ls="--", lw=1,
               label=f"Peak compute ({PEAK_FLOPS/1e12:.0f} TFLOPs)")
    ax.axvline(RIDGE, color="#999", ls=":", lw=1,
               label=f"Ridge point (AI={RIDGE:.0f})")

    # Shade memory vs compute bound regions
    ax.axvspan(1e-2, RIDGE, alpha=0.12, color="red", label="Memory-bound region")
    ax.axvspan(RIDGE, 1e4, alpha=0.12, color="green", label="Compute-bound region")

    # Plot decode point
    ax.scatter([ai_d], [measured_flops_d / 1e9],
               s=220, color="red", marker="o", zorder=5, edgecolor="black", lw=1.5,
               label=f"Decode (bs=1)")
    ax.annotate("DECODE\n(memory-bound)\n0.5% of peak compute",
                xy=(ai_d, measured_flops_d / 1e9),
                xytext=(ai_d * 0.15, measured_flops_d / 1e9 * 8),
                fontsize=10, ha="center", color="darkred", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5))

    # Plot prefill points
    colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]
    for (L, ai_p, meas_p, peak_p), color in zip(prefill_data, colors):
        ax.scatter([ai_p], [meas_p / 1e9],
                   s=140, color=color, marker="s", zorder=5, edgecolor="black", lw=1,
                   label=f"Prefill L={L}")

    # Annotate prefill region
    ax.annotate("PREFILL\n(compute-bound\nfor L>=128)",
                xy=(prefill_data[-1][1], prefill_data[-1][2] / 1e9),
                xytext=(2000, 8000),
                fontsize=10, ha="center", color="darkgreen", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5))

    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)", fontsize=12)
    ax.set_ylabel("Achievable Performance (GFLOPs/s)", fontsize=12)
    ax.set_title("Roofline Model: GPT-2 on RTX 4090 Laptop\nWhy decode is ~500 tok/s no matter what",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(1e0, 1e5)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig("W2/roofline_chart.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved: W2/roofline_chart.png")

    # ── Key insight summary ──
    print("\n" + "=" * 78)
    print("KEY INSIGHTS (why decode is slow)")
    print("=" * 78)
    print(f"1. Decode AI = 0.5 FLOPs/byte, ridge point = {RIDGE:.0f} FLOPs/byte")
    print(f"   -> Decode operates at 1/{RIDGE/ai_d:.0f} of the ridge = deeply memory-bound")
    print()
    print(f"2. Decode theoretical ceiling = {peak_d/1e9:.0f} GFLOPs/s")
    print(f"   = {peak_d/PEAK_FLOPS*100:.2f}% of RTX 4090's 33 TFLOPs peak compute")
    print(f"   You could have 100x more compute and decode wouldn't speed up")
    print()
    print(f"3. Decode time per token is physically lower-bounded by:")
    print(f"   500 MB / 576 GB/s = {1000*WEIGHT_BYTES/PEAK_BW:.2f} ms (just loading weights)")
    print(f"   Your measured: {1000/measured_decode_tok_s:.2f} ms (includes overhead)")
    print()
    print(f"4. Prefill at L>=128 enters compute-bound region")
    print(f"   Running at {prefill_data[-1][2]/prefill_data[-1][3]*100:.0f}% of compute peak (healthy)")
    print()
    print("CONCLUSION:")
    print("  - Decode slow is NOT a bug. It's the physical bandwidth limit.")
    print("  - Only way to speed up decode: move the red dot right or up.")
    print("    -> Right: increase AI via KV Cache / batch / speculative decoding")
    print("    -> Up:   reduce bytes via quantization (INT8/INT4)")


if __name__ == "__main__":
    main()
