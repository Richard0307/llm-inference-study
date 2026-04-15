"""
GPT-2 Inference Profiling: Prefill vs Decode Phase Analysis
============================================================
Task: Measure prefill and decode latency separately across different prompt lengths.
Hypothesis: Decode phase accounts for >80% of total inference time.
"""

import torch
import time
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.cuda.nvtx as nvtx
# ── Config ──
MODEL_PATH = "W2/gpt2_model/AI-ModelScope/gpt2"
PROMPT_LENGTHS = [32, 64, 128, 256]
GEN_LENGTH = 64
WARMUP_RUNS = 3
MEASURE_RUNS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print(f"Loading GPT-2 from {MODEL_PATH} on {DEVICE}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model, tokenizer


def make_input(tokenizer, target_len):
    """Generate a dummy prompt of exactly target_len tokens"""
    # Repeat a simple sentence until we reach the target length
    base = "The quick brown fox jumps over the lazy dog. "
    text = base * (target_len // 8 + 1)
    ids = tokenizer.encode(text)[:target_len]
    return torch.tensor([ids], device=DEVICE)


def measure_prefill(model, input_ids):
    """Measure prefill (prompt processing) latency using CUDA events"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start.record()
        nvtx.range_push("prefill")
        outputs = model(input_ids, use_cache=True)
        nvtx.range_pop() 
        end.record()

    torch.cuda.synchronize()
    prefill_ms = start.elapsed_time(end)
    return prefill_ms, outputs.past_key_values


def measure_decode(model, input_ids, past_key_values, gen_length):
    """Measure decode (token generation) latency using CUDA events"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    next_token = input_ids[:, -1:]
    kv_cache = past_key_values

    with torch.no_grad():
        start.record()
        for step in range(gen_length):
            if step == 10:
                nvtx.range_push("decode_step") # Only test the 10th steps
            outputs = model(next_token, past_key_values=kv_cache, use_cache=True)
            kv_cache = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if step == 10:
                nvtx.range_pop()
        end.record()

    torch.cuda.synchronize()
    decode_ms = start.elapsed_time(end)
    return decode_ms


def measure_generate_e2e(model, input_ids, gen_length):
    """Measure end-to-end model.generate() latency for comparison"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start.record()
        model.generate(
            input_ids,
            max_new_tokens=gen_length,
            do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )
        end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end)


def run_profiling(model, tokenizer):
    """Run profiling across all prompt lengths"""
    results = []

    for prompt_len in PROMPT_LENGTHS:
        input_ids = make_input(tokenizer, prompt_len)
        print(f"\n--- Prompt length: {prompt_len} tokens, Generate: {GEN_LENGTH} tokens ---")

        # Warmup
        for _ in range(WARMUP_RUNS):
            prefill_ms, kv = measure_prefill(model, input_ids)
            measure_decode(model, input_ids, kv, GEN_LENGTH)

        # Measure
        prefill_times = []
        decode_times = []
        e2e_times = []

        for i in range(MEASURE_RUNS):
            p_ms, kv = measure_prefill(model, input_ids)
            d_ms = measure_decode(model, input_ids, kv, GEN_LENGTH)
            e2e_ms = measure_generate_e2e(model, input_ids, GEN_LENGTH)
            prefill_times.append(p_ms)
            decode_times.append(d_ms)
            e2e_times.append(e2e_ms)

        prefill_avg = sum(prefill_times) / len(prefill_times)
        decode_avg = sum(decode_times) / len(decode_times)
        e2e_avg = sum(e2e_times) / len(e2e_times)
        total = prefill_avg + decode_avg
        decode_ratio = decode_avg / total * 100

        result = {
            "prompt_len": prompt_len,
            "gen_len": GEN_LENGTH,
            "prefill_ms": round(prefill_avg, 2),
            "decode_ms": round(decode_avg, 2),
            "total_ms": round(total, 2),
            "decode_ratio_pct": round(decode_ratio, 1),
            "e2e_generate_ms": round(e2e_avg, 2),
            "decode_per_token_ms": round(decode_avg / GEN_LENGTH, 2),
            "prefill_tokens_per_sec": round(prompt_len / (prefill_avg / 1000), 1),
            "decode_tokens_per_sec": round(GEN_LENGTH / (decode_avg / 1000), 1),
        }
        results.append(result)

        print(f"  Prefill:  {prefill_avg:8.2f} ms")
        print(f"  Decode:   {decode_avg:8.2f} ms ({GEN_LENGTH} tokens)")
        print(f"  Total:    {total:8.2f} ms")
        print(f"  Decode %: {decode_ratio:8.1f}%")
        print(f"  Per-token decode: {decode_avg / GEN_LENGTH:.2f} ms/token")

    return results


def print_table(results):
    """Print formatted results table"""
    print("\n" + "=" * 90)
    print("PROFILING RESULTS: GPT-2 Prefill vs Decode")
    print("=" * 90)
    header = f"{'prompt_len':>10} | {'prefill(ms)':>11} | {'decode(ms)':>10} | {'total(ms)':>9} | {'decode%':>7} | {'tok/s(pf)':>9} | {'tok/s(dc)':>9}"
    print(header)
    print("-" * 90)
    for r in results:
        print(f"{r['prompt_len']:>10} | {r['prefill_ms']:>11.2f} | {r['decode_ms']:>10.2f} | {r['total_ms']:>9.2f} | {r['decode_ratio_pct']:>6.1f}% | {r['prefill_tokens_per_sec']:>9.1f} | {r['decode_tokens_per_sec']:>9.1f}")
    print("=" * 90)


def verify_hypothesis(results):
    """Verify: Does decode phase account for >80% of total time?"""
    print("\n" + "=" * 90)
    print("HYPOTHESIS VERIFICATION")
    print("Hypothesis: Decode phase accounts for >80% of total inference time")
    print("=" * 90)

    all_above_80 = True
    for r in results:
        status = "CONFIRMED" if r["decode_ratio_pct"] > 80 else "REJECTED"
        if r["decode_ratio_pct"] <= 80:
            all_above_80 = False
        print(f"  prompt_len={r['prompt_len']:>3}: decode = {r['decode_ratio_pct']:.1f}% -> {status}")

    print()
    if all_above_80:
        print("  CONCLUSION: Hypothesis CONFIRMED for all prompt lengths.")
        print("  Decode (autoregressive, one token at a time) dominates inference cost.")
    else:
        print("  CONCLUSION: Hypothesis PARTIALLY REJECTED.")
        print("  At longer prompt lengths, prefill cost becomes significant.")
        print("  This is expected: prefill is compute-bound (GEMM on full sequence),")
        print("  while decode is memory-bound (GEMV, one token at a time).")

    print()
    print("  KEY INSIGHT:")
    print("  - Prefill: compute-bound (parallel matrix multiply over all prompt tokens)")
    print("  - Decode:  memory-bound (sequential, loads full KV cache per token)")
    print("  - As prompt gets longer, prefill grows but decode stays ~constant per token")
    print("=" * 90)


def run_torch_profiler(model, tokenizer):
    """Generate Chrome trace for detailed profiling"""
    input_ids = make_input(tokenizer, 64)
    print("\nGenerating Chrome trace (prompt_len=64, gen=64)...")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            model.generate(
                input_ids,
                max_new_tokens=GEN_LENGTH,
                do_sample=False,
                pad_token_id=model.config.eos_token_id,
            )

    trace_path = "W2/gpt2_profiling_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace saved to {trace_path}")
    print("Open in: chrome://tracing or https://ui.perfetto.dev/")

    # Print top CUDA operations
    print("\nTop 15 CUDA operations by total time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def load_model_precision(precision):
    """Load GPT-2 in the requested precision: fp32 / fp16 / int8."""
    print(f"Loading GPT-2 from {MODEL_PATH} on {DEVICE} (precision={precision})...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    if precision == "fp32":
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE)
    elif precision == "fp16":
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    elif precision == "int8":
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, quantization_config=bnb_cfg, device_map="auto")
    else:
        raise ValueError(f"unknown precision: {precision}")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded. Parameters: {n_params:.1f}M, dtype_sample={next(model.parameters()).dtype}")
    return model, tokenizer


def run_ncu_target(precision="fp32"):
    """Minimal entry point for ncu profiling. Only runs prefill + 11 decode steps."""
    model, tokenizer = load_model_precision(precision)
    input_ids = make_input(tokenizer, 128)
    print(f"[ncu] input shape: {tuple(input_ids.shape)}")

    print("[ncu] warmup...")
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        kv = out.past_key_values
        tok = input_ids[:, -1:]
        for _ in range(5):
            out = model(tok, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    print("[ncu] measured prefill (nvtx range: prefill)")
    measure_prefill(model, input_ids)
    _, kv = measure_prefill(model, input_ids)

    print("[ncu] measured decode (nvtx range: decode_step, step=10)")
    measure_decode(model, input_ids, kv, gen_length=12)

    print("[ncu] done")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ncu":
        precision = sys.argv[2] if len(sys.argv) > 2 else "fp32"
        run_ncu_target(precision)
    else:
        model, tokenizer = load_model()
        # 1. Prefill vs Decode profiling
        results = run_profiling(model, tokenizer)
        print_table(results)
        verify_hypothesis(results)

        # 2. Chrome trace
        run_torch_profiler(model, tokenizer)

        # 3. Save results to JSON
        with open("W2/gpt2_profiling_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to W2/gpt2_profiling_results.json")

        # 4. Peak GPU memory
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU memory: {peak_mem:.1f} MB")
