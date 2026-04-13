"""
W3 Hypothesis Verification: Is GPU SM util < 30% at batch_size=1?

Strategy: run a sustained decode loop, sample GPU metrics via nvidia-smi dmon,
then aggregate SM / memory utilization statistics.
"""

import torch
import subprocess
import threading
import time
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_PATH = "W2/gpt2_model/AI-ModelScope/gpt2"
DEVICE = "cuda"
SAMPLE_DURATION_SEC = 20
BATCH_SIZE = 1
PROMPT_LEN = 64
GEN_LEN = 64


class DmonSampler:
    """Sample GPU metrics via nvidia-smi dmon in a background thread"""
    def __init__(self):
        self.samples = []
        self.proc = None
        self.thread = None
        self.running = False

    def _read_loop(self):
        for line in iter(self.proc.stdout.readline, ""):
            if not self.running:
                break
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < 12:
                continue
            try:
                self.samples.append({
                    "pwr":   float(parts[1]),
                    "gtemp": float(parts[2]),
                    "sm":    float(parts[4]),
                    "mem":   float(parts[5]),
                    "mclk":  float(parts[10]),
                    "pclk":  float(parts[11]),
                })
            except (ValueError, IndexError):
                pass

    def start(self):
        self.running = True
        self.proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "pucmt", "-i", "0", "-d", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        if self.thread:
            self.thread.join(timeout=2)


def sustained_decode(model, input_ids, stop_event):
    """Run GPT-2 decode in a loop until stop_event is set"""
    count = 0
    with torch.no_grad():
        while not stop_event.is_set():
            # Full prefill + decode cycle
            outputs = model(input_ids, use_cache=True)
            kv = outputs.past_key_values
            next_token = input_ids[:, -1:]
            for _ in range(GEN_LEN):
                if stop_event.is_set():
                    return count
                outputs = model(next_token, past_key_values=kv, use_cache=True)
                kv = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                count += 1
    return count


def aggregate(samples, label):
    if not samples:
        return f"{label}: no samples"
    n = len(samples)
    def avg(key): return sum(s[key] for s in samples) / n
    def mx(key):  return max(s[key] for s in samples)
    return {
        "label": label,
        "count": n,
        "sm_avg":    round(avg("sm"), 1),
        "sm_max":    round(mx("sm"), 1),
        "mem_avg":   round(avg("mem"), 1),
        "mem_max":   round(mx("mem"), 1),
        "pwr_avg":   round(avg("pwr"), 1),
        "pclk_avg":  round(avg("pclk"), 0),
    }


def print_result(r):
    print(f"\n  [{r['label']}] (n={r['count']} samples)")
    print(f"    SM util   avg: {r['sm_avg']:5.1f}%   max: {r['sm_max']:5.1f}%")
    print(f"    Mem util  avg: {r['mem_avg']:5.1f}%   max: {r['mem_max']:5.1f}%")
    print(f"    Power     avg: {r['pwr_avg']:5.1f} W")
    print(f"    GPU clock avg: {r['pclk_avg']:5.0f} MHz")


def main():
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    text = "The quick brown fox jumps over the lazy dog. " * 10
    ids = tokenizer.encode(text)[:PROMPT_LEN]
    input_ids = torch.tensor([ids] * BATCH_SIZE, device=DEVICE)

    print(f"Config: batch_size={BATCH_SIZE}, prompt_len={PROMPT_LEN}, gen_len={GEN_LEN}")
    print(f"Sampling duration: {SAMPLE_DURATION_SEC}s\n")

    # ── Phase 1: idle baseline ──
    print("[Phase 1] Idle baseline (5s)...")
    idle_sampler = DmonSampler()
    idle_sampler.start()
    time.sleep(5)
    idle_sampler.stop()

    # ── Phase 2: warmup ──
    print("[Phase 2] Warming up GPT-2...")
    stop_event = threading.Event()
    for _ in range(5):
        with torch.no_grad():
            model(input_ids, use_cache=True)
    torch.cuda.synchronize()

    # ── Phase 3: sustained decode + sampling ──
    print(f"[Phase 3] Running sustained decode for {SAMPLE_DURATION_SEC}s...")
    decode_sampler = DmonSampler()
    decode_sampler.start()

    decode_thread = threading.Thread(
        target=lambda: setattr(decode_thread, "result",
                               sustained_decode(model, input_ids, stop_event))
    )
    decode_thread.start()

    time.sleep(SAMPLE_DURATION_SEC)
    stop_event.set()
    decode_thread.join()
    torch.cuda.synchronize()
    decode_sampler.stop()

    total_tokens = getattr(decode_thread, "result", 0)
    throughput = total_tokens / SAMPLE_DURATION_SEC

    # ── Results ──
    idle_r = aggregate(idle_sampler.samples, "IDLE baseline")
    decode_r = aggregate(decode_sampler.samples, f"DECODE bs={BATCH_SIZE}")

    print("\n" + "=" * 70)
    print("W3 H1 VERIFICATION: GPU SM util < 30% at batch_size=1?")
    print("=" * 70)

    if isinstance(idle_r, dict):
        print_result(idle_r)
    if isinstance(decode_r, dict):
        print_result(decode_r)
        print(f"\n  Decode throughput: {throughput:.1f} tokens/sec ({total_tokens} tokens)")

        print("\n" + "-" * 70)
        if decode_r["sm_avg"] < 30:
            print(f"  HYPOTHESIS CONFIRMED: SM avg = {decode_r['sm_avg']}% < 30%")
        else:
            print(f"  HYPOTHESIS REJECTED: SM avg = {decode_r['sm_avg']}% >= 30%")
        print(f"  Mem util avg = {decode_r['mem_avg']}%")
        if decode_r["mem_avg"] > decode_r["sm_avg"]:
            print(f"  Memory-bound confirmed: mem util > SM util")
        print("-" * 70)

    # Save raw data
    with open("W2/gpu_util_results.json", "w") as f:
        json.dump({
            "idle": idle_r if isinstance(idle_r, dict) else {},
            "decode": decode_r if isinstance(decode_r, dict) else {},
            "throughput_tok_per_sec": round(throughput, 1),
            "total_tokens": total_tokens,
            "config": {
                "batch_size": BATCH_SIZE,
                "prompt_len": PROMPT_LEN,
                "gen_len": GEN_LEN,
                "sample_duration_sec": SAMPLE_DURATION_SEC,
            },
        }, f, indent=2)
    print("\nResults saved to W2/gpu_util_results.json")


if __name__ == "__main__":
    main()
