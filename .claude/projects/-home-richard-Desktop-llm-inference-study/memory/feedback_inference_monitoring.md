---
name: LLM inference experiment monitoring reminder
description: When user runs LLM inference experiments, proactively remind them to add latency/throughput/memory/parallel efficiency monitoring to code and results markdown
type: feedback
---

When the user is running LLM inference experiments (like day2.py style agent loop or similar), proactively remind and help them monitor four metrics in their code AND write results to the markdown report:

1. **Latency** — `time.perf_counter()` around each LLM call, record per-step and total
2. **Throughput** — tokens/second, plus vLLM's `/metrics` endpoint if using local vLLM
3. **GPU Memory** — `nvidia-smi` or vLLM's `gpu_cache_usage_perc` metric
4. **Parallel Efficiency** — if applicable, compare serial vs concurrent request total time, compute speedup ratio

**Why:** User explicitly requested this as a standing reminder for all future inference experiments. They want these metrics baked into both the experiment code and the results markdown document.

**How to apply:** When the user starts writing or modifying inference experiment code, proactively suggest adding these four monitoring dimensions before they run the experiment. Continue reminding until the user explicitly says to stop.
