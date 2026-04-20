# Journal — Weekly Study Log

This directory is the **chronological source of truth** for the project: every script, raw data file, and memo, organized by the week it was produced.

The polished, portfolio-facing version of these findings lives at the repository root ([README.md](../README.md)) and (progressively) under `../docs/`. This `journal/` directory is preserved as-is so that:

- The reasoning trail (hypothesis → experiment → result → next-week hypothesis) stays inspectable.
- Raw data is never lost when polished writeups are factored out.
- Future weeks can be added without disturbing the public-facing structure.

## Index

| Week | Dates | Theme | Status |
|---|---|---|:---:|
| [W1](W1/) | Apr 1 – 5  | Transformer foundations: attention, multi-head attention | ✅ |
| [W2](W2/) | Apr 7 – 11 | Decoder block + GPT-2 prefill/decode profiling | ✅ |
| [W3](W3/) | Apr 13 – 19 | GPU memory hierarchy + Roofline + precision study | ✅ |
| [W4](W4/) | Apr 20 – 26 | KV cache implementation + FastAPI serving | 🚧 |

## Naming Conventions Inside Each Week

- `dayN.py` / `dayN.md` — daily working scripts and notes
- `*_memo.md` — end-of-week consolidated memo
- `*_results.json` / `*_results.md` — measurement outputs
- `*.png` — figures

## Hypothesis Trail

Each week opens with a falsifiable hypothesis and closes with a verdict that seeds the next week.

| Week | Hypothesis | Verdict |
|---|---|---|
| W2 | Decode accounts for > 80 % of inference time | ✅ Confirmed (96.7 – 98.2 %) |
| W3 | GPT-2 decode is deeply memory-bound, real FLOPs utilization < 5 % | ✅ Confirmed (0.36 %) |
| W4 | KV cache gives > 2× decode speedup at seq_len ≥ 256 | 🚧 Testing |
