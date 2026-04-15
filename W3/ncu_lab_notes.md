#  GPT-2(SLM) 的真实 GPU 利用率




我用 ncu 把 GPT-2 的 prefill 和 decode 在 FP32 / FP16 / INT8 三种精度下各按在地上摩擦了一遍，记录了它的每一次心跳（HMMA）、每一口呼吸（DRAM），然后得出了三个让我从椅子上弹起来的结论。


---

## 1. 起因：nvidia-smi 骗了我

之前推理的时候我看 `nvidia-smi`，decode 阶段 SM util 蹦到 80%，当时还兴奋地以为 GPU 正在卖命干活。直到我读完 Roofline 那一章才反应过来：SM util 这个指标根本就是个社交辞令，它只告诉你 SM 没闲着，至于它是在算数还是在等 HBM 发货，smi 一点都不关心。

所以今天的任务是：用 ncu 把真相扒出来。

---

## 2. 实验设置（能复现的那种）

### 2.1 范围

1. 在 gpt2_profiling.py 里加一个极简入口 `run_ncu_target(precision)`，只跑一次 prefill 加 11 步 decode，带 5 步 warmup。
2. 用 `torch.cuda.nvtx.range_push/pop` 在 prefill 和第 10 步 decode 外面打两个圈。
3. ncu 用 `--nvtx --nvtx-include "prefill/"` 精确切片，只抓圈里那点 kernel。
4. 三种精度分别跑一遍，共 6 次 ncu profile。

### 2.2 指标三件套

| Metric | 俗称 | 作用 |
|---|---|---|
| `sm__pipe_tensor_op_hmma_cycles_active.pct_of_peak` | Tensor Core 心跳 | 真实算力有没有被吃到 |
| `dram__throughput.pct_of_peak_sustained_elapsed` | HBM 呼吸 | 带宽被挤了多少 |
| `smsp__cycles_active.pct_of_peak` | SM 活跃度 | SM 有没有在椅子上坐着 |

这三个必须一起看，单看任何一个都会得出离谱结论。举例：光看 SM 活跃度，你会以为 decode 很努力；加上另外两个你才会发现它其实在等快递。

---

## 3. 踩坑笔记（血泪版）

在看数据之前必须先讲坑，因为这些坑每个都让我吃了半小时。

### 坑 1：静默成功看起来像崩溃

我的 `run_ncu_target()` 一行 print 都没写，跑完就退出。终端显示到 Model loaded 就不动了，我以为它卡死，实际上它已经跑完散场灯都关了。教训：profile 入口函数必须带进度 print，否则你根本不知道它是在工作还是在给你表演默剧。

### 坑 2：NVTX range 名字对不齐

代码里我写 `range_push("decode_step")`，命令里我写 `--nvtx-include "decode/"`。ncu 非常体贴地不报错，只给我一句 No kernels were profiled，然后跑完交差。教训：ncu 对 NVTX 名字是字面匹配，一个字母都不能差，而且不匹配时它选择性失明。

### 坑 3：cuDNN 库找不到（最耗时间的那个）

ncu 启动 Python 子进程时会搞乱 conda 的动态库解析顺序。PyTorch 的 cuDNN 是以 wheel 形式装在 `site-packages/nvidia/cudnn/lib/` 的，平时 PyTorch 启动时自己 dlopen 没事，但 ncu 把它拦在半路上，子进程就哭着说 `libcudnn_graph.so.9` 找不到，然后返回 error code 6。

解决方案是在 ncu 命令前显式设置 LD_LIBRARY_PATH：

```bash
CUDNN_LIB=/miniforge3/envs/llm-infer/lib/python3.11/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$CUDNN_LIB:$LD_LIBRARY_PATH
```

然后全部正常了。这个坑教会我一件事：**ncu 不是魔法，它只是个复杂的 subprocess wrapper**。

---

## 4. 主菜：数据来了

### 4.1 硬件利用率三连拍

![ncu utilization comparison](ncu_utilization_comparison.png)

#### Prefill（seq_len=128）

| 精度 | Kernel 数 | DRAM 峰值 | HMMA 峰值 | SM 活跃均值 |
|---|---:|---:|---:|---:|
| FP32 | 468 | 70.7% | 5.7% | 66.1% |
| FP16 | 468 | 56.4% | **27.9%** | 54.5% |
| INT8 | **1552** | 53.1% | 26.2% | 31.8% |

#### Decode（单 token）

| 精度 | Kernel 数 | DRAM 峰值 | HMMA 峰值 | SM 活跃均值 |
|---|---:|---:|---:|---:|
| FP32 | 235 | **96.1%** | 4.1% | 15.3% |
| FP16 | 235 | 94.6% | 4.3% | 16.8% |
| INT8 | 555 | 94.7% | 4.3% | 10.8% |

### 4.2 端到端延迟和显存

![latency comparison](precision_latency_comparison.png)

| 精度 | Prefill (ms) | Decode (ms/token) | Peak Mem (MB) |
|---|---:|---:|---:|
| FP32 | 3.29 | 2.58 | 542.4 |
| FP16 | 3.05 | 2.64 | 508.5 |
| INT8 | 14.64 | 9.94 | 320.4 |

看到 INT8 的延迟你先别揉眼睛，数据是真的。下面马上解释这场惨案。

---

## 5. 三个让我从椅子上弹起来的结论

### 结论 1：Decode 的 DRAM 峰值跨三种精度都逼近 95%，HBM 被按在摩擦

三种精度 decode 的 DRAM 峰值是 96.1 / 94.6 / 94.7，几乎压满带宽天花板。更损的是 SM 活跃度只有 10% 到 17%。翻译成人话：**SM 坐在那里百无聊赖，HBM 累得气喘吁吁**。

为什么？因为 decode 每一步都是一个 GEMV（矩阵乘向量）。ncu 抓到的热点 kernel 名字就是证据：

```
gemv2T_kernel_val          (FP32)
gemvx::kernel              (FP16)
gemvx::kernel + gemmSN_int32 (INT8)
```

GEMV 的算术强度大约是 2，这意味着每从 HBM 搬 1 个字节，只做 2 次浮点运算。Roofline 上这个点深陷在 memory-bound 的屋檐下，连屋顶边都摸不到。

### 结论 2：Tensor Core 只在 Prefill + FP16 真正被唤醒，Decode 永远别想

这个表是今天最关键的一张：

| 场景 | HMMA 峰值 | 翻译 |
|---|---:|---|
| Prefill FP32 | 5.7% | Tensor Core 在睡觉，FP32 走 CUDA core 的 FMA 流水线 |
| **Prefill FP16** | **27.9%** | Tensor Core 真的被叫醒了，热点 kernel 换成了 `cutlass::gemm...half` |
| Prefill INT8 | 26.2% | 别高兴太早，这个数值来自 bnb 的 fp16 中间层（LayerNorm、softmax），不是 INT8 GEMM 本身 |
| Decode 任意精度 | ~4% | Tensor Core 看了一眼 GEMV，转身就走 |

**为什么 decode 永远吃不到 Tensor Core**：HMMA 指令的本质是一次处理一个 tile 的矩阵乘矩阵。decode 的激活只有一个向量，不是矩阵，Tensor Core 的硬件结构直接对不上。所以只要你还在 batch=1 做自回归 decode，你买 GPU 的 Tensor Core 那部分钱基本上是白交的。

这也是为什么所有 LLM serving 框架（vLLM、TGI、TensorRT-LLM）都在死命推 batching：**把多个 decode 请求拼成矩阵，GEMV 变成 GEMM，Tensor Core 才会睁眼**。

### 结论 3：INT8（bitsandbytes）在 GPT-2 小模型上是反向优化，请不要拿它加速

看这张表的时候请坐稳：

| 指标 | FP32 | INT8 | 变化 |
|---|---:|---:|---|
| Prefill | 3.29 ms | 14.64 ms | **慢 4.4 倍** |
| Decode (ms/token) | 2.58 | 9.94 | **慢 3.85 倍** |
| Peak memory | 542 MB | 320 MB | 省 41% |
| Prefill kernel 数 | 468 | 1552 | 膨胀 3.3 倍 |
| Decode kernel 数 | 235 | 555 | 膨胀 2.4 倍 |

我第一眼看到这个结果的反应是重跑一遍，以为自己手滑。重跑之后反应是翻文档，想找哪个参数开错了。没开错。bnb 就是这样。

**为什么 INT8 这么惨**：

1. bnb 在每次矩阵乘前后都插入 quantize 和 dequantize 的额外 kernel。你看到的 kernel 数从 468 膨胀到 1552 就是这个原因。每个 kernel launch 有几微秒的 CPU-GPU 同步开销，1500 次启动加起来比 INT8 GEMM 节省的算力还多。
2. GPT-2 124M 本身太小。bnb 的 INT8 设计目标是在 24GB 卡上放下 13B 模型，它的价值是**能装下**，不是**跑得快**。对于 500MB 的 GPT-2，你 FP16 随便装，INT8 只是把省显存的收益包装在一个慢 4 倍的盒子里给你。
3. bnb 的 `gemmSN_kernel_int32` 用 INT32 累加器，在 Ada 这代 GPU 上吞吐并不比 FP16 Tensor Core GEMM 高，小模型的 GEMM 规模还吃不满量化 kernel 的启动 overhead。

**正确的 INT8 使用场景**：你有一个 24GB 卡，想跑一个 13B 模型，FP16 装不下，这时候 bnb INT8 救你一命。其他场景请慎用。



### 需要警惕的失败模式

如果优化后 DRAM 利用率下降了但延迟没变，那说明瓶颈已经从 HBM 带宽转移到 kernel launch overhead 或 CPU-GPU 同步。这是下一轮优化的新战场。

---


> SM utilization 告诉你 SM 很忙，Tensor Core utilization 告诉你 SM 在忙什么。前者是朋友圈，后者是体检报告。

> INT8 对小模型来说，是用速度去换你根本不需要的显存。

> ncu 不报错不等于实验成功，No kernels were profiled 是它最温柔的羞辱。


