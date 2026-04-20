import numpy as np
import time
'''
这段代码定义了一个函数 `matrix_multiply`，
它使用 NumPy 的 `dot` 函数来执行矩阵乘法。
然后，它生成不同大小的随机矩阵，并测量每次矩阵乘法的执行时间。
最后，它计算并打印每个输入大小的平均时间、最小时间和最大时间。
'''
def matrix_multiply(A, B):
    return np.dot(A, B)
# 定义输入大小和运行次数
input_sizes = [100, 200, 500, 1000, 2000]
# 运行每个输入大小的次数
num_runs = 5
# 对每个输入大小进行测试
for size in input_sizes:
    # 生成随机矩阵 A 和 B
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    times = []
# 运行矩阵乘法并记录时间
    for _ in range(num_runs):
        # 记录开始时间
        start_time = time.perf_counter()
        # 执行矩阵乘法
        matrix_multiply(A, B)
        # 记录结束时间
        end_time = time.perf_counter()
        # 计算并存储运行时间
        times.append(end_time - start_time)
    # 取得平均时间、最小时间和最大时间
    avg_time = sum(times) / len(times)
    print(
    f"Input size: {size}x{size}, "
    f"avg: {avg_time:.6f}s, "
    f"min: {min(times):.6f}s, "
    f"max: {max(times):.6f}s"
)
