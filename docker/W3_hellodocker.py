"""Detect GPU inside a Docker container and print basic info."""
import sys
import platform

print("=" * 60)
print("Hello from inside a Docker container!")
print("=" * 60)
print(f"Python version: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print(f"Hostname: {platform.node()}")

print("\n--- PyTorch + CUDA check ---")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    SM count: {props.multi_processor_count}")
            print(f"    Compute capability: {props.major}.{props.minor}")

        # Run a tiny GPU op to prove it actually works
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        print(f"\nTiny GEMM (1024x1024) on GPU: OK (result sum = {y.sum().item():.2f})")
except ImportError:
    print("PyTorch not installed in this container.")

print("\nContainer exited cleanly.")
