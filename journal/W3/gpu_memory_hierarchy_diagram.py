"""Generate GPU memory hierarchy diagram for RTX 4090 Laptop."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 14)
ax.axis("off")

# Title
ax.text(7, 13.3, "GPU Memory Hierarchy (RTX 4090 Laptop)",
        ha="center", va="center", fontsize=16, fontweight="bold")
ax.text(7, 12.8, "Higher = Faster & Smaller  |  Lower = Slower & Larger",
        ha="center", va="center", fontsize=10, color="#666", style="italic")

# Layers: (y_center, height, width, label_main, specs, color, speed_note)
layers = [
    (11.5, 1.1, 4.5,  "Registers",
     "Per-thread | ~256 KB total\n~20 TB/s | ~0 cycle",        "#FFCDD2",
     "Fastest"),
    (10.0, 1.1, 6.0,  "L1 Cache / Shared Memory (SRAM)",
     "Per-SM | 128 KB each | 128 SMs\n~19 TB/s | ~30 cycles",  "#FFE0B2",
     "~10x slower than reg"),
    (8.3,  1.3, 7.5,  "L2 Cache",
     "Chip-wide | ~64 MB\n~5 TB/s | ~200 cycles",              "#FFF9C4",
     "~4x slower than L1"),
    (6.3,  1.5, 9.5,  "HBM / GDDR6X (Device Memory)",
     "Global GPU memory | 16 GB\n576 GB/s | ~500 cycles",      "#C8E6C9",
     "~10x slower than L2"),
    (4.0,  1.7, 11.5, "CPU DRAM (Host Memory)",
     "System RAM | 32-64 GB\n~60 GB/s | over PCIe",            "#BBDEFB",
     "~10x slower than HBM"),
    (1.7,  1.7, 13.0, "SSD / Disk",
     "Persistent storage | TB scale\n~7 GB/s | PCIe 4.0 NVMe", "#E1BEE7",
     "~10x slower than DRAM"),
]

for y, h, w, main, spec, color, note in layers:
    x = 7 - w / 2
    rect = mpatches.FancyBboxPatch(
        (x, y - h / 2), w, h, boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="#333", linewidth=1.8,
    )
    ax.add_patch(rect)
    ax.text(7, y + 0.18, main, ha="center", va="center",
            fontsize=12, fontweight="bold")
    ax.text(7, y - 0.25, spec, ha="center", va="center",
            fontsize=9, color="#444", family="monospace")
    # Speed note on the right
    ax.text(x + w + 0.15, y, note, ha="left", va="center",
            fontsize=9, color="#C62828", style="italic")

# Arrows showing data flow direction
for y1, y2 in [(11.5, 10.0), (10.0, 8.3), (8.3, 6.3), (6.3, 4.0), (4.0, 1.7)]:
    ax.annotate("",
                xy=(12.8, (y1 + y2) / 2 - 0.1),
                xytext=(12.8, (y1 + y2) / 2 + 0.1),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))

# Left-side labels: speed / latency gradient
ax.text(0.8, 11.5, "FAST", fontsize=12, color="#2E7D32", fontweight="bold",
        ha="center", rotation=90, va="center")
ax.text(0.8, 1.7, "SLOW", fontsize=12, color="#C62828", fontweight="bold",
        ha="center", rotation=90, va="center")
ax.annotate("", xy=(0.8, 2.5), xytext=(0.8, 10.8),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

# Bottom annotation: where GPT-2 lives
ax.text(7, 0.3,
        "GPT-2 weights (500 MB FP32) live in HBM.  Every decode step must reload them through the cache hierarchy.",
        ha="center", va="center", fontsize=9.5, color="#1B5E20", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#2E7D32"))

plt.tight_layout()
plt.savefig("W3/gpu_memory_hierarchy.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: W3/gpu_memory_hierarchy.png")
plt.close()
