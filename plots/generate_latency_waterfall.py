import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Set publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 10
})

stages = [
    "Backbone Sensory\n(conv1-layer3)",
    "O(1) Fast-Gate\nDistance",
    "Conformal\nThreshold Gating",
    "Active Sub-Circuit\n(layer4, |C|<=2)",
    "Logit Fusion\n& Argmax"
]

times = [4.15, 0.18, 0.04, 6.22, 0.68]  # ms
cum_times = [0.0]
for t in times[:-1]:
    cum_times.append(cum_times[-1] + t)

total_time = sum(times)  # 11.27 ms

fig, ax = plt.subplots(figsize=(9, 4.2), dpi=300)

colors = ['#2b6cb0', '#319795', '#d69e2e', '#805ad5', '#38a169']

for i in range(len(stages)):
    bar = ax.bar(stages[i], times[i], bottom=cum_times[i], color=colors[i], width=0.55, edgecolor='black', linewidth=1)
    # Add time labels inside or above
    y_pos = cum_times[i] + times[i] / 2.0
    ax.text(i, y_pos, f"+{times[i]:.2f} ms", ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)
    
    # Connecting dashed line
    if i < len(stages) - 1:
        ax.plot([i + 0.275, i + 0.725], [cum_times[i] + times[i], cum_times[i] + times[i]], 'k--', alpha=0.5, lw=1.2)

# Total summary bracket / line
ax.axhline(total_time, color='#e53e3e', linestyle='-', linewidth=1.5, alpha=0.8)
ax.text(2.0, total_time + 0.45, f"Total Batch Inference: {total_time:.2f} ms  (0.176 ms/sample across batch of 64 on RTX 5070 Ti)", 
        ha='center', va='bottom', color='#9b2c2c', fontweight='bold', fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff5f5", ec="#feb2b2"))

ax.set_ylabel("Cumulative Batch Execution Latency (ms)", fontweight='bold')
ax.set_ylim(0, 13.2)
ax.set_title("Physical Microsecond Latency Waterfall Profile (NVIDIA RTX 5070 Ti Silicon)", fontweight='bold', fontsize=12)
ax.grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
os.makedirs("manuscript/figures", exist_ok=True)
plt.savefig("manuscript/figures/fig3_latency_waterfall.pdf", bbox_inches='tight', dpi=300)
plt.savefig("manuscript/figures/fig3_latency_waterfall.png", bbox_inches='tight', dpi=300)
plt.close()

print("Generated Figure 3: Physical Latency Waterfall Chart successfully!")
