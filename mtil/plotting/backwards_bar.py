import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

x_axis = ["DTD", "MNIST", "EuroSAT", "Flowers", "Average"]

zscl       = [-2.18, -0.17, -0.17, 0.00, -0.62]
ogd        = [-1.81, -0.04, -0.28, 0.00, -0.53]
lora       = [-2.23, -0.11, -0.26, 0.00, -0.65]
share_lora = [-4.15, -0.40, -0.50, 0.00, -1.26]
sfao       = [-4.89, -0.05, -0.35, 0.00, -1.32]

methods = {
    "ZSCL":       zscl,
    "OGD":        ogd,
    "LoRA":       lora,
    "Share-LoRA": share_lora,
    "SFAO":       sfao,
}

datasets  = x_axis[:-1]   # DTD, MNIST, EuroSAT, Flowers
avg_label = x_axis[-1]    # Average

n_methods  = len(methods)
bar_width  = 0.15
gap        = 0.6           # extra space before Average group

# x positions: 0,1,2,3 for datasets then gap+4 for Average
x_datasets = np.arange(len(datasets))
x_avg      = x_datasets[-1] + 1 + gap

fig, ax = plt.subplots(figsize=(11, 5))

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for i, (label, values) in enumerate(methods.items()):
    offset = (i - n_methods / 2 + 0.5) * bar_width
    # dataset bars
    ax.bar(x_datasets + offset, values[:-1], bar_width,
           label=label, color=colors[i])
    # average bar (same color, no extra label)
    ax.bar(x_avg + offset, values[-1], bar_width,
           color=colors[i])

# vertical divider between datasets and average
divider_x = (x_datasets[-1] + x_avg) / 2
ax.axvline(divider_x, color="grey", linewidth=1.2, linestyle="--", alpha=0.7)

# x-tick positions and labels
all_x      = list(x_datasets) + [x_avg]
all_labels = datasets + [avg_label]
ax.set_xticks(all_x)
ax.set_xticklabels(all_labels)

ax.set_ylabel("Backwards Transfer (%)")
ax.set_title("Backwards Transfer by Method and Dataset")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.legend()
ax.set_xlim(-0.6, x_avg + 0.6)
ax.set_ylim(min(v for vals in methods.values() for v in vals) * 1.2, 0.5)

plt.tight_layout()
plt.savefig("backwards_transfer.png", dpi=150)
print("Saved to backwards_transfer.png")
