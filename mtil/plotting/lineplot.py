import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--log", action="store_true", help="Use logarithmic y-axis")
parser.add_argument("--log-base", type=float, default=10, help="Base for logarithmic scale (default: 10)")
args = parser.parse_args()

tasks = ["Base Model", "DTD", "DTD,MNIST", "DTD,MNIST,\nEuroSAT", "DTD,MNIST,\nEuroSAT,Flowers"]
finetune  = [65.39,  7.94,  7.35,  7.70,  4.75]
sfao      = [65.86, 65.88, 53.92, 49.14, 70.17]
ogd       = [65.86, 68.99, 69.91, 67.51, 74.72]
sharelora = [65.86, 67.59, 70.00, 72.44, 75.46]
lora      = [65.86, 68.07, 70.28, 72.82, 76.61]
zscl      = [65.86, 68.60, 70.52, 70.29, 75.70]

x = list(range(len(tasks)))

methods = {
    "ZSCL+SFAO":      sfao,
    "ZSCL+OGD":       ogd,
    "ZSCL+ShareLoRA": sharelora,
    "ZSCL+LoRA":      lora,
    "ZSCL":      zscl,
    # "Finetune":  finetune,
}

fig, ax = plt.subplots(figsize=(8, 5))

for name, values in methods.items():
    ax.plot(x, values, marker="o", label=name)

ax.set_xlabel("Task")
ax.set_ylabel("Average Accuracy Across All Tasks (%)")
ax.set_title("Continual Learning — Accuracy per Task")
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=8)
if args.log:
    ax.set_yscale("log", base=args.log_base)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(r"a:\thesis\thesis\mtil\plotting\lineplot.png", dpi=150)
plt.show()
