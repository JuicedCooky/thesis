import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--log", action="store_true", help="Use logarithmic y-axis")
parser.add_argument("--log-base", type=float, default=10, help="Base for logarithmic scale (default: 10)")
args = parser.parse_args()

tasks = ["Base Model", "DTD", "DTD,MNIST", "DTD,MNIST,\nEuroSAT", "DTD,MNIST,\nEuroSAT,Flowers"]
finetune  = [57.60,	16.70,	16.63,	17.25,	9.81]
sfao      = [57.61,	66.33,	62.91,	73.41,	92.93]
ogd       = [57.61,	67.87,	74.81,	81.44,	93.19]
sharelora = [57.62,	58.53,	67.35,	79.82,	90.77]
lora      = [57.61,	64.89,	73.48,	84.59,	93.08]
zscl      = [57.61,	66.46,	74.89,	83.00,	93.02]

x = list(range(len(tasks)))

methods = {
    "ZSCL+SFAO":      sfao,
    "ZSCL+OGD":       ogd,
    "ZSCL+ShareLoRA": sharelora,
    "ZSCL+LoRA":      lora,
    "ZSCL":      zscl,
    "Finetune":  finetune,
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
plt.savefig(r"a:\thesis\thesis\mtil\plotting\lineplot_trained_only.png", dpi=150)
plt.show()
