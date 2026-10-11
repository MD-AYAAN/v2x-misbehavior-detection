import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F = "Times New Roman"
sweep = json.load(open(r"C:\projects\v2x-detection\outputs\sweep_v2.json"))

trees = [r["trees"] for r in sweep]
p99   = [r["p99_ms_mean"] for r in sweep]
p99sd = [r["p99_ms_std"] for r in sweep]
acc   = [r["acc_mean"] * 100 for r in sweep]

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

for budget, label in [(100, "Basic safety (100 ms)"),
                      (10, "Platooning (10 ms)"),
                      (3, "Advanced eV2X (3 ms)")]:
    ax.axhline(budget, color="black", lw=0.9, ls="--")
    ax.text(0.02, budget * 1.15, label, ha="left", va="bottom",
            fontsize=9, family=F, transform=ax.get_yaxis_transform())

ax.errorbar(trees, p99, yerr=p99sd, marker="o", color="black",
            lw=1.3, capsize=3, markerfacecolor="white", zorder=3)

for i, (x, y, a) in enumerate(zip(trees, p99, acc)):
    off = (-40, -14) if i == len(trees) - 1 else (6, -14)
    ax.annotate(f"{a:.2f}%", (x, y), textcoords="offset points",
                xytext=off, fontsize=8.5, family=F)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(trees)
ax.set_xticklabels(trees)
ax.set_xlabel("Number of trees", family=F, fontsize=11)
ax.set_ylabel("p99 inference latency (ms, log scale)", family=F, fontsize=11)
ax.set_title("Latency vs model complexity against 3GPP TS 22.186 budgets",
             family=F, fontsize=12)
ax.grid(True, which="both", lw=0.3, alpha=0.4)

plt.tight_layout()
plt.savefig(r"C:\projects\v2x-detection\outputs\fig_tradeoff.png",
            bbox_inches="tight", facecolor="white")
print("Saved fig_tradeoff.png")