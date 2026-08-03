"""Рис. 7v3: простая гистограмма — pull (мм) по моделям, smart vs stupid, 2 панели."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ["Magma", "SpatialVLA", "InternVLA", "RLDX2", "Xiaomi", "xVLA", "GR00T"]
G_SMART = [-4.5, 4.7, -8.8, -1.2, -2.5, 0.2, -9.6]
G_STUPID = [-3.2, -13.6, -14.5, 0.6, 0.8, -2.9, 6.4]
G_STARS = {("SpatialVLA", "stupid"): "*", ("InternVLA", "smart"): "**",
           ("InternVLA", "stupid"): "**"}
R_SMART = [-7.3, 2.7, 3.7, 0.8, -2.5, -5.9, -3.1]
R_STUPID = [-4.2, -0.3, 12.5, 4.6, -1.3, -2.4, -3.2]
R_STARS = {("InternVLA", "stupid"): "**", ("xVLA", "smart"): "*"}

SURF = "#fcfcfb"; T1 = "#0b0b0b"; T2 = "#52514e"; GRID = "#e9e8e4"
C_SMART = "#2a78d6"; C_STUPID = "#eb6834"
W = 0.36

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
fig.patch.set_facecolor(SURF)
x = np.arange(len(MODELS))

for ax, sm, st, stars, title, up, dn in [
    (axes[0], G_SMART, G_STUPID, G_STARS, "ГЕНДЕР", "к мужчине ↑", "↓ к женщине"),
    (axes[1], R_SMART, R_STUPID, R_STARS, "РАСА", "к белому ↑", "↓ к небелому"),
]:
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax.axhline(0, color=T2, lw=1.2, zorder=2)
    b1 = ax.bar(x - W / 2, sm, W * 0.94, color=C_SMART, zorder=3)
    b2 = ax.bar(x + W / 2, st, W * 0.94, color=C_STUPID, zorder=3)
    for xi, (m, v1, v2) in enumerate(zip(MODELS, sm, st)):
        for v, q, dx in [(v1, "smart", -W / 2), (v2, "stupid", W / 2)]:
            s = stars.get((m, q))
            if s:
                ax.annotate(s, (xi + dx, v), xytext=(0, 4 if v > 0 else -14),
                            textcoords="offset points", ha="center",
                            color=T1, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, color=T1, fontsize=10.5, rotation=20, ha="right")
    ax.set_title(title, color=T1, fontsize=13, pad=8, fontweight="bold")
    ax.set_ylim(-18, 16)
    ax.tick_params(colors=T2, labelsize=10)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.text(0.005, 0.98, up, transform=ax.transAxes, color=T2, fontsize=9.5,
            va="top", style="italic")
    ax.text(0.005, 0.02, dn, transform=ax.transAxes, color=T2, fontsize=9.5,
            va="bottom", style="italic")

axes[0].set_ylabel("притяжение кубика, мм", color=T2, fontsize=10.5)
leg = fig.legend([plt.Rectangle((0, 0), 1, 1, color=C_SMART),
                  plt.Rectangle((0, 0), 1, 1, color=C_STUPID)],
                 ["вопрос «кто умный?»", "вопрос «кто глупый?»"],
                 loc="upper center", ncol=2, frameon=False,
                 bbox_to_anchor=(0.5, 1.02), fontsize=11.5)
for t in leg.get_texts():
    t.set_color(T1)
fig.suptitle("Интеллект-вопрос: куда робот тянет кубик (7 VLA, пары noswap/swap)",
             color=T1, fontsize=14.5, y=1.1, fontweight="bold")
fig.text(0.5, -0.06, "* p<.05   ** p<.01 (t-тест против нуля)",
         color=T2, fontsize=10, ha="center")
fig.tight_layout(w_pad=2.5)
out = "/workspace/moskalenko/bias-vla-benchmark-main/docs/figs/fig7_smart_bars.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURF)
print("saved", out)
