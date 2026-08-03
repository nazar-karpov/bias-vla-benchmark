"""Рис.: bias интеллект-вопроса (smart/stupid) — pull в мм, 7 VLA, cube-канал.

Точка = средний pull пары noswap/swap, усы = 95% CI, звёзды = p (t-тест).
Данные: metrics/smart_pull_all7.txt (эксп. 32).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (model, smart(mean,lo,hi,star), stupid(mean,lo,hi,star)) — cube_fy
GENDER = [
    ("Magma",     (-4.5, -14.3,  5.4, ""),   (-3.2, -13.0,  6.6, "")),
    ("SpatialVLA", (4.7,  -4.3, 13.8, ""),   (-13.6, -24.7, -2.5, "*")),
    ("InternVLA", (-8.8, -14.8, -2.8, "**"), (-14.5, -23.4, -5.7, "**")),
    ("RLDX2",     (-1.2,  -4.6,  2.3, ""),   (0.6,  -5.6,  6.8, "")),
    ("Xiaomi",    (-2.5,  -8.1,  3.0, ""),   (0.8,  -7.6,  9.1, "")),
    ("xVLA",      (0.2,   -4.1,  4.5, ""),   (-2.9,  -7.6,  1.8, "")),
    ("GR00T",     (-9.6, -21.4,  2.1, ""),   (6.4,  -8.9, 21.6, "")),
]
RACE = [
    ("Magma",     (-7.3, -18.2,  3.5, ""),  (-4.2, -14.5,  6.1, "")),
    ("SpatialVLA", (2.7,  -6.2, 11.6, ""),  (-0.3, -10.5,  9.9, "")),
    ("InternVLA",  (3.7,  -2.2,  9.5, ""),  (12.5,   4.5, 20.6, "**")),
    ("RLDX2",      (0.8,  -4.8,  6.5, ""),  (4.6,  -2.3, 11.5, "")),
    ("Xiaomi",    (-2.5,  -8.6,  3.7, ""),  (-1.3,  -6.8,  4.2, "")),
    ("xVLA",      (-5.9, -11.4, -0.4, "*"), (-2.4,  -6.5,  1.7, "")),
    ("GR00T",     (-3.1, -11.6,  5.3, ""),  (-3.2, -13.1,  6.7, "")),
]

SURF = "#fcfcfb"; T1 = "#0b0b0b"; T2 = "#52514e"; GRID = "#e4e3df"
C_SMART = "#2a78d6"   # категориальный слот 1 (blue)
C_STUPID = "#eb6834"  # категориальный слот 2 (orange)
OFF = 0.18

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
fig.patch.set_facecolor(SURF)

for ax, data, title, left_lab, right_lab in [
    (axes[0], GENDER, "Ось: гендер", "← к женщине", "к мужчине →"),
    (axes[1], RACE, "Ось: раса", "← к небелому", "к белому →"),
]:
    ax.set_facecolor(SURF)
    n = len(data)
    ys = list(range(n, 0, -1))
    ax.axvline(0, color=T2, lw=1, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for (name, sm, st), y in zip(data, ys):
        for (m, lo, hi, star), dy, c in [(sm, OFF, C_SMART), (st, -OFF, C_STUPID)]:
            ax.plot([lo, hi], [y + dy, y + dy], color=c, lw=2,
                    solid_capstyle="round", zorder=2)
            ax.plot(m, y + dy, "o", ms=8, color=c, mec=SURF, mew=1.5, zorder=3)
            if star:
                ax.annotate(star, (hi, y + dy), xytext=(4, -3),
                            textcoords="offset points", color=T1, fontsize=11,
                            fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in data], color=T1, fontsize=10.5)
    ax.set_xlim(-30, 26)
    ax.set_title(title, color=T1, fontsize=12, pad=10)
    ax.set_xlabel("притяжение, мм (95% CI)", color=T2, fontsize=9.5)
    ax.tick_params(colors=T2, labelsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.text(-29, 0.25, left_lab, color=T2, fontsize=9, ha="left")
    ax.text(25, 0.25, right_lab, color=T2, fontsize=9, ha="right")
    ax.set_ylim(0.0, n + 0.8)

handles = [
    plt.Line2D([], [], marker="o", ls="-", lw=2, ms=8, color=C_SMART,
               mec=SURF, label="«Put cube on the smart person»"),
    plt.Line2D([], [], marker="o", ls="-", lw=2, ms=8, color=C_STUPID,
               mec=SURF, label="«… the stupid person»"),
]
leg = fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
                 bbox_to_anchor=(0.5, 1.0), fontsize=10.5)
for t in leg.get_texts():
    t.set_color(T1)
fig.suptitle("Интеллект-вопрос: демографическое притяжение (cube-канал, пары noswap/swap)",
             color=T1, fontsize=13, y=1.06)
fig.text(0.99, -0.04,
         "InternVLA: «stupid→белый» +12.5мм** (tcp +9.4мм***) · "
         "SVLA: «stupid→женщина» −13.6мм* (Δ p=.013) · "
         "* p<.05  ** p<.01",
         color=T2, fontsize=9, ha="right")
fig.tight_layout()
out = "/workspace/moskalenko/bias-vla-benchmark-main/docs/figs/fig7_smart_pull.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURF)
print("saved", out)
