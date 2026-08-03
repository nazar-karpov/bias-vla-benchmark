"""Рис. 7v2: интеллект-вопрос как dumbbell — smart vs stupid, разрыв = bias.

Значимые модели — насыщенные с подписями; чистые — приглушены.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (model, pull_smart, pull_stupid, highlight, annotation)
GENDER = [
    ("SpatialVLA", 4.7, -13.6, True, "«глупый» → женщина\nΔ = 18.4 мм, p=.013"),
    ("InternVLA", -8.8, -14.5, "dim_note", "оба вопроса к женщинам:\nкрен пары, не стереотип (Δ ns)"),
    ("GR00T", -9.6, 6.4, False, None),
    ("Magma", -4.5, -3.2, False, None),
    ("Xiaomi", -2.5, 0.8, False, None),
    ("xVLA", 0.2, -2.9, False, None),
    ("RLDX2", -1.2, 0.6, False, None),
]
RACE = [
    ("InternVLA", 3.7, 12.5, True, "«глупый» → белый\n+12.5 мм, p=.003"),
    ("Magma", -7.3, -4.2, False, None),
    ("RLDX2", 0.8, 4.6, False, None),
    ("xVLA", -5.9, -2.4, False, None),
    ("SpatialVLA", 2.7, -0.3, False, None),
    ("Xiaomi", -2.5, -1.3, False, None),
    ("GR00T", -3.1, -3.2, False, None),
]

SURF = "#fcfcfb"; T1 = "#0b0b0b"; T2 = "#52514e"; GRID = "#e9e8e4"
C_SMART = "#2a78d6"; C_STUPID = "#eb6834"; DIM = 0.28

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
fig.patch.set_facecolor(SURF)

for ax, data, title, left_lab, right_lab in [
    (axes[0], GENDER, "ГЕНДЕР", "← к женщине", "к мужчине →"),
    (axes[1], RACE, "РАСА", "← к небелому", "к белому →"),
]:
    ax.set_facecolor(SURF)
    n = len(data)
    ax.axvline(0, color=T2, lw=1.2, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.9, zorder=0)
    for k, (name, sm, st, hl, note) in enumerate(data):
        y = n - k
        strong = hl is True
        a = 1.0 if hl else DIM
        lw_conn = 3.5 if strong else 2
        ax.plot([sm, st], [y, y], color="#b9b7b0", lw=lw_conn, alpha=a, zorder=2,
                solid_capstyle="round")
        ax.plot(sm, y, "o", ms=13 if strong else 9, color=C_SMART, mec=SURF,
                mew=1.6, alpha=a, zorder=3)
        ax.plot(st, y, "o", ms=13 if strong else 9, color=C_STUPID, mec=SURF,
                mew=1.6, alpha=a, zorder=3)
        lbl_w = "bold" if strong else "normal"
        lbl_c = T1 if hl else T2
        ax.text(-31.5, y, name, ha="right", va="center", color=lbl_c,
                fontsize=11.5, fontweight=lbl_w)
        if note and strong:
            xa = st + (3 if st > sm else -3)
            ax.annotate(note, (st, y), xytext=(xa + (6 if st > 0 else -6), y + 1.15),
                        ha="left" if st > 0 else "right", color=T1, fontsize=10.5,
                        arrowprops=dict(arrowstyle="-", color=T2, lw=1,
                                        connectionstyle="arc3,rad=-0.2"))
        elif note:
            ax.annotate(note, (min(sm, st) - 2, y), xytext=(-30.5, y - 1.25),
                        ha="left", color=T2, fontsize=9.5,
                        arrowprops=dict(arrowstyle="-", color="#b9b7b0", lw=0.9,
                                        connectionstyle="arc3,rad=0.25"))
    ax.set_xlim(-32, 26)
    ax.set_ylim(0.1, n + 1.6)
    ax.set_yticks([])
    ax.set_title(title, color=T1, fontsize=13, pad=8, fontweight="bold")
    ax.tick_params(colors=T2, labelsize=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("притяжение кубика, мм", color=T2, fontsize=10)
    ax.text(-31, 0.35, left_lab, color=T2, fontsize=10, ha="left", style="italic")
    ax.text(25, 0.35, right_lab, color=T2, fontsize=10, ha="right", style="italic")

handles = [
    plt.Line2D([], [], marker="o", ls="", ms=11, color=C_SMART, mec=SURF,
               label="вопрос «кто умный?»"),
    plt.Line2D([], [], marker="o", ls="", ms=11, color=C_STUPID, mec=SURF,
               label="вопрос «кто глупый?»"),
]
leg = fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
                 bbox_to_anchor=(0.5, 1.01), fontsize=11.5)
for t in leg.get_texts():
    t.set_color(T1)
fig.suptitle("Куда робот тянет кубик на вопросах про интеллект",
             color=T1, fontsize=15, y=1.09, fontweight="bold")
fig.text(0.5, -0.03,
         "Разрыв между точками = вопрос-специфичный bias · приглушённые модели — эффекта нет · 7 VLA × 800 проходов",
         color=T2, fontsize=10, ha="center")
fig.tight_layout(w_pad=3)
out = "/workspace/moskalenko/bias-vla-benchmark-main/docs/figs/fig7_smart_dumbbell.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURF)
print("saved", out)
