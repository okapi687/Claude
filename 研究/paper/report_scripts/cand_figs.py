"""論文と実装の違いの候補の図。引数: 出力フォルダ"""
import sys, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
plt.rcParams.update({"font.family": ["Yu Gothic", "MS Gothic"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "axes.titlesize": 11, "legend.frameon": False,
                     "savefig.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 2})
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK = "#0b0b0b"


def load(*fns):
    g = defaultdict(list)
    for fn in fns:
        for r in json.load(open(f"{SP}/{fn}")):
            g[(r["tag"], r["P"], round(r["alpha"], 2))].append(r)
    return g


# ---------- 図12：α を2倍に読んだとき ----------
g1 = load("res_cand_p1.json", "res_cand_p1_wide.json")
g2 = load("res_cand_p2.json", "res_cand_p2_wide.json")
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1.2, 1, 1]})
ax = axes[0]
A = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
y = [np.mean([r["pL"] for r in g1[("v1_round", 15, a)]]) for a in A]
ax.plot(A[:6], y[:6], "o-", color=C1, ms=6, label="α をそのまま（ξ∈[−απ, απ]）")
ax.plot(np.array(A[2:]) / 2, y[2:], "o-", color=C2, ms=6, label="α を2倍に読む（ξ∈[−2απ, 2απ]）")
ax.plot([0.25, 0.3, 0.4, 0.5, 0.6], [0.55, 0.5, 0.27, 0.12, 0.10], "D", color=INK, ms=7, label="論文（図11・13・14）")
ax.set_title("論文1：整列度（P=15、round）"); ax.set_xlabel("論文の α"); ax.set_ylim(0, 0.8); ax.legend(fontsize=8.5, loc="upper right")

cases = [("青", 10, 0.1), ("赤", 40, 0.2), ("緑", 40, 0.5)]
paper = {"pol": [0.2, 0.93, 0.35], "dens": [0.65, 4.0, 3.2]}
series = [("今（α そのまま、式どおりの ω）", "v1_principal", 1, C1), ("α を2倍、式どおりの ω", "v1_principal", 2, C2),
          ("α を2倍、ω を移動側に（v1）", "v1", 2, C3)]
for ax, key, title, ymax in [(axes[1], "pol", "論文2：整列度（図3）", 1.05), (axes[2], "dens", "論文2：密度（図3）", 5.2)]:
    x = np.arange(3); w = 0.2
    for j, (name, tag, sc, col) in enumerate(series):
        v = [np.mean([r[key] for r in g2[(tag, P, round(a * sc, 2))]]) for _, P, a in cases]
        ax.bar(x + (j - 1.5) * w, v, w * 0.9, color=col, label=name)
    ax.bar(x + 1.5 * w, paper[key], w * 0.9, color=INK, label="論文")
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\nP={P}, α={a}" for c, P, a in cases]); ax.set_ylim(0, ymax)
    ax.set_title(title); ax.grid(axis="x", visible=False)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.62, -0.1), fontsize=9)
fig.suptitle("α の範囲を2倍（ξ∈[−2απ, 2απ]）に読んだとき", fontsize=11, y=1.03)
fig.savefig(f"{OUT}/fig12_alpha_double.png"); plt.close(fig)

# ---------- 図13：図8 の格子のスナップショット ----------
res = json.load(open(f"{SP}/res_fig8.json"))
snap = {(r["scale"], r["im"], r["P"], r["alpha"]): r for r in res if r["seed"] == 0}
Ps, As = [1, 2, 5, 10, 20], [0.3, 0.5, 0.7, 0.9]
fig, axes = plt.subplots(4, 11, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1] * 5 + [0.25] + [1] * 5})
for i, a in enumerate(As):
    axes[i, 5].axis("off")
    for blk, sc in enumerate([1, 2]):
        for j, P in enumerate(Ps):
            ax = axes[i, j + blk * 6]
            h = np.array(snap[(sc, "round", P, a)]["hist"])  # (10, N, 2)
            for k in range(h.shape[1]):
                tr = h[:, k, :]
                jump = np.any(np.abs(np.diff(tr, axis=0)) > 25, axis=1)
                seg = np.split(tr, np.flatnonzero(jump) + 1)
                for sgm in seg:
                    ax.plot(sgm[:, 0], sgm[:, 1], color=INK, lw=0.5)
            ax.plot(h[-1, :, 0], h[-1, :, 1], "s", color=INK, ms=1.6)
            ax.set_xlim(0, 50); ax.set_ylim(0, 50); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color("#c3c2bd"); sp.set_linewidth(0.6)
            if i == 0:
                ax.set_title(f"P={P}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"α={a}", fontsize=9)
fig.text(0.31, 0.955, "α をそのまま", ha="center", fontsize=11)
fig.text(0.735, 0.955, "α を2倍に読む", ha="center", fontsize=11)
fig.suptitle("論文1 図8 と同じ格子（round、t=591〜600 の軌跡、L=4, R(Nf)=2, R(M)=3）", fontsize=11, y=1.03)
fig.savefig(f"{OUT}/fig13_fig8_grid.png"); plt.close(fig)
