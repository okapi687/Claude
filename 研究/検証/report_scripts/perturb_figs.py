"""外部摂動の確認の図。引数: 出力フォルダ"""
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
INK2 = "#52514e"

g = defaultdict(lambda: defaultdict(list))
for fn in ["res_perturb.json", "res_perturb_round.json"]:
    for r in json.load(open(f"{SP}/{fn}")):
        g[r["tag"]][(r["alpha"], r["eps"])].append(r)


def line(tag, key, fix_alpha=None, fix_eps=None):
    d = g[tag]
    ks = sorted(k for k in d if (fix_alpha is None or abs(k[0] - fix_alpha) < 1e-9) and (fix_eps is None or abs(k[1] - fix_eps) < 1e-9))
    x = [k[1] if fix_alpha is not None else k[0] for k in ks]
    return np.array(x), np.array([np.mean([r[key] for r in d[k]]) for k in ks])


fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
ax = axes[0]
for tag, eps, name, col in [("ma_noise", 0.0, "floor・摂動なし（v1）", C1), ("ma_noise", 0.3, "floor・摂動 ε=0.3", C2),
                            ("round_noise", 0.3, "round・摂動 ε=0.3", C3)]:
    x, y = line(tag, "pol", fix_eps=eps)
    ax.plot(x, y, "o-", color=col, ms=6, label=name)
ax.plot([0.25, 0.3, 0.4, 0.5, 0.6], [0.55, 0.5, 0.27, 0.12, 0.10], "D", color="#0b0b0b", ms=7, label="論文（図11・13・14）")
ax.axhline(0.18, color=INK2, lw=1, ls="--")
ax.text(0.7, 0.19, "floor で向きが完全にばらばらのときの値", ha="right", va="bottom", fontsize=8.5, color=INK2)
ax.set_title("整列度（P=15）"); ax.set_xlabel("α"); ax.set_ylim(0, 1); ax.legend(fontsize=8.5, loc="upper right")

for ax, key, title in [(axes[1], "phi", "基本ベクトルの向きのそろい"), (axes[2], "dens", "密度（Ls=10）")]:
    for tag, a, name, col in [("p1_noise", 0.7, "相互予期なし（P=1）", C1), ("ma_noise", 0.5, "P=15, α=0.5", C2),
                              ("p30_noise", 0.7, "P=30, α=0.7（図10右と同じ）", C3)]:
        x, y = line(tag, key, fix_alpha=a)
        ax.plot(x, y, "o-", color=col, ms=6, label=name)
    ax.set_title(title); ax.set_xlabel("外部摂動の強さ ε（θ に ε·π·U[-1,1] を足す）"); ax.set_ylim(0, 1.05)
axes[2].legend(fontsize=8.5, loc="upper right")
fig.suptitle("論文1：速度一致のあとに θ へ外部摂動を足したとき（floor、t=300〜900 の平均、5シード）", fontsize=11, y=1.04)
fig.savefig(f"{OUT}/fig11_perturbation_paper1.png"); plt.close(fig)
