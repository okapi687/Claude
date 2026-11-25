"""計画Dの集計と図。引数: planD_blue.py の出力 json、図の出力 png
横軸は「実際の扇の半分の角度」（= alpha_scale × α × 180°）。α そのままと2倍の結果は、この軸で重なるはず
"""
import sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = dict(pol=0.2, dens=0.65, ma=63, fol=5, free=32)
res = json.load(open(sys.argv[1]))
g = defaultdict(list)
for r in res:
    g[(r["P"], round(r["scale"] * r["alpha"] * 180, 1))].append(r)


def curve(P, key):
    xs = sorted(x for (p, x) in g if p == P)
    ys = []
    for x in xs:
        v = g[(P, x)]
        ys.append(np.mean([r["rules"][key] * 100 for r in v]) if isinstance(key, int) else np.mean([r[key] for r in v]))
    return xs, ys


print("| 実際の扇（±度） | 相互予期 / 追従 / 自由移動 [%] | 整列度 | 密度 |")
print("|---|---|---|---|")
for x in sorted(x for (p, x) in g if p == 10):
    v = g[(10, x)]; ru = np.mean([r["rules"] for r in v], 0) * 100
    print(f"| {x:.0f} | {ru[0]:.0f} / {ru[1]:.0f} / {ru[2]:.0f} | {np.mean([r['pol'] for r in v]):.2f} | {np.mean([r['dens'] for r in v]):.2f} |")

C1, C2, C3, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.family": ["Yu Gothic", "MS Gothic"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))
for ax, key, ylab, pv in [(axes[0], 0, "相互予期の割合 [%]", PAPER["ma"]), (axes[1], "dens", "密度（周り 3×3 の個体数）", PAPER["dens"]),
                          (axes[2], "pol", "整列度", PAPER["pol"])]:
    for P, col in [(8, C3), (10, C1), (12, C2)]:
        xs, ys = curve(P, key)
        ax.plot(xs, ys, "o-", color=col, ms=4, lw=1.8 if P == 10 else 1.2, label=f"P={P}")
    ax.axhline(pv, color=INK, ls="--", lw=1)
    for x, nm in [(18, "論文の青を\nα そのまま（±18°）"), (36, "α 2倍\n（±36°）")]:
        ax.axvline(x, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("実際の扇の半分の角度（度）"); ax.set_ylabel(ylab)
axes[0].text(17, 85.5, "α そのまま ±18°", fontsize=8.5, color="#5a5955", ha="right")
axes[0].text(38, 85.5, "α 2倍 ±36°", fontsize=8.5, color="#5a5955")
axes[0].text(80, PAPER["ma"] + 1.5, "論文の青 63%", fontsize=8.5)
axes[0].set_title("相互予期の割合（図S2 の青）"); axes[1].set_title("密度（図3 の青）"); axes[2].set_title("整列度（図3 の青）")
axes[0].legend(fontsize=8.5, loc="lower right"); axes[1].set_ylim(0.3, 0.8); axes[2].set_ylim(0, 0.3)
fig.suptitle("論文2の青（P=10）：扇の広さとルールの割合（ω 式どおり、round、5シード、t=1000〜1500）", y=1.03)
fig.savefig(sys.argv[2]); print("fig:", sys.argv[2])
