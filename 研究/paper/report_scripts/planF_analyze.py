"""計画Fの集計と図。引数: planF_dip.py の出力 json、図の出力 png
密度の谷：区画 2×2 の密度が、P=8〜12 で前後（P=5〜6 と P=15〜20）より下がるか。10シードの平均と標準誤差で見る
論文の図11 の赤線（目で読んだもの）：α=0.3 は P=4〜5 で 0.89、P=8〜14 で 0.60〜0.67、P=20 で 0.88。α=0.6 は P=4〜6 で 0.83、P=8〜12 で 0.73〜0.75、P=20 で 0.88
"""
import sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG11 = {0.3: {4: 0.89, 6: 0.80, 8: 0.62, 10: 0.67, 12: 0.65, 14: 0.61, 15: 0.70, 17: 0.78, 19: 0.86},
         0.6: {4: 0.83, 6: 0.82, 8: 0.74, 10: 0.75, 12: 0.73, 13: 0.77, 15: 0.82, 18: 0.85, 20: 0.88}}
res = json.load(open(sys.argv[1]))
g = defaultdict(list)
for r in res:
    g[(r["tag"], r["alpha"], r["P"])].append(r)
Ps = sorted({r["P"] for r in res}); As = sorted({r["alpha"] for r in res})


def ms(tag, a, P, k):
    v = np.array([r[k] for r in g[(tag, a, P)]])
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


for tag in ["v1", "as2_round"]:
    print(f"\n## {tag}：区画 2×2 の密度（10シードの平均 ± 標準誤差）")
    for a in As:
        print(f"α={a}: " + " ".join(f"P{P}:{ms(tag, a, P, 'grid2')[0]:.2f}±{ms(tag, a, P, 'grid2')[1]:.2f}" for P in Ps))
    print("谷の深さ（P=8〜12 の平均 − 前後 P=5〜6 と P=15〜20 の平均の小さい方。負なら谷）")
    for a in As:
        mid = np.mean([ms(tag, a, P, "grid2")[0] for P in range(8, 13)])
        lo = np.mean([ms(tag, a, P, "grid2")[0] for P in (5, 6)])
        hi = np.mean([ms(tag, a, P, "grid2")[0] for P in range(15, 21)])
        print(f"  α={a}: P8〜12 {mid:.3f} / P5〜6 {lo:.3f} / P15〜20 {hi:.3f} → {mid - min(lo, hi):+.3f}")

C = {0.3: "#2a78d6", 0.4: "#1baf7a", 0.5: "#e0a020", 0.6: "#eb6834"}
plt.rcParams.update({"font.family": ["Yu Gothic", "MS Gothic"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
for ax, tag, nm in [(axes[0], "v1", "α そのまま・floor（v1）"), (axes[1], "as2_round", "α 2倍・round")]:
    for a in As:
        y = [ms(tag, a, P, "grid2") for P in Ps]
        ax.errorbar(Ps, [v[0] for v in y], yerr=[v[1] for v in y], fmt="o-", color=C[a], ms=3.5, lw=1.5, capsize=2, label=f"α={a}")
    for a, mk in [(0.3, "D"), (0.6, "s")]:
        ax.plot(list(FIG11[a]), list(FIG11[a].values()), mk, color=C[a], mfc="white", ms=6, mew=1.5,
                label=f"論文 図11 α={a}")
    ax.axvspan(7.5, 12.5, color="#e4e3df", alpha=0.6, lw=0)
    ax.set_xlabel("P"); ax.set_ylabel("密度（区画 2×2）"); ax.set_ylim(0.2, 1.0); ax.set_title(nm)
axes[0].legend(fontsize=8, ncol=2, loc="lower right")
fig.suptitle("計画F：密度の谷（灰色は P=8〜12、10シードの平均 ± 標準誤差）", y=1.02)
fig.savefig(sys.argv[2]); print("\nfig:", sys.argv[2])
