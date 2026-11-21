"""計画Hの集計と図。引数: planH_omega.py の出力 json、図の出力 png
論文の図S3（t=1500、1本のシミュレーションと思われる。目で読んだもの、11/22）
  整列度：ω=0 → 0.10、1 → 0.70、2 → 0.93、3 → 0.93、4 → 0.92、5 → 0.95、6 → 0.95、7 → 0.93、8 → 0.55、9 → 0.46、10 → 0.73
  密度  ：ω=0 → 1.9、1 → 2.4、2 → 3.4、3 → 3.9、4 → 3.8、5 → 4.3、6 → 4.1、7 → 3.7、8 → 3.9、9 → 3.95、10 → 3.5
"""
import sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PPOL = {0: 0.10, 1: 0.70, 2: 0.93, 3: 0.93, 4: 0.92, 5: 0.95, 6: 0.95, 7: 0.93, 8: 0.55, 9: 0.46, 10: 0.73}
PDEN = {0: 1.9, 1: 2.4, 2: 3.4, 3: 3.9, 4: 3.8, 5: 4.3, 6: 4.1, 7: 3.7, 8: 3.9, 9: 3.95, 10: 3.5}
NAMES = {("principal", 2): "ω 式どおり・α 2倍", ("velocity", 2): "ω 説明文どおり・α 2倍",
         ("principal", 1): "ω 式どおり・α そのまま", ("velocity", 1): "ω 説明文どおり・α そのまま"}
res = json.load(open(sys.argv[1]))
g = defaultdict(list)
for r in res:
    g[(r["on"], r["scale"], r["omega"])].append(r)
OM = sorted({r["omega"] for r in res})


def end(r, k):          # t=1400〜1500 の平均（時系列は 10 ステップごと）
    return float(np.mean(r[k][140:150]))


def t_align(r, thr=0.6):
    p = np.array(r["pol"])
    hit = np.flatnonzero(p >= thr)
    return int((hit[0] + 1) * 10) if len(hit) else None


for key, nm in NAMES.items():
    print(f"\n## {nm}（5シード）")
    print("| ω | 整列度 t=1400〜1500（シードの範囲） | 論文 | 密度 | 論文 | θ のそろい φ | 整列度が 0.6 を超えた時刻（超えたシード数） |")
    print("|---|---|---|---|---|---|---|")
    for w in OM:
        rs = g[(key[0], key[1], w)]
        pe = [end(r, "pol") for r in rs]
        ta = [t_align(r) for r in rs]
        tav = [t for t in ta if t is not None]
        print(f"| {w} | {np.mean(pe):.2f}（{min(pe):.2f}〜{max(pe):.2f}） | {PPOL.get(w, '—')} | "
              f"{np.mean([end(r, 'dens') for r in rs]):.2f} | {PDEN.get(w, '—')} | {np.mean([end(r, 'phi') for r in rs]):.2f} | "
              f"{(str(int(np.median(tav))) if tav else '—')}（{len(tav)}/5） |")

C = {("principal", 2): "#2a78d6", ("velocity", 2): "#eb6834", ("principal", 1): "#2a78d6", ("velocity", 1): "#eb6834"}
LS = {2: "-", 1: "--"}
plt.rcParams.update({"font.family": ["Yu Gothic", "MS Gothic"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for key, nm in NAMES.items():
    y = [np.mean([end(r, "pol") for r in g[(key[0], key[1], w)]]) for w in OM]
    axes[0].plot(OM, y, LS[key[1]], color=C[key], marker="o", ms=3.5, lw=1.7, label=nm)
    y = [np.mean([end(r, "dens") for r in g[(key[0], key[1], w)]]) for w in OM]
    axes[1].plot(OM, y, LS[key[1]], color=C[key], marker="o", ms=3.5, lw=1.7)
axes[0].plot(list(PPOL), list(PPOL.values()), "D", color="#0b0b0b", ms=6, label="論文 図S3")
axes[1].plot(list(PDEN), list(PDEN.values()), "D", color="#0b0b0b", ms=6)
axes[0].set_title("整列度（t=1400〜1500）"); axes[0].set_xlabel("ω"); axes[0].set_ylim(0, 1.05)
axes[1].set_title("密度（t=1400〜1500）"); axes[1].set_xlabel("ω"); axes[1].set_ylim(1, 5)
ax = axes[2]
t = np.arange(1, 151) * 10
for w, col in [(0, "#0b0b0b"), (0.5, "#8a8984"), (1, "#1baf7a"), (2, "#2a78d6")]:
    for on, ls in [("principal", "-"), ("velocity", "--")]:
        rs = g[(on, 2, w)]
        ax.plot(t, np.mean([r["pol"] for r in rs], 0), ls, color=col, lw=1.4,
                label=f"ω={w}（{'式どおり' if on == 'principal' else '説明文どおり'}）")
ax.set_title("整列度の時間変化（α 2倍、5シードの平均）"); ax.set_xlabel("ステップ"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=7.5, ncol=2, loc="center right", bbox_to_anchor=(1.0, 0.42))
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.36, -0.1), fontsize=9)
fig.suptitle("計画H：図S3（ω を変えたとき、P=40・α=0.2・round・1個体1票）", y=1.02)
fig.savefig(sys.argv[2]); print("\nfig:", sys.argv[2])
