"""計画Gの図。引数: planG_rules.py の出力 json、図の出力 png"""
import sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"v1": "v1（α そのまま・floor）", "as2_round": "α 2倍・round", "as1_round_E2": "α そのまま・round＋E2",
         "literal": "文字どおりの競合（literal）"}
C = {"v1": "#0b0b0b", "as2_round": "#2a78d6", "as1_round_E2": "#eb6834", "literal": "#1baf7a"}
res = json.load(open(sys.argv[1]))
g = defaultdict(list)
for r in res:
    g[(r["tag"], r["P"])].append(r)
Ps = sorted({r["P"] for r in res})
plt.rcParams.update({"font.family": ["Yu Gothic", "MS Gothic"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
for tag, nm in NAMES.items():
    ma = [np.mean([x["rules"][0] for x in g[(tag, P)]]) * 100 for P in Ps]
    axes[0].plot(Ps, ma, "o-", color=C[tag], ms=3.5, lw=1.7, label=nm)
    axes[1].plot(Ps, [np.mean([x["dens10"] for x in g[(tag, P)]]) for P in Ps], "o-", color=C[tag], ms=3.5, lw=1.7)
    axes[2].plot(Ps, [np.mean([x["dens25"] for x in g[(tag, P)]]) for P in Ps], "o-", color=C[tag], ms=3.5, lw=1.7)
axes[0].axhspan(98, 99.5, color="#e4e3df", lw=0); axes[0].text(16, 92, "論文2 図S2 の赤・緑（98〜99%）", fontsize=8.5, color="#5a5955")
axes[0].set_title("相互予期の割合"); axes[0].set_xlabel("P"); axes[0].set_ylabel("%"); axes[0].set_ylim(0, 102)
axes[1].axhspan(0.27, 0.33, color="#e4e3df", lw=0); axes[1].text(13, 0.235, "図13・14（Ls=10 と読んだとき）", fontsize=8.5, color="#5a5955")
axes[1].set_title("密度（区画 Ls=10、5×5）"); axes[1].set_xlabel("P"); axes[1].set_ylim(0, 0.9)
axes[2].set_title("密度（区画 Ls=25、2×2：図11 の読み方）"); axes[2].set_xlabel("P"); axes[2].set_ylim(0, 0.9)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=9)
fig.suptitle("計画G：P を変えたときの相互予期の割合と密度（論文の α=0.5、5シード）", y=1.02)
fig.savefig(sys.argv[2]); print("fig:", sys.argv[2])
