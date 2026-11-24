"""計画M の集計と図（fig34）。入力: report_data/planM_Psweep.json（planM_Psweep.py の出力）
論文の値（目で読んだもの、論文1_図をつくる.ipynb と同じ）
  図11 の整列度（P が大きい所）：α=0.3 → 0.53、0.4 → 0.30、0.5 → 0.13、0.6 → 0.10
  図11 の密度（区画 Ls=25）：α=0.3 と 0.6 の P ごとの値
使い方: python planM_analyze.py
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "report_data" / "planM_Psweep.json"
OUT = HERE.parent / "report_fig" / "fig34_planM_Psweep.png"

FIG11_POL = {0.3: 0.53, 0.4: 0.30, 0.5: 0.13, 0.6: 0.10}
FIG11_DENS = {0.3: {1.5: 0.31, 4: 0.89, 6: 0.80, 8: 0.62, 10: 0.67, 12: 0.65, 14: 0.61, 15: 0.70, 17: 0.78,
                    19: 0.86, 21: 0.88, 24: 0.91, 30: 0.93},
              0.6: {1.5: 0.32, 4: 0.83, 6: 0.82, 8: 0.74, 10: 0.75, 12: 0.73, 13: 0.77, 15: 0.82, 18: 0.85,
                    20: 0.88, 24: 0.91, 30: 0.95}}
NAMES = {"disp_tr": "移動で速度一致・遷移の本数", "disp_ind": "候補B（移動で速度一致・1個体1票）"}
COL = {0.3: "C0", 0.4: "C2", 0.5: "C1", 0.6: "C3", 0.7: "C4"}

res = json.load(open(DATA, encoding="utf-8"))
g = defaultdict(list)
for r in res:
    g[(r["tag"], r["P"], r["alpha"])].append(r)
Ps = sorted({r["P"] for r in res})
alphas = sorted({r["alpha"] for r in res})


def m(tag, P, a, key):
    return float(np.mean([r[key] for r in g[(tag, P, a)]]))


for tag in NAMES:
    for key, title in [("pol", "整列度"), ("dens25", "密度 Ls=25"), ("dens10", "密度 Ls=10"), ("free", "自由移動の割合"),
                       ("fol", "追従の割合")]:
        print(f"\n## {NAMES[tag]}：{title}（3シードの平均）")
        print("| α | " + " | ".join(f"P={P}" for P in Ps) + " |")
        print("|---|" + "---|" * len(Ps))
        for a in alphas:
            print(f"| {a} | " + " | ".join(f"{m(tag, P, a, key):.2f}" for P in Ps) + " |")

fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
for a in alphas:
    for tag, ls in (("disp_tr", "-"), ("disp_ind", "--")):
        lab = f"α={a}" if tag == "disp_tr" else None
        ax[0].plot(Ps, [m(tag, P, a, "pol") for P in Ps], color=COL[a], ls=ls, marker="o", ms=3, label=lab)
    if a in FIG11_POL:
        ax[0].plot([24, 30], [FIG11_POL[a]] * 2, color=COL[a], lw=5, alpha=0.5)
ax[0].set_title("整列度（実線：遷移の本数、破線：候補B）\n右端の太い横線：論文 図11 の値")
ax[0].set_xlabel("P"); ax[0].set_ylabel("整列度 pL"); ax[0].set_ylim(0, 1.1); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
for k, a in enumerate((0.3, 0.6)):
    x = ax[1 + k]
    for tag, ls in (("disp_tr", "-"), ("disp_ind", "--")):
        x.plot(Ps, [m(tag, P, a, "dens25") for P in Ps], color=COL[a], ls=ls, marker="o", ms=3, label=NAMES[tag] + "（Ls=25）")
        x.plot(Ps, [m(tag, P, a, "dens10") for P in Ps], color="0.5", ls=ls, marker=".", ms=3, label=NAMES[tag] + "（Ls=10）")
    x.plot(list(FIG11_DENS[a]), list(FIG11_DENS[a].values()), "k*", ms=9, label="論文 図11（Ls=25 で合う）")
    x.set_title(f"密度（α={a}）"); x.set_xlabel("P"); x.set_ylabel("密度"); x.set_ylim(0, 1); x.grid(alpha=0.3)
    x.legend(fontsize=7, loc="lower right")
fig.suptitle("計画M：P を変えたとき（α そのまま・round・sequential、3シードの平均、t=300〜900）")
fig.tight_layout(); fig.savefig(OUT, dpi=120)
print("\nfig:", OUT)
