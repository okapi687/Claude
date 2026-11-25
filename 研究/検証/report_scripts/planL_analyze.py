"""計画L の集計と図（fig33）。入力: report_data/planL_paper1_alt.json（planL_paper1_alt.py の出力）
論文の値：α=0.2 → 0.55、0.3 → 0.50、0.4 → 0.27、0.5 → 0.12、0.6 → 0.10（図11・13・14 を目で読んだもの）
使い方: python planL_analyze.py
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
DATA = HERE.parent / "report_data" / "planL_paper1_alt.json"
OUT = HERE.parent / "report_fig" / "fig33_planL_paper1_alt.png"

PAPER = {0.2: 0.55, 0.3: 0.50, 0.4: 0.27, 0.5: 0.12, 0.6: 0.10}
SETS = [  # tag, 名前, 色, 線
    ("theta_ind", "基本ベクトルで速度一致・1個体1票", "0.45", "--"),
    ("theta_tr", "基本ベクトルで速度一致・遷移の本数", "0.45", "-"),
    ("disp_ind", "移動で速度一致・1個体1票（候補B）", "C0", "--"),
    ("disp_tr", "移動で速度一致・遷移の本数", "C0", "-"),
    ("unit_tr", "移動の単位ベクトルで速度一致・遷移の本数", "C1", "-"),
]
ALPHAS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

res = json.load(open(DATA, encoding="utf-8"))
g = defaultdict(list)
for r in res:
    g[(r["tag"], r["P"], r["alpha"])].append(r)


def m(tag, Ps, a, key):
    return float(np.mean([r[key] for P in Ps for r in g[(tag, P, a)]]))


print("## 整列度（P=10/15/20 と 3シードの平均）")
print("| 設定 | " + " | ".join(f"α={a}" for a in ALPHAS) + " |")
print("|---|" + "---|" * len(ALPHAS))
print("| 論文 | " + " | ".join(f"{PAPER[a]:.2f}" if a in PAPER else "—" for a in ALPHAS) + " |")
for tag, name, *_ in SETS:
    print(f"| {name} | " + " | ".join(f"{m(tag, (10, 15, 20), a, 'pol'):.2f}" for a in ALPHAS) + " |")
for key, title in [("phi", "θ のそろい φ"), ("dens6", "密度 Ls=6"), ("dens10", "密度 Ls=10"), ("dens25", "密度 Ls=25"),
                   ("pol_sd", "整列度の時間の揺れ（標準偏差）")]:
    print(f"\n## {title}（P=15/20）")
    for tag, name, *_ in SETS:
        print(f"| {name} | " + " | ".join(f"{m(tag, (15, 20), a, key):.2f}" for a in ALPHAS) + " |")
print("\n## P=1・P=2 の整列度（α=0.5 / 0.7）")
for tag, name, *_ in SETS:
    print(f"| {name} | " + " | ".join(f"{m(tag, (P,), a, 'pol'):.2f}" for P in (1, 2) for a in (0.5, 0.7)) + " |")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1.3, 1, 0.8]))
for tag, name, c, ls in SETS:
    ax[0].plot(ALPHAS, [m(tag, (10, 15, 20), a, "pol") for a in ALPHAS], color=c, ls=ls, marker="o", ms=4, lw=1.8, label=name)
    ax[1].plot(ALPHAS, [m(tag, (15, 20), a, "phi") for a in ALPHAS], color=c, ls=ls, marker="o", ms=4, lw=1.8)
ax[0].plot(list(PAPER), list(PAPER.values()), "k*", ms=13, label="論文（図11・13・14）")
ax[0].set_title("整列度（α そのまま・round・sequential、P=10/15/20 の平均）")
ax[0].set_xlabel("論文の α"); ax[0].set_ylabel("整列度 pL"); ax[0].set_ylim(0, 0.7); ax[0].grid(alpha=0.3)
ax[0].legend(fontsize=8)
ax[1].set_title("基本ベクトルのそろい φ（P=15/20）"); ax[1].set_xlabel("論文の α"); ax[1].set_ylim(0, 1.05); ax[1].grid(alpha=0.3)
x = np.arange(len(SETS))
for j, (a, hatch) in enumerate([(0.5, ""), (0.7, "//")]):
    ax[2].bar(x + (j - 0.5) * 0.38, [m(t, (2,), a, "pol") for t, *_ in SETS], 0.38,
              color=[c for *_, c, _ in SETS], hatch=hatch, edgecolor="k", label=f"α={a}")
ax[2].set_xticks(x); ax[2].set_xticklabels(["θ・1票", "θ・本数", "移動・1票\n（候補B）", "移動・本数", "単位・本数"], fontsize=8)
ax[2].set_title("P=2 の整列度（論文：群れが崩れる）"); ax[2].set_ylim(0, 1.1); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(OUT, dpi=120)
print("\nfig:", OUT)
