"""計画2（fig17）から literal を除き、sequential の4通り（alpha_scale × Int）だけを描く図（fig32）
入力: report_data/plan2_grid.json（plan2_grid.py の出力）。新しい計算はしない
論文の値：α=0.2 → 0.55、0.3 → 0.50、0.4 → 0.27、0.5 → 0.12、0.6 → 0.10（図11・13・14 を目で読んだもの）
使い方: python plan2_seq_only.py
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
DATA = HERE.parent / "report_data" / "plan2_grid.json"
OUT = HERE.parent / "report_fig" / "fig32_plan2_sequential.png"

PAPER = {0.2: 0.55, 0.3: 0.50, 0.4: 0.27, 0.5: 0.12, 0.6: 0.10}
# (tag, 表示名, 色, 線, 印)  色＝α の読み方、線と印＝Int
SETS = [
    ("as1_floor_sequential", "α そのまま・floor（v1 の既定）", "C0", "--", "s"),
    ("as1_round_sequential", "α そのまま・round", "C0", "-", "o"),
    ("as2_floor_sequential", "α 2倍・floor", "C1", "--", "s"),
    ("as2_round_sequential", "α 2倍・round（候補A）", "C1", "-", "o"),
]

res = [r for r in json.load(open(DATA, encoding="utf-8")) if r["tag"].endswith("_sequential")]
alphas = sorted({r["alpha"] for r in res})
Ps = sorted({r["P"] for r in res})
agg = defaultdict(list)
for r in res:
    agg[(r["tag"], r["P"], r["alpha"])].append(r["pL"])
    agg[(r["tag"], "all", r["alpha"])].append(r["pL"])

print("| 設定 | " + " | ".join(f"α={a}" for a in alphas) + " |")
print("|---|" + "---|" * len(alphas))
print("| 論文 | " + " | ".join(f"{PAPER[a]:.2f}" if a in PAPER else "—" for a in alphas) + " |")
for tag, name, *_ in SETS:
    print(f"| {name} | " + " | ".join(f"{np.mean(agg[(tag, 'all', a)]):.2f}" for a in alphas) + " |")

panels = Ps + ["all"]
fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4), sharey=True)
for ax, P in zip(axes, panels):
    for i, (tag, name, c, ls, mk) in enumerate(SETS):
        y = [np.mean(agg[(tag, P, a)]) for a in alphas]
        e = [np.std(agg[(tag, P, a)]) for a in alphas]
        x = [a + (i - 1.5) * 0.008 for a in alphas]     # 誤差棒が重ならないよう少し横にずらす
        ax.errorbar(x, y, yerr=e, color=c, ls=ls, marker=mk, ms=5, capsize=2, lw=1.8, label=name)
    ax.plot(list(PAPER), list(PAPER.values()), "k*", ms=13, label="論文（図11・13・14）")
    ax.set_title("P=10/15/20 の平均" if P == "all" else f"P={P}")
    ax.set_xlabel("論文の α"); ax.set_ylim(0, 0.75); ax.grid(alpha=0.3)
axes[0].set_ylabel("整列度 pL（|平均の移動| / L）")
axes[0].legend(fontsize=8.5, loc="lower left")
fig.suptitle("計画2 のうち sequential（競合は空いている次に人気のサイトへ）だけ：α の読み方 × Int の4通り"
             "（5シードの平均 ± 標準偏差。右端は 15本の平均 ± 標準偏差。点は見やすいよう横に少しずらした）")
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("fig:", OUT)
