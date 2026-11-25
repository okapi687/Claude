"""計画Eの集計と図。引数: planE_sweep.py の出力 json、planE_extra.py の出力 json、図の出力 png
論文の値（図11、P=10〜20 の範囲を目で読んだもの、11/20 に読み直し）：α=0.3 → 0.53、0.4 → 0.30、0.5 → 0.13、0.6 → 0.10
"""
import sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = {0.3: 0.53, 0.4: 0.30, 0.5: 0.13, 0.6: 0.10}
NAMES = {"base": "今の v1", "E1": "E1 サイト単位で選ぶ", "E2": "E2 移動ベクトルで速度一致", "E1E2": "E1＋E2"}
READ = {"as1_floor": "α そのまま・floor", "as1_round": "α そのまま・round", "as2_round": "α 2倍・round"}

res = json.load(open(sys.argv[1])) + json.load(open(sys.argv[2]))
g = defaultdict(list)
for r in res:
    g[(r["tag"], r["alpha"])].append(r)
alphas = sorted({r["alpha"] for r in res})
tags = list(dict.fromkeys(r["tag"] for r in res))


def m(tag, a, k):
    return float(np.mean([r[k] for r in g[(tag, a)]]))


for rd in READ:
    print(f"\n## {READ[rd]}：整列度 pL（P=10/15/20・5シードの平均）と θ のそろい φ")
    print("| 設定 | " + " | ".join(f"α={a}" for a in alphas) + " | φ（平均） | α=0.3〜0.6 の最大のずれ | 密度（ずらす Ls=10） | 密度（区画 Ls=10） | 相互予期 / 追従 / 自由 [%] |")
    print("|---|" + "---|" * (len(alphas) + 5))
    print("| 論文（図11） | " + " | ".join(f"{PAPER[a]:.2f}" if a in PAPER else "—" for a in alphas) + " | — | — | — | — | — |")
    for v in NAMES:
        tag = f"{rd}|{v}"
        if not g[(tag, alphas[0])]:
            continue
        pl = [m(tag, a, "pL") for a in alphas]
        dev = max(abs(m(tag, a, "pL") - PAPER[a]) for a in PAPER)
        phi = np.mean([m(tag, a, "phi") for a in alphas])
        ru = [np.mean([m(tag, a, k) for a in alphas]) * 100 for k in ("ma", "fol", "free")]
        print(f"| {NAMES[v]} | " + " | ".join(f"{x:.2f}" for x in pl) +
              f" | {phi:.3f} | {dev:.2f} | {np.mean([m(tag, a, 'dens') for a in alphas]):.2f} | "
              f"{np.mean([m(tag, a, 'densG') for a in alphas]):.2f} | {ru[0]:.0f} / {ru[1]:.0f} / {ru[2]:.0f} |")
    print("φ を α ごとに：")
    for v in NAMES:
        tag = f"{rd}|{v}"
        if not g[(tag, alphas[0])]:
            continue
        print(f"  {NAMES[v]}: " + " ".join(f"{m(tag, a, 'phi'):.3f}" for a in alphas))

C = {"base": "#0b0b0b", "E1": "#2a78d6", "E2": "#eb6834", "E1E2": "#1baf7a"}
plt.rcParams.update({"font.family": ["Yu Gothic", "MS Gothic"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight"})
fig, axes = plt.subplots(2, 3, figsize=(17, 7.4), gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.4})
for j, rd in enumerate(READ):
    ax = axes[0, j]
    for v in NAMES:
        tag = f"{rd}|{v}"
        if g[(tag, alphas[0])]:
            ax.plot(alphas, [m(tag, a, "pL") for a in alphas], "o-", color=C[v], ms=4, lw=1.8, label=NAMES[v])
    ax.plot(list(PAPER), list(PAPER.values()), "D", color="#8a8984", ms=8, label="論文（図11）")
    ax.set_title(f"整列度（{READ[rd]}）"); ax.set_xlabel("論文の α"); ax.set_ylim(0, 0.8)
    ax = axes[1, j]
    for v in NAMES:
        tag = f"{rd}|{v}"
        if g[(tag, alphas[0])]:
            ax.plot(alphas, [m(tag, a, "phi") for a in alphas], "o-", color=C[v], ms=4, lw=1.8)
    ax.set_title(f"θ のそろい φ（{READ[rd]}）"); ax.set_xlabel("論文の α"); ax.set_ylim(0, 1.05)
h, l = axes[0, 2].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03), fontsize=9)
fig.suptitle("計画E：同率サイトの選び方（E1）と速度一致で平均するもの（E2）を変えたとき（P=10/15/20、5シード）", y=0.99)
fig.savefig(sys.argv[3]); print("\nfig:", sys.argv[3])
