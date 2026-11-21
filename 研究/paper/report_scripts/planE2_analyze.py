"""計画E2の集計と図。引数: planE_sweep.json、planE_extra.json、planE2_sweep.json、planE2_dip.json、図1の出力 png、図2の出力 png
図1：速度一致で平均するもの3通り（基本ベクトルの向き／移動ベクトル／移動の単位ベクトル）× α の読み方2通り（そのまま・round／2倍・round）
図2：E2 の設定での密度の谷（区画 2×2、10シード）
"""
import sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = {0.3: 0.53, 0.4: 0.30, 0.5: 0.13, 0.6: 0.10}
FIG11 = {0.3: {4: 0.89, 6: 0.80, 8: 0.62, 10: 0.67, 12: 0.65, 14: 0.61, 15: 0.70, 17: 0.78, 19: 0.86},
         0.6: {4: 0.83, 6: 0.82, 8: 0.74, 10: 0.75, 12: 0.73, 13: 0.77, 15: 0.82, 18: 0.85, 20: 0.88}}
VM = {"base": "基本ベクトルの向き（今の v1）", "E2": "移動ベクトル（disp）", "E2u": "移動の単位ベクトル（unit_disp）"}
READ = {"as1_round": "α そのまま・round", "as2_round": "α 2倍・round"}

res = sum((json.load(open(f)) for f in sys.argv[1:4]), [])
g = defaultdict(list)
for r in res:
    g[(r["tag"], r["alpha"])].append(r)
alphas = sorted({r["alpha"] for r in res})


def m(tag, a, k):
    return float(np.mean([r[k] for r in g[(tag, a)]]))


print("| 設定 | " + " | ".join(f"α={a}" for a in alphas) + " | α=0.3〜0.6 の最大のずれ | φ（α=0.2 → 0.7） | 密度（区画 Ls=10） | 相互予期 / 追従 / 自由 [%] |")
print("|---|" + "---|" * (len(alphas) + 4))
print("| 論文（図11） | " + " | ".join(f"{PAPER[a]:.2f}" if a in PAPER else "—" for a in alphas) + " | — | — | 0.3 前後 | — |")
for rd in READ:
    for v in VM:
        tag = f"{rd}|{v}"
        if not g[(tag, alphas[0])]:
            continue
        dev = max(abs(m(tag, a, "pL") - PAPER[a]) for a in PAPER)
        ru = [np.mean([m(tag, a, k) for a in alphas]) * 100 for k in ("ma", "fol", "free")]
        print(f"| {READ[rd]}・{VM[v]} | " + " | ".join(f"{m(tag, a, 'pL'):.2f}" for a in alphas) +
              f" | {dev:.2f} | {m(tag, alphas[0], 'phi'):.2f} → {m(tag, alphas[-1], 'phi'):.2f} | "
              f"{np.mean([m(tag, a, 'densG') for a in alphas]):.2f} | {ru[0]:.0f} / {ru[1]:.0f} / {ru[2]:.0f} |")

C = {"base": "#0b0b0b", "E2": "#eb6834", "E2u": "#2a78d6"}
plt.rcParams.update({"font.family": ["Yu Gothic", "MS Gothic"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight"})
fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.4})
for j, rd in enumerate(READ):
    for v in VM:
        tag = f"{rd}|{v}"
        if not g[(tag, alphas[0])]:
            continue
        axes[0, j].plot(alphas, [m(tag, a, "pL") for a in alphas], "o-", color=C[v], ms=4, lw=1.8, label=VM[v])
        axes[1, j].plot(alphas, [m(tag, a, "phi") for a in alphas], "o-", color=C[v], ms=4, lw=1.8)
    axes[0, j].plot(list(PAPER), list(PAPER.values()), "D", color="#8a8984", ms=8, label="論文（図11）")
    axes[0, j].set_title(f"整列度（{READ[rd]}）"); axes[0, j].set_ylim(0, 0.8); axes[0, j].set_xlabel("論文の α")
    axes[1, j].set_title(f"θ のそろい φ（{READ[rd]}）"); axes[1, j].set_ylim(0, 1.05); axes[1, j].set_xlabel("論文の α")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03), fontsize=9)
fig.suptitle("計画E2：速度一致で平均するものを変えたとき（P=10/15/20、5シード）", y=0.99)
fig.savefig(sys.argv[5]); print("fig:", sys.argv[5])

# ---------- 密度の谷 ----------
dip = json.load(open(sys.argv[4]))
d = defaultdict(list)
for r in dip:
    d[(r["tag"], r["alpha"], r["P"])].append(r["grid2"])
Ps = sorted({r["P"] for r in dip}); As = sorted({r["alpha"] for r in dip})
tags = list(dict.fromkeys(r["tag"] for r in dip))
print("\n## E2 の設定での区画 2×2 の密度（10シード）。谷：P=8〜13 の最小 − P=5〜8 の最大（負なら谷）")
for tag in tags:
    for a in As:
        y = {P: np.mean(d[(tag, a, P)]) for P in Ps}
        se = {P: np.std(d[(tag, a, P)], ddof=1) / np.sqrt(len(d[(tag, a, P)])) for P in Ps}
        pmin = min(range(8, 14), key=lambda P: y[P]); pmax = max(range(5, 9), key=lambda P: y[P])
        print(f"{tag} α={a}: " + " ".join(f"{y[P]:.2f}" for P in Ps) +
              f" | 谷 {y[pmin] - y[pmax]:+.2f}（P={pmax}→{pmin}、標準誤差 {se[pmin]:.2f}）")
CA = {0.3: "#2a78d6", 0.4: "#1baf7a", 0.5: "#e0a020", 0.6: "#eb6834"}
fig, axes = plt.subplots(1, len(tags), figsize=(5.2 * len(tags), 4.2), sharey=True)
for ax, tag in zip(axes, tags):
    for a in As:
        y = [np.mean(d[(tag, a, P)]) for P in Ps]
        e = [np.std(d[(tag, a, P)], ddof=1) / np.sqrt(len(d[(tag, a, P)])) for P in Ps]
        ax.errorbar(Ps, y, yerr=e, fmt="o-", color=CA[a], ms=3, lw=1.4, capsize=2, label=f"α={a}")
    for a, mk in [(0.3, "D"), (0.6, "s")]:
        ax.plot(list(FIG11[a]), list(FIG11[a].values()), mk, color=CA[a], mfc="white", ms=6, mew=1.5)
    ax.axvspan(7.5, 12.5, color="#e4e3df", alpha=0.6, lw=0)
    rd, v = tag.split("|")
    ax.set_title(f"{READ[rd]}・{'移動ベクトル' if v == 'disp' else '移動の単位ベクトル'}", fontsize=10)
    ax.set_xlabel("P"); ax.set_ylim(0.2, 1.0)
axes[0].set_ylabel("密度（区画 2×2）"); axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("計画E2：速度一致で移動を平均したときの密度の谷（白抜きは論文の図11、灰色は P=8〜12）", y=1.03)
fig.savefig(sys.argv[6]); print("fig:", sys.argv[6])
