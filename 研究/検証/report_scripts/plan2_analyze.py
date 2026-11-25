"""計画2の集計。引数: plan2_grid.py の出力 json、図の出力 png
整列度は v1 と同じ pL（|平均変位| / L）を主に見る。pUnit（単位変位の平均の長さ）も並べる
論文の値：α=0.2 → 0.55、0.3 → 0.50、0.4 → 0.27、0.5 → 0.12、0.6 → 0.10（図11・13・14 を目で読んだもの）
"""
import sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = {0.2: 0.55, 0.3: 0.50, 0.4: 0.27, 0.5: 0.12, 0.6: 0.10}
TARGET = (0.3, 0.4, 0.5)

res = json.load(open(sys.argv[1]))
tags = list(dict.fromkeys(r["tag"] for r in res))
alphas = sorted({r["alpha"] for r in res})
Ps = sorted({r["P"] for r in res})

agg = defaultdict(list)
for r in res:
    for key in ("pL", "pUnit", "dens", "ma", "fol", "free"):
        agg[(r["tag"], r["P"], r["alpha"], key)].append(r[key])
        agg[(r["tag"], "all", r["alpha"], key)].append(r[key])


def m(tag, P, a, key):
    v = agg[(tag, P, a, key)]
    return float(np.mean(v)) if v else np.nan


def sd(tag, P, a, key):
    v = agg[(tag, P, a, key)]
    return float(np.std(v)) if v else np.nan


print("## 整列度 pL（P=10/15/20 と 5シードの平均）")
print("| 設定 | " + " | ".join(f"α={a}" for a in alphas) + " | α=0.3/0.4/0.5 の最大のずれ | 3点とも ±0.1 |")
print("|---|" + "---|" * (len(alphas) + 2))
print("| 論文 | " + " | ".join(f"{PAPER[a]:.2f}" if a in PAPER else "—" for a in alphas) + " | — | — |")
for t in tags:
    dev = max(abs(m(t, "all", a, "pL") - PAPER[a]) for a in TARGET)
    ok = all(abs(m(t, P, a, "pL") - PAPER[a]) <= 0.1 for P in Ps for a in TARGET)
    print(f"| {t} | " + " | ".join(f"{m(t, 'all', a, 'pL'):.2f}" for a in alphas) + f" | {dev:.2f} | {'○' if ok else '×'} |")

for key, name in [("pUnit", "整列度 pUnit（単位変位）"), ("dens", "密度 Ls=10")]:
    print(f"\n## {name}（P=10/15/20 と 5シードの平均）")
    print("| 設定 | " + " | ".join(f"α={a}" for a in alphas) + " |")
    print("|---|" + "---|" * len(alphas))
    for t in tags:
        print(f"| {t} | " + " | ".join(f"{m(t, 'all', a, key):.2f}" for a in alphas) + " |")

print("\n## P ごとの pL（α=0.3 / 0.4 / 0.5、平均 ± 標準偏差）")
print("| 設定 | " + " | ".join(f"P={P}" for P in Ps) + " |")
print("|---|" + "---|" * len(Ps))
for t in tags:
    cells = [" / ".join(f"{m(t, P, a, 'pL'):.2f}±{sd(t, P, a, 'pL'):.2f}" for a in TARGET) for P in Ps]
    print(f"| {t} | " + " | ".join(cells) + " |")

print("\n## ルールの割合（相互予期 / 追従 / 自由移動、P=15）")
print("| 設定 | " + " | ".join(f"α={a}" for a in alphas) + " |")
print("|---|" + "---|" * len(alphas))
for t in tags:
    cells = [f"{m(t, 15, a, 'ma'):.2f}/{m(t, 15, a, 'fol'):.2f}/{m(t, 15, a, 'free'):.2f}" for a in alphas]
    print(f"| {t} | " + " | ".join(cells) + " |")

# 図：左から P=10 / 15 / 20、8通りを線で、論文の値を黒丸で
fig, axes = plt.subplots(1, len(Ps), figsize=(5 * len(Ps), 4.2), sharey=True, squeeze=False)
axes = axes[0]
colors = plt.cm.tab10(np.arange(len(tags)))
for ax, P in zip(axes, Ps):
    for t, c in zip(tags, colors):
        ls = "-" if "as2" in t else "--"
        mk = "o" if "round" in t else "s"
        ax.errorbar(alphas, [m(t, P, a, "pL") for a in alphas], yerr=[sd(t, P, a, "pL") for a in alphas],
                    ls=ls, marker=mk, ms=4, color=c, capsize=2, label=t)
    ax.plot(list(PAPER), list(PAPER.values()), "k*", ms=12, label="paper")
    ax.set_title(f"P={P}"); ax.set_xlabel("alpha"); ax.grid(alpha=0.3)
axes[0].set_ylabel("polarization pL"); axes[-1].legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=120)
print("\nfig:", sys.argv[2])
