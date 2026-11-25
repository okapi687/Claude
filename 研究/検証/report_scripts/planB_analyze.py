"""計画Bの集計。引数: planB_sim.py の出力 json、集計結果の出力 json、図の出力 png
密度を2通りの測り方で、窓（区画）の大きさ Ls を変えて測る
  ずらす（今の実装）：Ls×Ls の窓をトーラス上でずらしたときの最大個体数 / N
  区画（論文 p.17 の文面）：X×X の空間を X/Ls × X/Ls の区画に分け、区画の中の最大個体数 / N
群れの大きさの目安として、回転半径 Rg（トーラスを考えた重心からの距離の2乗平均の平方根）も測る
論文の値（目で読んだもの）
  図11 の赤線（密度）：P≈1.5 で最小 0.31〜0.32、P=10〜20 で α=0.3 は 0.6〜0.85、α=0.6 は 0.73〜0.88。P=30 で 0.93〜0.95
  図13・14：10<P<20 で 0.3 前後（分布の中心）
"""
import sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S, N = 50, 100
SLIDE = [5, 8, 10, 12, 15, 20, 25]
GRID = [10, 8, 6, 5, 4, 3, 2]          # 区画の数 k（一辺）。Ls = 50 / k
FIG11 = {0.3: {1.5: 0.31, 4: 0.89, 6: 0.80, 8: 0.62, 10: 0.67, 12: 0.65, 14: 0.61, 15: 0.70, 17: 0.78, 19: 0.86, 21: 0.88, 24: 0.91, 30: 0.93},
         0.6: {1.5: 0.32, 4: 0.83, 6: 0.82, 8: 0.74, 10: 0.75, 12: 0.73, 13: 0.77, 15: 0.82, 18: 0.85, 20: 0.88, 24: 0.91, 30: 0.95}}


def slide(pos, Ls):
    g = np.zeros((S, S), int); g[pos[:, 0], pos[:, 1]] = 1
    c = np.pad(np.tile(g, (2, 2)), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    w = c[Ls:Ls + S, Ls:Ls + S] - c[:S, Ls:Ls + S] - c[Ls:Ls + S, :S] + c[:S, :S]
    return w.max() / N


def grid(pos, k):
    e = np.floor(np.arange(k + 1) * S / k).astype(int)
    ix = np.searchsorted(e, pos[:, 0], side="right") - 1
    iy = np.searchsorted(e, pos[:, 1], side="right") - 1
    h = np.zeros((k, k), int); np.add.at(h, (ix, iy), 1)
    return h.max() / N


def rg(pos):
    th = pos * 2 * np.pi / S
    c = np.angle(np.exp(1j * th).mean(0)) * S / (2 * np.pi)
    d = (pos - c + S / 2) % S - S / 2
    return float(np.sqrt((d ** 2).sum(1).mean()))


raw = json.load(open(sys.argv[1]))
summ = []
for r in raw:
    P = [np.array(p) for p in r["pos"]]
    out = dict(tag=r["tag"], P=r["P"], alpha=r["alpha"], seed=r["seed"], pL=r["pL"], Rg=float(np.mean([rg(p) for p in P])))
    for Ls in SLIDE:
        out[f"slide{Ls}"] = float(np.mean([slide(p, Ls) for p in P]))
    for k in GRID:
        out[f"grid{k}"] = float(np.mean([grid(p, k) for p in P]))
    summ.append(out)
json.dump(summ, open(sys.argv[2], "w"), indent=0)

g = defaultdict(list)
for r in summ:
    g[(r["tag"], r["P"], r["alpha"])].append(r)
Ps = sorted({r["P"] for r in summ}); As = sorted({r["alpha"] for r in summ})
mid = [P for P in Ps if 10 <= P <= 20]


def m(tag, P, a, key):
    return float(np.mean([r[key] for r in g[(tag, P, a)]]))


def mm(tag, a, key):
    return float(np.mean([m(tag, P, a, key) for P in mid]))


keys = [f"slide{L}" for L in SLIDE] + [f"grid{k}" for k in GRID]
names = {**{f"slide{L}": f"ずらす Ls={L}" for L in SLIDE}, **{f"grid{k}": f"区画 Ls={S / k:.1f}（{k}×{k}）" for k in GRID}}
for tag in ["v1", "as2_round"]:
    print(f"\n## {tag}：P=10〜20 の平均（α=0.3 / 0.4 / 0.5 / 0.6）")
    print("| 測り方 | " + " | ".join(f"α={a}" for a in As) + " |")
    print("|---|" + "---|" * len(As))
    for k in keys:
        print(f"| {names[k]} | " + " | ".join(f"{mm(tag, a, k):.2f}" for a in As) + " |")
    print("| 回転半径 Rg | " + " | ".join(f"{mm(tag, a, 'Rg'):.1f}" for a in As) + " |")
    print("| 整列度 pL | " + " | ".join(f"{mm(tag, a, 'pL'):.2f}" for a in As) + " |")

for tag in ["v1", "as2_round"]:
    for a in [0.3, 0.6]:
        print(f"\n## {tag} α={a}：P ごと（ずらす Ls=10 / 区画 5×5 / 区画 2×2 / Rg）")
        print(" ".join(f"P={P}:{m(tag, P, a, 'slide10'):.2f}/{m(tag, P, a, 'grid5'):.2f}/{m(tag, P, a, 'grid2'):.2f}/{m(tag, P, a, 'Rg'):.1f}" for P in Ps))

# ---------------- 図 ----------------
C1, C2, C3, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.family": ["Yu Gothic", "MS Gothic"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
ax = axes[0]
for tag, ls, nm in [("v1", "--", "v1"), ("as2_round", "-", "α2倍・round")]:
    ax.plot([S / k for k in GRID], [np.mean([mm(tag, a, f"grid{k}") for a in As]) for k in GRID], "o" + ls, color=C1, ms=4, label=f"区画（{nm}）")
    ax.plot(SLIDE, [np.mean([mm(tag, a, f"slide{L}") for a in As]) for L in SLIDE], "s" + ls, color=C2, ms=4, label=f"ずらす（{nm}）")
ax.axhspan(0.73, 0.88, color=MUTED, alpha=0.25, lw=0); ax.text(16, 0.75, "図11 の範囲（P=10〜20）", fontsize=8.5, color="#5a5955")
ax.axhspan(0.27, 0.33, color=MUTED, alpha=0.45, lw=0); ax.text(16, 0.345, "図13・14 の中心", fontsize=8.5, color="#5a5955")
ax.set_xlabel("窓・区画の一辺 Ls"); ax.set_ylabel("密度（P=10〜20、α=0.3〜0.6 の平均）"); ax.set_ylim(0, 1.05)
ax.set_title("Ls と測り方で密度がどう変わるか"); ax.legend(fontsize=8, loc="lower right")

for ax, a in [(axes[1], 0.3), (axes[2], 0.6)]:
    for key, col, nm in [("slide10", C2, "ずらす Ls=10（今の実装）"), ("grid2", C1, "区画 Ls=25（2×2）"), ("grid5", C3, "区画 Ls=10（5×5）")]:
        ax.plot(Ps, [m("as2_round", P, a, key) for P in Ps], "o-", color=col, ms=3.5, lw=1.6, label=nm)
    ax.plot(list(FIG11[a]), list(FIG11[a].values()), "D", color=INK, ms=5, label="論文 図11（赤線）")
    ax.set_xlabel("P"); ax.set_ylim(0, 1.05); ax.set_title(f"P に対する密度（α={a}、α2倍・round）")
    ax.legend(fontsize=8, loc="lower right")
fig.savefig(sys.argv[3]); print("\nfig:", sys.argv[3])
