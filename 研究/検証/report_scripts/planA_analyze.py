"""計画Aの集計。引数: planA_sweep.py の出力 json、図の出力 png
- 計測区間 t=300〜900（計画2と同じ）と t=2000〜3000 の平均を比べる
- 50 ステップごとの平均の分布を見て、「そろった状態」と「ばらばらの状態」の2つの山に分かれていないかを見る
論文の値（11/21 に図を読み直した）
  図11（P=10〜20 の範囲、青線）：α=0.3 → 0.53、0.4 → 0.30、0.5 → 0.13、0.6 → 0.10
  図14（10<P<20、α の幅ごとの分布の中心）：0.2〜0.3 → 0.53、0.3〜0.4 → 0.35、0.4〜0.5 → 0.20、0.5〜0.6 → 0.08
"""
import sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG11 = {0.3: 0.53, 0.4: 0.30, 0.5: 0.13}
FIG14 = [((0.2, 0.3), 0.53), ((0.3, 0.4), 0.35), ((0.4, 0.5), 0.20)]

res = json.load(open(sys.argv[1]))
alphas = sorted({r["alpha"] for r in res})
Ps = sorted({r["P"] for r in res})
g = defaultdict(list)
for r in res:
    g[(r["P"], r["alpha"])].append(r)
    g[("all", r["alpha"])].append(r)


def win(r, key, a, b):
    return float(np.mean(r[key][a:b]))


def stat(P, a, key, s, e):
    v = [win(r, key, s, e) for r in g[(P, a)]]
    return float(np.mean(v)), float(np.std(v))


def blocks(r, key="pL", start=300, n=50):
    x = np.array(r[key][start:])
    return x[: len(x) // n * n].reshape(-1, n).mean(1)


def bimodality(x):
    """Sarle の二峰性係数。0.555 を超えると二峰の疑い"""
    x = np.asarray(x); n = len(x)
    m = x.mean(); s = x.std()
    g1 = np.mean((x - m) ** 3) / s ** 3
    g2 = np.mean((x - m) ** 4) / s ** 4 - 3
    return (g1 ** 2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


print("## 整列度 pL（P=10/15/20 と5シードの平均）")
print("| 論文の α | t=300〜900 | t=2000〜3000 | 差 | 50ステップ平均の標準偏差 | 二峰性係数 | θ のそろい φ | 密度 | 相互予期 / 追従 / 自由 |")
print("|---|---|---|---|---|---|---|---|---|")
rows = {}
for a in alphas:
    rs = g[("all", a)]
    e = np.mean([win(r, "pL", 300, 900) for r in rs])
    l = np.mean([win(r, "pL", 2000, 3000) for r in rs])
    bl = np.concatenate([blocks(r) for r in rs])
    bsd = float(np.mean([blocks(r).std() for r in rs]))
    bc = bimodality(bl)
    phi = np.mean([win(r, "phi", 300, 3000) for r in rs])
    dens = np.mean([np.mean(r["dens"][30:]) for r in rs])
    ru = np.mean([r["rules"] for r in rs], axis=0)
    rows[a] = (e, l)
    print(f"| {a:.3f} | {e:.3f} | {l:.3f} | {l - e:+.3f} | {bsd:.3f} | {bc:.2f} | {phi:.3f} | {dens:.2f} | "
          f"{ru[0]*100:.0f} / {ru[1]*100:.0f} / {ru[2]*100:.0f} |")

print("\n## P ごと（t=300〜900 / t=2000〜3000）")
print("| 論文の α | " + " | ".join(f"P={P}" for P in Ps) + " |")
print("|---|" + "---|" * len(Ps))
for a in alphas:
    cells = []
    for P in Ps:
        e, _ = stat(P, a, "pL", 300, 900)
        l, sd = stat(P, a, "pL", 2000, 3000)
        cells.append(f"{e:.2f} / {l:.2f}（±{sd:.2f}）")
    print(f"| {a:.3f} | " + " | ".join(cells) + " |")

print("\n## 傾き（α が 0.025 増えるごとの整列度の変化、t=2000〜3000）")
for a0, a1 in zip(alphas[:-1], alphas[1:]):
    print(f"{a0:.3f}→{a1:.3f}: {rows[a1][1] - rows[a0][1]:+.3f}")

# ---------------- 図 ----------------
C1, C2, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.family": ["Yu Gothic", "MS Gothic"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.1, 1, 1.2]})
ax = axes[0]
ax.plot(alphas, [rows[a][0] for a in alphas], "o--", color=C1, ms=4, lw=1.6, label="モデル t=300〜900")
ax.plot(alphas, [rows[a][1] for a in alphas], "o-", color=C2, ms=4, lw=2, label="モデル t=2000〜3000")
ax.plot(list(FIG11), list(FIG11.values()), "D", color=INK, ms=7, label="論文 図11（P=10〜20）")
for i, ((lo, hi), v) in enumerate(FIG14):
    ax.plot([lo, hi], [v, v], "-", color=MUTED, lw=4, alpha=0.7, label="論文 図14（α の幅ごと）" if i == 0 else None)
ax.set_xlabel("論文の α（モデルは α 2倍・round・sequential）"); ax.set_ylabel("整列度 pL")
ax.set_ylim(0, 0.7); ax.set_xlim(0.19, 0.51); ax.set_title("α に対する整列度（平均）")
ax.legend(fontsize=8, loc="lower left")

ax = axes[1]
bins = np.linspace(0, 0.8, 41)
cmap = plt.get_cmap("viridis")
for i, a in enumerate(alphas[::2]):
    bl = np.concatenate([blocks(r) for r in g[("all", a)]])
    h, _ = np.histogram(bl, bins=bins, density=True)
    ax.plot((bins[1:] + bins[:-1]) / 2, h, color=cmap(i / 4), lw=1.8, label=f"α={a:.2f}")
ax.set_xlabel("整列度 pL（50 ステップごとの平均）"); ax.set_ylabel("頻度（密度）")
ax.set_title("整列度の分布（t≥300）"); ax.legend(fontsize=8)

ax = axes[2]
for a, col in [(0.3, C1), (0.35, "#1baf7a"), (0.4, C2)]:
    for r in g[(15, a)][:2]:
        x = np.array(r["pL"]).reshape(-1, 50).mean(1)
        ax.plot(np.arange(len(x)) * 50 + 25, x, color=col, lw=1.2, alpha=0.9,
                label=f"α={a}" if r["seed"] == 0 else None)
ax.set_xlabel("ステップ"); ax.set_ylabel("整列度 pL（50 ステップ平均）"); ax.set_ylim(0, 0.8)
ax.set_title("時系列（P=15、シード 0・1）"); ax.legend(fontsize=8, ncol=3, loc="upper right")
fig.savefig(sys.argv[2]); print("\nfig:", sys.argv[2])
