"""α の効き方の調査の図。引数: 出力フォルダ"""
import sys, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
plt.rcParams.update({"font.family": ["Yu Gothic", "MS Gothic"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "axes.titlesize": 11, "legend.frameon": False,
                     "savefig.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 2})
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK2 = "#52514e"


def load(fn):
    g = defaultdict(lambda: defaultdict(list))
    for r in json.load(open(f"{SP}/{fn}")):
        g[r["tag"]][r["alpha"]].append(r)
    return g


def curve(d, key):
    a = sorted(d)
    return np.array(a), np.array([np.mean([r[key] for r in d[x]]) for x in a])


r1 = load("res_alpha.json"); r2 = load("res_alpha2.json")

# 選び方がランダムだった場合の整列度（基本ベクトルが全員同じ向きのとき）
rng = np.random.default_rng(0)
A = np.linspace(0.2, 0.7, 11)
rand_pol, rand_cos = [], []
for a in A:
    # 全員共通の向き θ を 0〜2π でいろいろ変えて平均（floor の偏りが θ によって効き方が変わるため）
    pp, cc = [], []
    for th in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        n = 20000
        eta = rng.uniform(0, 1, n); xi = rng.uniform(-a * np.pi, a * np.pi, n)
        vx, vy = np.floor(4 * eta * np.cos(th + xi)), np.floor(4 * eta * np.sin(th + xi))
        pp.append(np.hypot(vx.mean(), vy.mean()) / 4)
        nz = (vx != 0) | (vy != 0)
        cc.append(np.mean(np.cos(np.arctan2(vy[nz], vx[nz]) - th)))
    rand_pol.append(np.mean(pp)); rand_cos.append(np.mean(cc))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
a, pol = curve(r2["p1_base"], "pol"); _, phi = curve(r2["p1_base"], "phi"); _, cs = curve(r2["p1_base"], "fwd")
ax = axes[0]
ax.plot(a, pol, "o-", color=C1, ms=6, label="モデル（v1, P=15, 10シード）")
ax.plot(A, rand_pol, color=C2, label="遷移先をランダムに選んだ場合")
ax.plot([0.25, 0.3, 0.4, 0.5, 0.6], [0.55, 0.5, 0.27, 0.12, 0.10], "D", color="#0b0b0b", ms=7, label="論文（図11・13・14）")
ax.set_title("整列度"); ax.set_xlabel("α"); ax.set_ylim(0, 1); ax.legend(fontsize=8.5, loc="upper right")
ax = axes[1]
ax.plot(a, phi, "o-", color=C1, ms=6)
ax.set_title("基本ベクトルの向きのそろい"); ax.set_xlabel("α"); ax.set_ylim(0, 1.05)
ax.annotate("どの α でも 1.00\n（全員が同じ向き）", (0.45, 1.0), (0.32, 0.6), arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax = axes[2]
ax.plot(a, cs, "o-", color=C1, ms=6, label="実際の移動（相互予期で選ばれたもの）")
ax.plot(A, rand_cos, color=C2, label="可能的遷移の平均")
ax.set_title("移動の向きと基本ベクトルの cos の平均"); ax.set_xlabel("α"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8.5, loc="lower left")
fig.suptitle("論文1：α を変えたときの整列度の内訳（t=300〜900 の平均）", fontsize=11, y=1.04)
fig.savefig(f"{OUT}/fig09_alpha_paper1.png"); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
for tag, name, col in [("p2_base_principal", "式どおりのω", C2), ("p2_base_velocity", "ωを移動側に（v1）", C1),
                       ("p2_tiebreak_random", "v1 + 同率はランダム", C3)]:
    for ax, key in zip(axes, ["pol", "phi", "fwd"]):
        a, y = curve(r1[tag], key)
        ax.plot(a, y, "o-", color=col, ms=6, label=name)
axes[0].plot([0.2, 0.5], [0.93, 0.35], "D", color="#0b0b0b", ms=7, label="論文（図3 の赤・緑）")
axes[0].set_title("整列度"); axes[1].set_title("基本ベクトルの向きのそろい"); axes[2].set_title("移動の向きと基本ベクトルの cos の平均")
for ax in axes:
    ax.set_xlabel("α"); ax.set_ylim(0, 1.05); ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
fig.suptitle("論文2：α を変えたときの整列度の内訳（P=40、t=1000〜1500 の平均、3シード）", fontsize=11, y=1.04)
fig.savefig(f"{OUT}/fig10_alpha_paper2.png"); plt.close(fig)
print("rand_pol", dict(zip(np.round(A, 2).tolist(), np.round(rand_pol, 2).tolist())))
print("rand_cos", dict(zip(np.round(A, 2).tolist(), np.round(rand_cos, 2).tolist())))
