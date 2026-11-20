"""検証レポート用の図を作る。引数: 出力フォルダ"""
import sys, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, r"C:\Users\PC_User\ClaudeCode\研究")
from model.MAM_basic_v1 import SoldierCrabSwarm_v1

plt.rcParams.update({"font.family": ["Yu Gothic", "MS Gothic"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "axes.titlesize": 11, "legend.frameon": False,
                     "savefig.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 2})
C_RED = "#e34948"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # 比べる版の色（固定順）
INK2 = "#52514e"
PAPER = dict(color="#0b0b0b", ls="--", lw=1.2)


def ma(x, w=25):
    x = np.asarray(x, float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def group(res):
    g = defaultdict(list)
    for r in res:
        g[r["tag"]].append(r)
    return g


def clean(v):
    return [np.nan if x is None else x for x in v]


rep = group(json.load(open(f"{SP}/res_report.json")))
p2old = group(json.load(open(f"{SP}/res_p2.json")))
p1 = json.load(open(f"{SP}/res_p1.json"))
tab = {}

# ---------------------------------------------------------------- 図1 Int の偏り
rng = np.random.default_rng(0)
L, n = 6, 200000
th = rng.uniform(0, 2 * np.pi, n); eta = rng.uniform(0, 1, n)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
for ax, (name, f) in zip(axes, [("floor（切り捨て）", np.floor), ("round（四捨五入）", np.round)]):
    vx = f(L * eta * np.cos(th)).astype(int); vy = f(L * eta * np.sin(th)).astype(int)
    H, _, _ = np.histogram2d(vx, vy, bins=np.arange(-7.5, 7.5))
    ax.imshow(H.T / n * 100, origin="lower", extent=[-7.5, 6.5, -7.5, 6.5], cmap="Blues")
    ax.plot(vx.mean(), vy.mean(), "o", ms=9, mfc=C_RED, mec="white", mew=2)
    ax.annotate(f"平均 ({vx.mean():+.2f}, {vy.mean():+.2f})", (vx.mean(), vy.mean()), (0.5, 4.8),
                color="#0b0b0b", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6)
    ax.set_title(name); ax.set_xlabel("vx"); ax.set_ylabel("vy"); ax.grid(False)
    tab[f"int|{name}"] = (round(float(vx.mean()), 3), round(float(vy.mean()), 3), round(float(np.mean((vx == 0) & (vy == 0))), 3))
fig.suptitle("可能的遷移ベクトル Int(L·η·(cos, sin)) の分布（L=6、向きはランダム、色は出る割合）", fontsize=10)
fig.savefig(f"{OUT}/fig01_int_bias.png"); plt.close(fig)

# ---------------------------------------------------------------- 図2 論文2 図3b,c の時系列
conds = [("blue", "青 P=10, α=0.1", (0.2, 0.65)), ("red", "赤 P=40, α=0.2", (0.93, 4.0)),
         ("green", "緑 P=40, α=0.5", (0.35, 3.2))]
variants = [("floor・式どおりのω", lambda c: p2old[f"fig3_{c}_floor"], C3),
            ("round・式どおりのω", lambda c: rep[f"p2_{c}_principal"], C2),
            ("round・ωを移動側に（v1）", lambda c: rep[f"p2_{c}_velocity"], C1)]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
for j, (c, title, (pp, pd)) in enumerate(conds):
    for name, get, col in variants:
        rs = get(c)
        pol = np.nanmean([clean(r["pol"])[-1500:] for r in rs], axis=0)
        den = np.mean([r["dens"][-1500:] for r in rs], axis=0)
        t = np.arange(len(ma(pol))) + 13
        axes[0, j].plot(t, ma(pol), color=col, label=name)
        axes[1, j].plot(t, ma(den), color=col, label=name)
        rul = (np.mean([r["rules_late"] for r in rs], axis=0) if "rules_late" in rs[0]
               else np.mean([np.mean(np.array(r["rules"])[1000:], 0) for r in rs], axis=0)) * 100
        tab[f"fig3|{c}|{name}"] = (round(float(np.nanmean(pol[-500:])), 2), round(float(np.mean(den[-500:])), 2),
                                   np.round(rul, 1).tolist())
    axes[0, j].axhline(pp, **PAPER, label="論文（図から読んだ値）")
    axes[1, j].axhline(pd, **PAPER, label="論文（図から読んだ値）")
    axes[0, j].set_title(title); axes[0, j].set_ylim(0, 1.05); axes[1, j].set_ylim(0, 5.5)
    axes[1, j].set_xlabel("ステップ")
axes[0, 0].set_ylabel("整列度（polarity）"); axes[1, 0].set_ylabel("密度（3×3 の近傍数）")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("論文2 図3b・c の再現（6シードの平均、25ステップ移動平均）", fontsize=11)
fig.savefig(f"{OUT}/fig02_paper2_fig3_timeseries.png"); plt.close(fig)

# ---------------------------------------------------------------- 図3 論文2 図S3（ω）
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
w = np.arange(11)
for om, name, col in [("principal", "式どおり q = V_u + ω·v_u", C2), ("velocity", "入れ替え q = v_u + ω·V_u（v1）", C1)]:
    pol = [np.mean([np.nanmean(r["pol"][-50:]) for r in rep[f"p2S3_{om}_{k}"]]) for k in w]
    den = [np.mean([np.mean(r["dens"][-50:]) for r in rep[f"p2S3_{om}_{k}"]]) for k in w]
    axes[0].plot(w, pol, "o-", color=col, ms=6, label=name)
    axes[1].plot(w, den, "o-", color=col, ms=6, label=name)
    tab[f"S3|{om}"] = (np.round(pol, 2).tolist(), np.round(den, 2).tolist())
axes[0].plot([0], [0.1], "s", color="#0b0b0b", ms=7, label="論文 ω=0")
axes[0].plot([2, 7], [0.93, 0.93], **PAPER, label="論文 ω≈2〜7 の高い整列度")
axes[1].plot([0], [2.0], "s", color="#0b0b0b", ms=7, label="論文 ω=0")
axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("整列度（t=1450〜1500）"); axes[1].set_ylabel("密度（t=1450〜1500）")
for ax in axes:
    ax.set_xlabel("ω"); ax.set_xticks(w)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("論文2 図S3 の再現：ω を変えたとき（P=40, α=0.2, L=6、round、5シード）", fontsize=11)
fig.savefig(f"{OUT}/fig03_paper2_figS3_omega.png"); plt.close(fig)

# ---------------------------------------------------------------- 図4 論文2 図3a の格子（整列度）
Ps, As = [10, 20, 30, 40], [0.1, 0.2, 0.3, 0.4, 0.5]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, (name, get) in zip(axes, [("式どおりのω（Murakami）", lambda P, a: p2old[f"3a_P{P}_a{a}"]),
                                  ("ωを移動側に（v1）", lambda P, a: rep[f"p2grid_{P}_{a}"])]):
    M = np.array([[np.mean([np.nanmean(clean(r["pol"])[-500:]) for r in get(P, a)]) for a in As] for P in Ps])
    im = ax.imshow(M, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(Ps)):
        for j in range(len(As)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(5), As); ax.set_yticks(range(4), Ps); ax.set_xlabel("α"); ax.set_ylabel("P")
    ax.set_title(name); ax.grid(False)
    tab[f"grid|{name}"] = np.round(M, 2).tolist()
fig.colorbar(im, ax=axes, shrink=0.85, label="整列度（t=1000〜1500）")
fig.suptitle("論文2 図3a：P と α を変えたときの整列度（round、3シード）", fontsize=11, y=1.04)
fig.savefig(f"{OUT}/fig04_paper2_fig3a_grid.png"); plt.close(fig)

# ---------------------------------------------------------------- 図5 論文1 追従の連鎖
Pl = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ch, name, col in [(True, "連鎖あり（MAM_basic_ex）", C2), (False, "連鎖なし（ex_fix / ex_v2 / v1）", C1)]:
    den = [np.mean([np.mean(r["dens"][300:]) for r in rep[f"p1chain_{ch}_{P}"]]) for P in Pl]
    fr = [np.mean([r["rules_late"][2] for r in rep[f"p1chain_{ch}_{P}"]]) * 100 for P in Pl]
    axes[0].plot(Pl, den, "o-", color=col, ms=6, label=name)
    axes[1].plot(Pl, fr, "o-", color=col, ms=6, label=name)
    tab[f"chain|{ch}"] = (np.round(den, 2).tolist(), np.round(fr, 0).tolist())
axes[0].set_ylabel("密度（Ls=10 の窓の最大個体数 / N）"); axes[1].set_ylabel("自由移動の割合 [%]")
for ax in axes:
    ax.set_xlabel("P"); ax.set_xticks(Pl)
axes[0].set_ylim(0, 1)
axes[0].annotate("P=2 で群れが崩れる", (2, tab["chain|False"][0][1]), (5, 0.08),
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
axes[0].legend(loc="upper left")
fig.suptitle("論文1 図8：追従の連鎖をなくす修正（11/15）の効果（α=0.7、t=300〜600 の平均、3シード）", fontsize=11)
fig.savefig(f"{OUT}/fig05_paper1_follow_chain.png"); plt.close(fig)

# ---------------------------------------------------------------- 図6 論文1 図10・11
g1 = defaultdict(list)
for r in p1:
    g1[(r["int_mode"], r["alpha"], r["P"])].append(r)
alphas = [0.7, 0.6, 0.5, 0.4, 0.3]
seq = ["#b7d3f2", "#7fb0e8", "#4a8edd", "#2a78d6", "#14498a"]   # α が小さいほど濃い（青の単色）
paper_pol = {0.6: 0.10, 0.5: 0.12, 0.4: 0.27, 0.3: 0.50}
Pall = list(range(1, 31))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for a, col in zip(alphas, seq):
    pol = [np.mean([r["pol"] for r in g1[("floor", a, P)]]) for P in Pall]
    den = [np.mean([r["dens"]["10"] for r in g1[("floor", a, P)]]) for P in Pall]
    axes[0].plot(Pall, pol, color=col, label=f"α={a}", lw=1.8)
    axes[1].plot(Pall, den, color=col, label=f"α={a}", lw=1.8)
    if a in paper_pol:
        axes[0].plot([22, 30], [paper_pol[a]] * 2, color=col, ls="--", lw=1.4)
axes[0].text(26, 0.0, "破線：論文の値（P が大きいところ）", fontsize=8, color=INK2, ha="center", va="bottom")
axes[0].set_ylabel("整列度（|平均変位| / L）"); axes[1].set_ylabel("密度（Ls=10）")
axes[1].axvspan(8, 12, color="#e4e3df", alpha=0.7, lw=0)
axes[1].text(10, 0.05, "論文では密度が\nいったん下がる帯", ha="center", fontsize=8, color=INK2)
for ax in axes:
    ax.set_xlabel("P"); ax.set_ylim(0, 1.0)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.07))
fig.suptitle("論文1 図10・11：P を変えたときの整列度と密度（ex_v2 = v1、floor、3シード）", fontsize=11)
fig.savefig(f"{OUT}/fig06_paper1_fig10_11.png"); plt.close(fig)

# ---------------------------------------------------------------- 図7 論文1 図13・14
bands = [("0.2〜0.3", [0.2, 0.25, 0.3], 0.55), ("0.3〜0.4", [0.3, 0.35, 0.4], 0.35),
         ("0.4〜0.5", [0.4, 0.45, 0.5], 0.20), ("0.5〜0.6", [0.5, 0.55, 0.6], 0.10)]
fig, ax = plt.subplots(figsize=(8, 4))
for im, col, dx in [("floor", C1, -0.18), ("round", C2, 0.18)]:
    data = [[x for a in al for P in [10, 12, 14, 16, 18, 20] for r in g1.get((im, a, P), []) for x in r["pol_series"]]
            for _, al, _ in bands]
    bp = ax.boxplot(data, positions=np.arange(4) + dx, widths=0.3, showfliers=False, patch_artist=True,
                    medianprops=dict(color="white", lw=2), whiskerprops=dict(color=col), capprops=dict(color=col))
    for b in bp["boxes"]:
        b.set(facecolor=col, edgecolor=col)
    ax.plot([], [], "s", color=col, ms=9, label=f"モデル（{im}）")
ax.plot(np.arange(4), [b[2] for b in bands], "D", color="#0b0b0b", ms=8, label="論文の中央値（図から読んだ値）")
ax.set_xticks(range(4), [f"α={b[0]}" for b in bands]); ax.set_ylabel("整列度（各ステップの値）"); ax.set_ylim(0, 1)
ax.legend(loc="upper right")
ax.set_title("論文1 図13・14：α の帯ごとの整列度（P=10〜20、箱は 25〜75%）")
fig.savefig(f"{OUT}/fig07_paper1_fig13_14.png"); plt.close(fig)


# ---------------------------------------------------------------- 図8 スナップショット
def snap(ax, kw, T, title):
    m = SoldierCrabSwarm_v1(**kw)
    hist = m.run(T, record_history=True)
    S = m.S
    for k in range(m.N):                                   # 直近3ステップの軌跡（トーラスをまたぐ所は切る）
        seg = hist[-4:, k].astype(float)
        for a, b in zip(seg[:-1], seg[1:]):
            if np.all(np.abs(b - a) < S / 2):
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#9ab9e0", lw=1)
    ax.scatter(hist[-1, :, 0], hist[-1, :, 1], s=9, marker="s", color=C1, zorder=3)
    ax.set_xlim(-1, S); ax.set_ylim(-1, S); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9.5); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)


fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
snap(axes[0], dict(model="paper1", P=1, alpha=0.7, seed=1), 600, "論文1 P=1, α=0.7（t=600）")
snap(axes[1], dict(model="paper1", P=2, alpha=0.7, seed=1), 600, "論文1 P=2, α=0.7（t=600）")
snap(axes[2], dict(model="paper1", P=10, alpha=0.7, seed=1), 600, "論文1 P=10, α=0.7（t=600）")
snap(axes[3], dict(model="paper2", P=40, alpha=0.2, seed=1), 1500, "論文2 赤 P=40, α=0.2（t=1500）")
fig.suptitle("v1 のスナップショット（四角が個体、線は直近3ステップの軌跡）", fontsize=11)
fig.savefig(f"{OUT}/fig08_snapshots.png"); plt.close(fig)

json.dump(tab, open(f"{SP}/report_table.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("figs done")
