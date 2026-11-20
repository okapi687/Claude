"""ゼミの指摘への対応の図（論文2）。引数: 出力フォルダ"""
import sys, json, os, collections
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
INK, MUTED = "#0b0b0b", "#8a8984"
RED, GREEN, BLUE = "#d62f2f", "#2ca02c", "#1f5fd6"

res = json.load(open(f"{SP}/res_seminar_p2.json"))
g = collections.defaultdict(list)
for r in res:
    g[(r["kind"], r["tag"], r.get("P"), r.get("alpha"), r["scale"])].append(r)
mean = lambda rs, k: float(np.mean([r[k] for r in rs]))
A = [round(0.05 * i, 2) for i in range(1, 21)]

# ---------- 図14：P=40 で α を振る ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
for tag, col, name in [("principal", C1, "ω 式どおり（Murakami）"), ("velocity", C2, "ω を移動側に（v1 既定）")]:
    rs = [g[("sweep", tag, 40, a, 1)] for a in A]
    y = [mean(r, "pol") for r in rs]
    lo = [min(x["pol"] for x in r) for r in rs]; hi = [max(x["pol"] for x in r) for r in rs]
    axes[0].fill_between(A, lo, hi, color=col, alpha=0.15, lw=0)
    axes[0].plot(A, y, "o-", color=col, ms=4, label=name)
    axes[1].plot(A, [mean(r, "dens") for r in rs], "o-", color=col, ms=4, label=name)
for ax, key in [(axes[0], "pol"), (axes[1], "dens")]:
    pv = {"pol": (0.93, 0.35), "dens": (4.0, 3.2)}[key]
    ax.plot([0.2, 0.5], pv, "D", color=INK, ms=7, label="論文（赤・緑、α そのまま）")
    ax.plot([0.4, 1.0], pv, "D", mfc="white", mec=INK, ms=7, mew=1.5, label="論文（赤・緑、α を2倍に読んだ位置）")
    ax.set_xlabel("シミュレーションの α（ξ∈[−απ, απ]）")
axes[0].set_title("整列度（P=40、t=1000〜1500、3シード）"); axes[0].set_ylim(0, 1)
axes[1].set_title("密度（3×3 の近傍の個体数）"); axes[1].set_ylim(0, 5)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.36, -0.1), fontsize=8.5)
ax = axes[2]
rs = [g[("sweep", "principal", 40, a, 1)] for a in A]
bottom = np.zeros(len(A))
for k, col, name in [("ma", INK, "相互予期"), ("fol", C3, "追従"), ("free", C2, "自由移動"), ("stay", RED, "留まる")]:
    v = np.array([mean(r, k) for r in rs])
    ax.bar(A, v, 0.04, bottom=bottom, color=col, label=name); bottom += v
ax.set_ylim(0.75, 1.0); ax.set_title("ルールの割合（ω 式どおり）"); ax.set_xlabel("シミュレーションの α")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4); ax.grid(axis="x", visible=False)
fig.suptitle("論文2：P=40 で α を 0.05〜1.0 まで振ったとき", fontsize=11, y=1.02)
fig.savefig(f"{OUT}/fig14_paper2_alpha_sweep.png"); plt.close(fig)

# ---------- 図15：図3a と同じ格子のスナップショット ----------
Ps, As = [10, 20, 30, 40], [0.1, 0.2, 0.3, 0.4, 0.5]
fig, axes = plt.subplots(4, 11, figsize=(13, 5.3), gridspec_kw={"width_ratios": [1] * 5 + [0.25] + [1] * 5})
frame = {(10, 0.1): BLUE, (40, 0.2): RED, (40, 0.5): GREEN}
for i, P in enumerate(Ps):
    axes[i, 5].axis("off")
    for blk, sc in enumerate([1, 2]):
        for j, a in enumerate(As):
            ax = axes[i, j + blk * 6]
            key = ("sweep", "principal", 40, round(a * sc, 2), 1) if P == 40 else ("grid", "principal", P, a, sc)
            r0 = [r for r in g[key] if r["seed"] == 0][0]
            h = np.array(r0["hist"])
            for k in range(h.shape[1]):
                tr = h[:, k, :]
                jump = np.any(np.abs(np.diff(tr, axis=0)) > 25, axis=1)
                for sgm in np.split(tr, np.flatnonzero(jump) + 1):
                    ax.plot(sgm[:, 0], sgm[:, 1], color=MUTED, lw=0.6)
            ax.plot(h[-1, :, 0], h[-1, :, 1], "s", color=INK, ms=1.6)
            ax.set_xlim(0, 50); ax.set_ylim(0, 50); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            ax.grid(False)
            col = frame.get((P, a), "#c3c2bd")
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(col); sp.set_linewidth(2 if (P, a) in frame else 0.6)
            ax.text(25, -6, f"{mean(g[key], 'pol'):.2f}", ha="center", va="top", fontsize=7.5, color=MUTED)
            if i == 0:
                ax.set_title(f"α={a}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"P={P}", fontsize=9)
fig.text(0.31, 0.975, "α をそのまま", ha="center", fontsize=11)
fig.text(0.735, 0.975, "α を2倍に読む", ha="center", fontsize=11)
fig.suptitle("論文2 図3a と同じ格子（ω 式どおり、round、seed 0 の t=1500、3ステップの軌跡。下の数字は整列度の平均）", fontsize=11, y=1.04)
fig.savefig(f"{OUT}/fig15_paper2_fig3a_grid.png"); plt.close(fig)

# ---------- 図16：2体（図4・図S2b）と図S3 ----------
pairs = [("principal_x1_dedupe", "ω式どおり\nα そのまま"), ("principal_x2_dedupe", "ω式どおり\nα 2倍"),
         ("velocity_x1_dedupe", "ω移動側\nα そのまま"), ("velocity_x2_dedupe", "ω移動側\nα 2倍"),
         ("principal_x1_count", "ω式どおり\n本数で数える")]
P2 = {t: [r for r in res if r["kind"] == "pair" and r["tag"] == t] for t, _ in pairs}
fig = plt.figure(figsize=(14, 7.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.75, wspace=0.28)
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(pairs) + 1)
ax.bar(x[:-1], [mean(P2[t], "pol") for t, _ in pairs], 0.6, color=C1)
ax.bar(x[-1], 0.9, 0.6, color=INK)
ax.set_xticks(x); ax.set_xticklabels([n for _, n in pairs] + ["論文"], fontsize=7.5); ax.set_ylim(0, 1)
ax.set_title("2体の整列度（P=40, α=0.2）"); ax.grid(axis="x", visible=False)
ax = fig.add_subplot(gs[0, 1])
bottom = np.zeros(len(pairs) + 1)
for k, col, name, pv in [("ma", INK, "相互予期", 0.88), ("fol", C3, "追従", 0.0), ("free", C2, "自由移動", 0.12)]:
    v = np.array([mean(P2[t], k) for t, _ in pairs] + [pv])
    ax.bar(x, v, 0.6, bottom=bottom, color=col, label=name); bottom += v
ax.set_xticks(x); ax.set_xticklabels([n for _, n in pairs] + ["論文\n図S2b"], fontsize=7.5); ax.set_ylim(0, 1)
ax.set_title("2体のルールの割合（図S2b）"); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3); ax.grid(axis="x", visible=False)
ax = fig.add_subplot(gs[0, 2])
s3 = collections.defaultdict(list)
for r in json.load(open(f"{SP}/res_seminar_s3.json")):
    s3[(r["on"], r["scale"], r["omega"])].append(r["pol"])
for on, sc, col, ls, name in [("principal", 1, C1, "--", "ω式どおり・α そのまま"), ("principal", 2, C1, "-", "ω式どおり・α 2倍"),
                              ("velocity", 1, C2, "--", "ω移動側・α そのまま"), ("velocity", 2, C2, "-", "ω移動側・α 2倍")]:
    ax.plot(range(11), [np.mean(s3[(on, sc, w)]) for w in range(11)], ls, color=col, marker="o", ms=3.5, lw=1.6, label=name)
ax.axvspan(1.5, 7.5, color="#e4e3df", alpha=0.6, lw=0)
ax.text(4.5, 0.03, "論文で整列度が高い範囲", ha="center", fontsize=8, color=MUTED)
ax.set_xlabel("ω"); ax.set_ylim(0, 1); ax.set_title("図S3：ω を変えたときの整列度（t=1400〜1500）")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
bins = np.arange(-6, 7)
hb = np.linspace(-180, 180, 9)
for col_i, (t, name) in enumerate(pairs[:2]):
    ax = fig.add_subplot(gs[1, col_i])
    M = np.nanmean(np.array([[[np.nan if v is None else v for v in row] for row in r["tf_map"]] for r in P2[t]], float), 0)
    im = ax.imshow(M, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto",
                   extent=[bins[0] - 0.5, bins[-1] + 0.5, hb[0], hb[-1]])
    ax.set_xlabel("左右の距離（＋が左）"); ax.set_ylabel("相手との向きの差（度）"); ax.grid(False)
    ax.set_title("旋回の力（色、＋が左）\n" + name.replace(chr(10), "・"), fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.05)
ax = fig.add_subplot(gs[1, 2])
for t, name in pairs[:2]:
    sf = np.nanmean(np.array([[np.nan if v is None else v for v in r["sf_fb"]] for r in P2[t]], float), 0)
    ax.plot(bins, sf, "o-", ms=4, color=C1 if "x1" in t else C2, label=name.replace("\n", "・"))
ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlabel("前後の距離（＋が前）"); ax.set_ylabel("加速の力（＋が前向き）"); ax.set_title("加速の力（向きの差で平均）"); ax.legend(fontsize=8)
fig.suptitle("論文2：2体のシミュレーション（図4・図S2b、30000 ステップ、3シード）と図S3", fontsize=11, y=0.98)
fig.savefig(f"{OUT}/fig16_paper2_pair_S3.png"); plt.close(fig)
print("ok")
