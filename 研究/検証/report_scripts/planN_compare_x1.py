"""計画N の続き（11/25）：論文2 の全図を α 1倍（α そのまま・round・ω 式どおり）と α 2倍（ベスト）で並べて比べる図（fig46）。
入力: report_data/planN/summary.json（α 2倍）と report_data/planN_x1/summary.json（α 1倍）。どちらも planN_paper2_analyze.py の出力。
2つの設定で違うのは α の範囲だけ（ξ∈[−απ, απ] か [−2απ, 2απ]）。エラーバーは 95% 信頼区間。
論文の値は planN_paper2_analyze.py と同じ（目で読んだもの）。図4b の旋回の力の色の幅は ±2 くらい（論文2/α2倍_全図の再現.md）。
使い方: python planN_compare_x1.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
HERE = Path(__file__).resolve().parent
D = HERE.parent / "report_data"
FIG = HERE.parent / "report_fig"
X2 = json.load(open(D / "planN" / "summary.json", encoding="utf-8"))
X1 = json.load(open(D / "planN_x1" / "summary.json", encoding="utf-8"))
FIG3 = {"青": (0.2, 0.65), "赤": (0.93, 4.0), "緑": (0.35, 3.2)}
FIGS2 = {"青": (63, 5, 32), "赤": (99, 1, 0), "緑": (98, 2, 0)}
FIGS2B = (0.88, 0.0, 0.12)
FIGS3 = {0: 0.10, 1: 0.70, 2: 0.93, 3: 0.93, 5: 0.95, 8: 0.55, 10: 0.73}
C1, C2, CP = "tab:purple", "tab:orange", "0.2"
LAB1, LAB2 = "α 1倍（そのまま）", "α 2倍（ベスト）"
names = ["青", "赤", "緑"]

fig = plt.figure(figsize=(17, 13))
gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.35)


def bars3(ax, vals1, vals2, paper, labels, ylabel, title, ylim=None):
    x = np.arange(len(labels))
    ax.bar(x - 0.27, [v[0] for v in vals1], 0.27, yerr=[v[1] for v in vals1], capsize=3, color=C1, label=LAB1)
    ax.bar(x, [v[0] for v in vals2], 0.27, yerr=[v[1] for v in vals2], capsize=3, color=C2, label=LAB2)
    ax.bar(x + 0.27, paper, 0.27, color="white", edgecolor=CP, hatch="//", label="論文")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)


# (a)(b) 図3b,c の値
ax = fig.add_subplot(gs[0, 0])
bars3(ax, [X1["fig3bc"][n]["pol_ci"] for n in names], [X2["fig3bc"][n]["pol_ci"] for n in names],
      [FIG3[n][0] for n in names], names, "整列度", "(a) 図3b：整列度（t=1000〜1500、10シード）", (0, 1.05))
ax.legend(fontsize=8, loc="upper left")
ax = fig.add_subplot(gs[0, 1])
bars3(ax, [X1["fig3bc"][n]["dens_ci"] for n in names], [X2["fig3bc"][n]["dens_ci"] for n in names],
      [FIG3[n][1] for n in names], names, "密度（周り3×3）", "(b) 図3c：密度")

# (c) 図S2：相互予期の割合
ax = fig.add_subplot(gs[0, 2])
bars3(ax, [X1["figS2"][n][0] for n in names], [X2["figS2"][n][0] for n in names],
      [FIGS2[n][0] for n in names], names, "%", "(c) 図S2：相互予期の割合")
ax.set_ylim(40, 102)

# (d) 赤がそろう時刻
ax = fig.add_subplot(gs[0, 3])
for k, (S_, c_, lab) in enumerate(((X1, C1, LAB1), (X2, C2, LAB2))):
    t90 = [v for v in S_["fig3bc"]["赤"]["t90"] if v is not None]
    ax.scatter(np.full(len(t90), k) + np.random.default_rng(k).uniform(-0.1, 0.1, len(t90)), t90, color=c_, s=25)
    ax.plot([k - 0.2, k + 0.2], [np.median(t90)] * 2, color="k", lw=2)
ax.axhline(600, color=CP, ls=":", label="論文 t≈600")
ax.set_xticks([0, 1]); ax.set_xticklabels([LAB1, LAB2]); ax.set_ylabel("time step")
ax.set_title("(d) 図3b：赤がそろう時刻（10シード、黒線＝中央値）\nα 1倍の赤はほとんどそろわないので、参考の値", fontsize=10); ax.legend(fontsize=8)

# (e)(f) 図3a の整列度（P × α）
Ps, As = (10, 20, 30, 40), (0.1, 0.2, 0.3, 0.4, 0.5)
for j, (S_, lab) in enumerate(((X1, LAB1), (X2, LAB2))):
    ax = fig.add_subplot(gs[1, j])
    M = np.array([[S_["fig3a"][f"P{P}_a{a}"][0] for a in As] for P in Ps])
    im = ax.imshow(M, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(len(Ps)):
        for k in range(len(As)):
            ax.text(k, i, f"{M[i, k]:.2f}", ha="center", va="center", fontsize=8, color="w" if M[i, k] < 0.6 else "k")
    ax.set_xticks(range(len(As))); ax.set_xticklabels(As); ax.set_yticks(range(len(Ps))); ax.set_yticklabels(Ps)
    ax.set_xlabel("α"); ax.set_ylabel("P"); ax.set_title(f"({'ef'[j]}) 図3a の整列度：{lab}", fontsize=10)
    for (P, a), col in {(10, 0.1): "tab:blue", (40, 0.2): "tab:red", (40, 0.5): "tab:green"}.items():
        ax.add_patch(plt.Rectangle((As.index(a) - 0.5, Ps.index(P) - 0.5), 1, 1, fill=False, ec=col, lw=2.5))
plt.colorbar(im, ax=ax, fraction=0.04)

# (g) 図S3
ax = fig.add_subplot(gs[1, 2:])
ws = np.arange(11)
for S_, c_, lab, dx in ((X1, C1, LAB1, -0.08), (X2, C2, LAB2, 0.08)):
    v = [S_["figS3"][str(w)]["pol_t1500"] for w in ws]
    ax.errorbar(ws + dx, [p[0] for p in v], yerr=[p[1] for p in v], color=c_, marker="o", capsize=3, label=lab)
ax.plot(list(FIGS3), list(FIGS3.values()), "k*", ms=12, label="論文")
ax.set_xlabel("ω"); ax.set_ylabel("整列度（t=1500）"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title("(g) 図S3：ω を変えたときの整列度（P=40、α=0.2、10シード）", fontsize=10)

# (h) 2体のルールの割合と整列度
ax = fig.add_subplot(gs[2, 0])
bars3(ax, X1["pair"]["rules"], X2["pair"]["rules"], FIGS2B, ["相互予期", "追従", "自由移動"], "割合", "", (0, 1.05))
ax.set_title(f"(h) 図S2b：2体のルールの割合\n整列度 {X1['pair']['pol'][0]:.2f}（1倍）／{X2['pair']['pol'][0]:.2f}（2倍）、論文 約0.9", fontsize=10)

# (i) 図4b,c の力の大きさ
ax = fig.add_subplot(gs[2, 1])
x = np.arange(2)
r1 = [max(abs(v) for v in X1["fig4"][k]) for k in ("turn_range", "speed_range")]
r2 = [max(abs(v) for v in X2["fig4"][k]) for k in ("turn_range", "speed_range")]
ax.bar(x - 0.27, r1, 0.27, color=C1, label=LAB1); ax.bar(x, r2, 0.27, color=C2, label=LAB2)
ax.bar([0.27], [2.0], 0.27, color="white", edgecolor=CP, hatch="//", label="論文（色の幅、目で読んだ）")
ax.set_xticks(x); ax.set_xticklabels(["旋回の力", "加速の力"]); ax.set_ylabel("地図の値の最大の絶対値")
ax.set_title(f"(i) 図4b,c：力の大きさ\n左右の距離で説明できる割合（旋回）{X1['fig4']['turn_share']:.2f}／{X2['fig4']['turn_share']:.2f}", fontsize=10)
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

# (j) 3体
ax = fig.add_subplot(gs[2, 2])
lbl = ["加速：予測との相関", "旋回：予測との相関", "旋回の残差：間", "旋回の残差：同じ側"]
v1 = [X1["trio"]["speed"]["corr_sim_pred"], X1["trio"]["turn"]["corr_sim_pred"], X1["trio"]["turn"]["between"], X1["trio"]["turn"]["same_side"]]
v2 = [X2["trio"]["speed"]["corr_sim_pred"], X2["trio"]["turn"]["corr_sim_pred"], X2["trio"]["turn"]["between"], X2["trio"]["turn"]["same_side"]]
x = np.arange(4)
ax.bar(x - 0.2, v1, 0.4, color=C1, label=LAB1); ax.bar(x + 0.2, v2, 0.4, color=C2, label=LAB2)
ax.set_xticks(x); ax.set_xticklabels(lbl, rotation=20, fontsize=8); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
ax.set_title("(j) 図S4：3体（論文は「間にいるとき残差が強い」）", fontsize=10)

# (k) まとめの表
ax = fig.add_subplot(gs[2, 3]); ax.axis("off")
rows = [("赤 整列度", X1["fig3bc"]["赤"]["pol_ci"][0], X2["fig3bc"]["赤"]["pol_ci"][0], 0.93),
        ("緑 整列度", X1["fig3bc"]["緑"]["pol_ci"][0], X2["fig3bc"]["緑"]["pol_ci"][0], 0.35),
        ("青 整列度", X1["fig3bc"]["青"]["pol_ci"][0], X2["fig3bc"]["青"]["pol_ci"][0], 0.2),
        ("青 相互予期 %", X1["figS2"]["青"][0][0], X2["figS2"]["青"][0][0], 63),
        ("図S3 ω=0", X1["figS3"]["0"]["pol_t1500"][0], X2["figS3"]["0"]["pol_t1500"][0], 0.10),
        ("図S3 ω=10", X1["figS3"]["10"]["pol_t1500"][0], X2["figS3"]["10"]["pol_t1500"][0], 0.73),
        ("2体 整列度", X1["pair"]["pol"][0], X2["pair"]["pol"][0], 0.9),
        ("旋回の力の幅", r1[0], r2[0], 2.0)]
tb = ax.table(cellText=[[n, f"{a:.2f}", f"{b:.2f}", f"{p:.2f}"] for n, a, b, p in rows],
              colLabels=["", "1倍", "2倍", "論文"], loc="upper center", cellLoc="center", colWidths=[0.42, 0.19, 0.19, 0.19])
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.5)
ax.set_title("(k) 主な値（平均）", fontsize=10)

fig.suptitle("fig46　論文2 の全図：α 1倍（α そのまま・round・ω 式どおり）と α 2倍（ベスト）の比較（違うのは α の範囲だけ、エラーバーは 95% 信頼区間）", fontsize=12)
fig.savefig(FIG / "fig46_planN_x1_vs_x2.png", dpi=110, bbox_inches="tight"); plt.close(fig)
print("saved fig46")
