"""計画R の図。planR_flowchart.py（本番と long）の結果から作る。
fig47：赤（α 2倍）の 3000 ステップ後の位置。今のモデル / 流れ図どおり（全部）× 10シード。矢印は基本ベクトルの向き
fig48：整列度の時系列（α 2倍、赤・緑。シードごとの細い線と平均）と、読み方ごとの整列度（95% 信頼区間）
使い方: python planR_figs.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic"]
HERE = Path(__file__).resolve().parent
DATA, FIG = HERE.parent / "report_data", HERE.parent / "report_fig"
S = 50
T975 = {3: 4.303, 5: 2.776, 10: 2.262}   # t(0.975, n−1)


def ci95(x):
    x = np.asarray(x, float)
    return T975[len(x)] * x.std(ddof=1) / np.sqrt(len(x))


long = json.loads((DATA / "planR_flowchart_long.json").read_text(encoding="utf-8"))
main = json.loads((DATA / "planR_flowchart.json").read_text(encoding="utf-8"))

# ---- fig47 ----
fig, axes = plt.subplots(2, 10, figsize=(20, 4.6))
for row, var in enumerate(["今のモデル", "流れ図どおり（全部）"]):
    rs = sorted([r for r in long if r["var"] == var and r["name"] == "赤" and r["ascale"] == 2], key=lambda r: r["seed"])
    for ax, r in zip(axes[row], rs):
        p, th = np.array(r["pos"]), np.array(r["theta"])
        ax.quiver(p[:, 0], p[:, 1], np.cos(th), np.sin(th), color="tab:red", scale=25, width=0.012)
        ax.set_xlim(-1, S); ax.set_ylim(-1, S); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"seed {r['seed']}  整列度 {r['pol']:.2f}", fontsize=8)
    axes[row, 0].set_ylabel(var, fontsize=10)
fig.suptitle("fig47　赤（P=40, α=0.2、α 2倍・round）の t=3000 の位置と基本ベクトルの向き（整列度は t=2500〜3000 の平均）")
fig.tight_layout()
fig.savefig(FIG / "fig47_planR_red_snapshots.png", dpi=110)
plt.close(fig)

# ---- fig48 ----
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
t = (np.arange(60) + 0.5) * 50
for ax, name, paper in zip(axes[:2], ["赤", "緑"], [0.93, 0.35]):
    for var, col in [("今のモデル", "gray"), ("流れ図どおり（全部）", "tab:purple")]:
        rs = [r for r in long if r["var"] == var and r["name"] == name and r["ascale"] == 2]
        ts = np.array([r["pol_ts"] for r in rs])
        for y in ts:
            ax.plot(t, y, color=col, lw=0.5, alpha=0.4)
        ax.plot(t, ts.mean(0), color=col, lw=2, label=var)
    ax.axhline(paper, color="k", ls="--", lw=1, label=f"論文 {paper}")
    ax.set_title(f"{name}：整列度の時系列（α 2倍、細線はシードごと）"); ax.set_xlabel("t"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

ax = axes[2]
vars_ = list(dict.fromkeys(r["var"] for r in main))
paper = {"青": 0.2, "赤": 0.93, "緑": 0.35}
cols = {"青": "tab:blue", "赤": "tab:red", "緑": "tab:green"}
for j, name in enumerate(["青", "赤", "緑"]):
    for i, var in enumerate(vars_):
        x = [r["pol"] for r in main if r["var"] == var and r["name"] == name]
        ax.errorbar(i + (j - 1) * 0.22, np.mean(x), yerr=ci95(x), fmt="o", color=cols[name], ms=4, capsize=2,
                    label=name if i == 0 else None)
    ax.axhline(paper[name], color=cols[name], ls=":", lw=1)
ax.set_xticks(range(len(vars_))); ax.set_xticklabels(vars_, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("整列度（t=1000〜1500 の平均）"); ax.set_ylim(0, 1)
ax.set_title("読み方ごとの整列度（α 2倍、10シード、95% 信頼区間）\n点線は論文の値", fontsize=10)
ax.legend(fontsize=8)
fig.suptitle("fig48　流れ図（本文 図2）どおりの更新順の効果")
fig.tight_layout()
fig.savefig(FIG / "fig48_planR_polarization.png", dpi=110)
print("ok")
