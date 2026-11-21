"""計画Iの集計と図。引数: planI_scalefree.py の出力 json、図の出力 png
論文：図15d（モデル）は C ≈ 1.05 − 0.96·(r/ξ)、図15c（実際のカニ）は C ≈ 0.88 − 0.88·(r/ξ)（どちらも目で読んだもの）
"""
import sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"v1": "v1（α そのまま・floor）", "as2_round": "α 2倍・round", "as1_round_E2": "α そのまま・round＋E2",
         "as2_round_E2": "α 2倍・round＋E2"}
NORMS = {"self": "自分との積で割る", "first": "一番近い距離の C で割る", "raw": "割らない"}
XB = np.arange(0, 1.65, 0.05); XC = (XB[:-1] + XB[1:]) / 2

res = json.load(open(sys.argv[1]))
g = defaultdict(list)
for r in res:
    for key, f in r["fits"].items():
        if f is not None:
            g[(r["tag"], key)].append((r, f))
taus = sorted({int(k.split("|")[1]) for r in res for k in r["fits"]})

for norm in NORMS:
    print(f"\n## 規格化：{NORMS[norm]}（P・α・シードの平均。傾き / 切片 / R²）")
    print("| 設定 | " + " | ".join(f"速度 {t} ステップ" for t in taus) + " |")
    print("|---|" + "---|" * len(taus))
    print("| 論文（図15d） | 傾き −0.96 / 切片 1.05 | | | | |")
    for tag in NAMES:
        cells = []
        for t in taus:
            v = g[(tag, f"{norm}|{t}")]
            if not v:
                cells.append("—"); continue
            cells.append(f"{np.mean([f['slope'] for _, f in v]):+.2f} / {np.mean([f['icpt'] for _, f in v]):.2f} / {np.mean([f['r2'] for _, f in v]):.2f}")
        print(f"| {NAMES[tag]} | " + " | ".join(cells) + " |")

print("\n## 規格化「自分との積」・速度1ステップの傾きを α と P ごとに")
for tag in NAMES:
    v = g[(tag, "self|1")]
    tab = defaultdict(list)
    for r, f in v:
        tab[(r["alpha"], r["P"])].append(f["slope"])
    print(NAMES[tag] + ": " + " ".join(f"α{a}/P{P}:{np.mean(s):+.2f}" for (a, P), s in sorted(tab.items())))

print("\n## 相関長 ξ（自分との積・1ステップ、格子の単位）")
for tag in NAMES:
    v = g[(tag, "self|1")]
    print(f"{NAMES[tag]}: {np.mean([f['xi'] for _, f in v]):.1f}")

C = {"v1": "#0b0b0b", "as2_round": "#2a78d6", "as1_round_E2": "#eb6834", "as2_round_E2": "#1baf7a"}
plt.rcParams.update({"font.family": ["Yu Gothic", "MS Gothic"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight"})
panels = [("self", 1, "自分との積で割る・速度1ステップ"), ("self", 10, "自分との積で割る・速度10ステップ"),
          ("raw", 1, "割らない・速度1ステップ（格子の単位）")]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
xs = np.linspace(0, 1.6, 50)
for ax, (norm, t, title) in zip(axes, panels):
    for tag in NAMES:
        v = g[(tag, f"{norm}|{t}")]
        prof = np.array([[np.nan if p is None else p for p in f["prof"]] for _, f in v], float)
        ax.plot(XC, np.nanmean(prof, 0), "-", color=C[tag], lw=2, label=NAMES[tag])
    ax.plot(xs, 1.05 - 0.96 * xs, "--", color="#8a8984", lw=1.5, label="論文 図15d（モデル）")
    ax.plot(xs, 0.88 - 0.88 * xs, ":", color="#8a8984", lw=1.5, label="論文 図15c（カニ）")
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.set_xlim(0, 1.6); ax.set_ylim(-0.6, 2.0 if norm == "raw" else 1.2); ax.set_xlabel("r / ξ"); ax.set_title(title, fontsize=10)
axes[0].set_ylabel("相関関数 C")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1), fontsize=9)
fig.suptitle("計画I：スケールフリー相関（R(M)=2、P=10〜20・α=0.3〜0.7・3シードの平均）", y=1.02)
fig.savefig(sys.argv[2]); print("\nfig:", sys.argv[2])
