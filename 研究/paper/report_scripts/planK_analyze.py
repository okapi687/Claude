"""計画Kの集計と図。引数: planK_scalefree2.py の json、planK_size.py の json、図の出力 png
論文：図15d（モデル）は C ≈ 1.05 − 0.96·(r/ξ)、図15c（実際のカニ）は C ≈ 0.88 − 0.88·(r/ξ)（どちらも目で読んだもの）
"""
import sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]

NAMES = {"v1": "v1（α そのまま・floor）", "as2_round": "α 2倍・round", "as1_round_E2": "α そのまま・round＋E2"}
VELS = {"disp1": "移動1ステップ", "disp5": "移動5ステップ", "theta": "基本ベクトル θ"}
DOMS = {"all": "全体", "clus": "一番大きい群れ"}
XB = np.arange(0, 1.65, 0.05); XC = (XB[:-1] + XB[1:]) / 2

res = json.load(open(sys.argv[1]))
g = defaultdict(list)
for r in res:
    for key, f in r["fits"].items():
        if f is not None:
            g[(r["tag"], key)].append(f)

print("## 傾き / 切片 / R² / ξ（計算できた実行数 / 18）")
print("| 設定 | 範囲 | " + " | ".join(VELS.values()) + " |")
print("|---|---|" + "---|" * len(VELS))
for tag in NAMES:
    for d in DOMS:
        cells = []
        for v in VELS:
            L = g[(tag, f"{v}|{d}")]
            if not L:
                cells.append("—（0/18）"); continue
            m = lambda q: np.mean([f[q] for f in L])
            cells.append(f"{m('slope'):+.2f} / {m('icpt'):.2f} / {m('r2'):.2f} / ξ={m('xi'):.1f}（{len(L)}/18）")
        print(f"| {NAMES[tag]} | {DOMS[d]} | " + " | ".join(cells) + " |")

size = json.load(open(sys.argv[2]))
print("\n## ξ と群れの大きさ（N を変える、範囲は一番大きい群れ）")
sz = defaultdict(list)
for r in size:
    for row in r["rows"]:
        sz[(r["tag"], r["N"])].append(row)
for (tag, N), rows in sorted(sz.items()):
    s = np.median([q["size"] for q in rows]); n = np.median([q["n"] for q in rows])
    x1 = [q["disp1"] for q in rows if q["disp1"]]; x5 = [q["disp5"] for q in rows if q["disp5"]]
    print(f"{NAMES[tag]} N={N}: 群れ {n:.0f} 個体, 大きさ {s:.1f}, ξ(1)={np.median(x1):.1f}, ξ(5)={np.median(x5):.1f}, "
          f"ξ(5)/大きさ={np.median(x5)/s:.2f}")

fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
x = np.linspace(0, 1.4, 10)
for k, tag in enumerate(NAMES):
    a = ax[k]
    for v, c in zip(VELS, ["C0", "C1", "C3"]):
        L = g[(tag, f"{v}|clus")]
        if not L:
            continue
        P = np.array([[np.nan if q is None else q for q in f["prof"]] for f in L])
        a.plot(XC, np.nanmean(P, 0), color=c, lw=2, label=f"{VELS[v]}（{len(L)}/18）")
    a.plot(x, 1.05 - 0.96 * x, "k--", label="論文 図15d（モデル）")
    a.plot(x, 0.88 - 0.88 * x, "k:", label="論文 図15c（カニ）")
    a.axhline(0, color="gray", lw=0.5)
    a.set_xlim(0, 1.5); a.set_ylim(-0.4, 1.2)
    a.set_title(NAMES[tag] + "：一番大きい群れ"); a.set_xlabel("r / ξ"); a.set_ylabel("C（自分との積で割る）")
    a.legend(fontsize=8)
a = ax[3]
for tag, mk in zip(["as2_round", "as1_round_E2"], ["o", "s"]):
    for v, c in (("disp1", "C0"), ("disp5", "C1")):
        xs, ys = [], []
        for (t2, N), rows in sorted(sz.items()):
            if t2 != tag:
                continue
            xs.append(np.median([q["size"] for q in rows]))
            ys.append(np.median([q[v] for q in rows if q[v]]))
        a.plot(xs, ys, marker=mk, color=c, label=f"{NAMES[tag]}・{VELS[v]}")
a.set_xlabel("群れの大きさ（重心からの平均距離）"); a.set_ylabel("相関長 ξ")
a.set_title("ξ と群れの大きさ（N=50〜400）"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(sys.argv[3], dpi=110)
print("saved", sys.argv[3])
