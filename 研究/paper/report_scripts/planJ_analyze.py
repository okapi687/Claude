"""計画J の集計。引数: planJ_best.py の出力 json、図の出力先フォルダ
図: fig29_planJ_p2_time.png（論文2 図3b,c の時系列）、fig30_planJ_snapshots.png（ベストのモデルのスナップショット）
"""
import os, sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.load(open(sys.argv[1]))
outdir = sys.argv[2]
COL = {"blue": "tab:blue", "red": "tab:red", "green": "tab:green"}
PAPER = {"blue": (0.2, 0.65), "red": (0.93, 4.0), "green": (0.35, 3.2)}


def ma(x, w=50):
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[w:] - c[:-w]) / w  # 長さ len(x)-w+1、i 番目は t=i..i+w-1 の平均


def t90(x, rise_min):
    """移動平均（50 ステップ）が、最初の値から t=1000〜1500 の平均までの 90% に初めて届くステップ。上がり幅が rise_min 未満なら None"""
    x = np.asarray(x, float)
    init, plat = x[:20].mean(), x[1000:1500].mean()
    if plat - init < rise_min:
        return None
    m = ma(x)
    idx = np.flatnonzero(m >= init + 0.9 * (plat - init))
    return int(idx[0] + 50) if len(idx) else None


p2 = defaultdict(list)
for r in res:
    if r["part"] == "p2":
        p2[(r["cond"], r["set"])].append(r)

print("## 論文2：そろう・詰まるまでのステップ数（t90：移動平均50が、最初の値から t=1000〜1500 の平均までの 90% に届く時刻）")
print("| 条件 | 設定 | シード数 | 整列度 t90（中央値 [最小〜最大]） | 密度 t90（中央値 [最小〜最大]） | 平均の時系列の t90（整列度 / 密度） |")
print("|---|---|---|---|---|---|")
for cond in ["blue", "red", "green"]:
    for sn in ["best", "murakami", "floor"]:
        rs = p2[(cond, sn)]
        cells = []
        for key, rmin in [("pol", 0.2), ("dens", 0.5)]:
            ts = [t90(r[key], rmin) for r in rs]
            ok = [t for t in ts if t is not None]
            if len(ok) == 0:
                cells.append("上がらない")
            else:
                cells.append(f"{int(np.median(ok))} [{min(ok)}〜{max(ok)}]" + (f"（{len(rs) - len(ok)} 本は上がらない）" if len(ok) < len(rs) else ""))
        mp = t90(np.mean([r["pol"] for r in rs], 0), 0.2); md = t90(np.mean([r["dens"] for r in rs], 0), 0.5)
        print(f"| {cond} | {sn} | {len(rs)} | {cells[0]} | {cells[1]} | {mp if mp else '—'} / {md if md else '—'} |")

print("\n## 論文2：平均の取り方による違い（整列度 / 密度）")
print("| 条件 | 設定 | 論文 | ① 各シード t=1000〜1500 の平均 → シードの平均 ± 標準偏差 | ② ①のシードの中央値 | ③ t=1500 の1点 → シードの平均 ± 標準偏差 | ④ t=200〜1500 の平均 | ⑤ 1本だけ（seed 0）の t=1500 |")
print("|---|---|---|---|---|---|---|---|")
for cond in ["blue", "red", "green"]:
    for sn in ["best", "murakami"]:
        rs = p2[(cond, sn)]
        P = np.array([r["pol"] for r in rs]); D = np.array([r["dens"] for r in rs])
        a1p, a1d = P[:, 1000:1500].mean(1), D[:, 1000:1500].mean(1)
        print(f"| {cond} | {sn} | {PAPER[cond][0]} / {PAPER[cond][1]} | "
              f"{a1p.mean():.2f}±{a1p.std():.2f} / {a1d.mean():.2f}±{a1d.std():.2f} | "
              f"{np.median(a1p):.2f} / {np.median(a1d):.2f} | "
              f"{P[:, -1].mean():.2f}±{P[:, -1].std():.2f} / {D[:, -1].mean():.2f}±{D[:, -1].std():.2f} | "
              f"{P[:, 200:].mean():.2f} / {D[:, 200:].mean():.2f} | {P[0, -1]:.2f} / {D[0, -1]:.2f} |")

print("\n## 論文2：1ステップごとの揺らぎ（t=1000〜1500 の、各シードの時間方向の標準偏差の平均）")
for cond in ["blue", "red", "green"]:
    rs = p2[(cond, "best")]
    print(f"- {cond}: 整列度 {np.mean([np.std(r['pol'][1000:1500]) for r in rs]):.3f}、密度 {np.mean([np.std(r['dens'][1000:1500]) for r in rs]):.3f}、"
          f"相互予期 {np.mean([np.mean(r['ma'][1000:1500]) for r in rs]):.2f}")

# 図29：時系列
fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
for j, sn in enumerate(["best", "murakami", "floor"]):
    for cond in ["blue", "red", "green"]:
        rs = p2[(cond, sn)]
        for i, (key, lab) in enumerate([("pol", "polarity"), ("dens", "density")]):
            X = np.array([r[key] for r in rs])
            for x in X:
                ax[i, j].plot(np.arange(50, 1501), ma(x), color=COL[cond], lw=0.5, alpha=0.35)
            ax[i, j].plot(np.arange(1, 1501), X.mean(0), color=COL[cond], lw=1.8)
            ax[i, j].axhline(PAPER[cond][i], color=COL[cond], ls=":", lw=1)
            ax[i, j].set_ylabel(lab)
    ax[0, j].set_title({"best": "best: alpha x2, omega eq.(13), round", "murakami": "alpha x1, omega eq.(13), round",
                        "floor": "alpha x2, omega eq.(13), floor"}[sn])
    ax[1, j].set_xlabel("time step")
fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig29_planJ_p2_time.png"), dpi=110)

# 論文1
p1 = defaultdict(list)
for r in res:
    if r["part"] == "p1":
        p1[(r["set"], r["alpha"])].append(r)
print("\n## 論文1（P=15、t=300〜900 の平均、5シード）")
print("| 設定 | α | 整列度 | θ のそろい φ | 密度 区画 Ls=10 | 密度 区画 Ls=25 | 相互予期 / 追従 / 自由移動 | 整列度の t90 |")
print("|---|---|---|---|---|---|---|---|")
for sn in ["A_as2_round_theta", "B_as1_round_disp", "v1_default"]:
    for a in (0.3, 0.5):
        rs = p1[(sn, a)]
        g = lambda k: np.mean([np.mean(r[k][300:]) for r in rs])
        ts = []
        for r in rs:
            x = np.asarray(r["pol"]); init, plat = x[:20].mean(), x[300:900].mean()
            m = ma(x); idx = np.flatnonzero(m >= init + 0.9 * (plat - init)) if plat - init > 0.05 else []
            if len(idx): ts.append(int(idx[0] + 50))
        print(f"| {sn} | {a} | {g('pol'):.2f} | {g('phi'):.2f} | {g('d10'):.2f} | {g('d25'):.2f} | "
              f"{g('ma'):.2f} / {g('fol'):.2f} / {g('free'):.2f} | {int(np.median(ts)) if ts else '—'} |")

# 図30：スナップショット（論文2 best の3条件 + 論文1 の3設定 × α 2つ）
fig, ax = plt.subplots(3, 3, figsize=(12, 12))
for j, cond in enumerate(["blue", "red", "green"]):
    r = p2[(cond, "best")][0]
    tail = np.array(r["tail"])
    a = ax[0, j]
    for k in range(tail.shape[1]):
        tr = tail[:, k]
        if np.abs(np.diff(tr, axis=0)).max() < 25:
            a.plot(tr[:, 0], tr[:, 1], color="0.6", lw=0.8)
    a.scatter(tail[-1, :, 0], tail[-1, :, 1], s=12, marker="s", color=COL[cond])
    P_, al = {"blue": (10, 0.1), "red": (40, 0.2), "green": (40, 0.5)}[cond]
    a.set_title(f"paper2 best  P={P_}, alpha={al} (t=1500)\npol={np.mean(r['pol'][1000:]):.2f}, dens={np.mean(r['dens'][1000:]):.2f}", fontsize=9)
for i, sn in enumerate(["A_as2_round_theta", "B_as1_round_disp"]):
    for j, a_ in enumerate((0.3, 0.5)):
        r = [x for x in p1[(sn, a_)] if x["seed"] == 0][0]
        tail = np.array(r["tail"]); a = ax[i + 1, j]
        for k in range(tail.shape[1]):
            tr = tail[:, k]
            if np.abs(np.diff(tr, axis=0)).max() < 25:
                a.plot(tr[:, 0], tr[:, 1], color="0.6", lw=0.8)
        a.scatter(tail[-1, :, 0], tail[-1, :, 1], s=12, marker="s", color="k")
        a.set_title(f"paper1 {sn}  P=15, alpha={a_} (t=900)\npol={np.mean(r['pol'][300:]):.2f}, phi={np.mean(r['phi'][300:]):.2f}", fontsize=9)
    # 右の列：整列度の時系列
    a = ax[i + 1, 2]
    for a_, c in ((0.3, "tab:orange"), (0.5, "tab:purple")):
        for r in p1[(sn, a_)]:
            a.plot(np.arange(50, 901), ma(r["pol"]), color=c, lw=0.7, alpha=0.6)
    a.set_title(f"paper1 {sn}: polarization (MA50)\norange alpha=0.3, purple alpha=0.5", fontsize=9)
    a.set_xlabel("time step")
for a in ax[:, :2].ravel().tolist() + [ax[0, 2]]:
    a.set_xlim(0, 50); a.set_ylim(0, 50); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig30_planJ_snapshots.png"), dpi=110)
print("\nfigs saved")
