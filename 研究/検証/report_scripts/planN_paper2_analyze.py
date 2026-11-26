"""計画N の集計と図（fig36〜fig41）。入力: report_data/planN/（planN_paper2_sim.py の出力）
エラーバー・帯はすべて 95% 信頼区間（シードの平均 ± t(0.975, n−1)·SD/√n）。
論文の値（目で読んだもの、論文2_図をつくる.ipynb と同じ）
  図3b,c（t が大きい所）：青 0.2 / 0.65、赤 0.93 / 4.0、緑 0.35 / 3.2（整列度 / 密度）
  図S2（相互予期 / 追従 / 自由移動、%）：青 63 / 5 / 32、赤 99 / 1 / 0、緑 98 / 2 / 0
  図S2b（2体）：0.88 / 0 / 0.12。本文 p.6：2体の整列度は約 0.9
  図S3（t=1500 の整列度）：ω=0 → 0.10、1 → 0.70、2 → 0.93、3 → 0.93、5 → 0.95、8 → 0.55、10 → 0.73
力の地図（図4b,c）は論文の Methods どおり：加速度 = v(t) − v(t−1)、向き = v(t−1)、相対的な向きは 0〜45°、
  ビンは向き 20° 幅を 5° ずつ、距離 2 幅を 0.5 ずつずらして重ねる（検証/05 の 3.8 節と同じ）
使い方: python planN_paper2_analyze.py [データのフォルダ名] [x1]
  既定は report_data/planN（ベスト）で fig36〜40。11/25 に α 1倍との比較用に引数を足した：
  python planN_paper2_analyze.py planN_x1 x1 で、α 1倍の同じ図を fig41〜45 として作る
  11/26：python planN_paper2_analyze.py planR_flow flow で、流れ図どおり（計画R）の同じ図を fig49〜53 として作る
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "report_data" / (sys.argv[1] if len(sys.argv) > 1 else "planN")
TAG = sys.argv[2] if len(sys.argv) > 2 else "best"
SETTING = {"best": "ベスト：α 2倍・round・ω 式どおり", "x1": "α 1倍：α そのまま・round・ω 式どおり",
           "flow": "流れ図どおり：α 2倍・round・ω 式どおり・相互予期はサイト中心"}[TAG]
SHORT = {"best": "ベスト（α 2倍）", "x1": "α 1倍", "flow": "流れ図どおり（α 2倍）"}[TAG]
NO = {"best": (36, 37, 38, 39, 40), "x1": (41, 42, 43, 44, 45), "flow": (49, 50, 51, 52, 53)}[TAG]
STEM = {"best": "planN", "x1": "planN_x1", "flow": "planR_flow"}[TAG]
FIG = HERE.parent / "report_fig"
S, L = 50, 6
T_DIST = {3: 4.303, 5: 2.776, 10: 2.262}
COL = {"青": "tab:blue", "赤": "tab:red", "緑": "tab:green"}
FIG3 = {"青": (0.2, 0.65), "赤": (0.93, 4.0), "緑": (0.35, 3.2)}
FIGS2 = {"青": (63, 5, 32), "赤": (99, 1, 0), "緑": (98, 2, 0)}
FIGS2B = (0.88, 0.0, 0.12)
FIGS3 = {0: 0.10, 1: 0.70, 2: 0.93, 3: 0.93, 5: 0.95, 8: 0.55, 10: 0.73}
SUMMARY = {}


def ci(x, axis=0):
    x = np.asarray(x, float)
    n = x.shape[axis]
    return x.mean(axis), T_DIST.get(n, 1.96) * x.std(axis, ddof=1) / np.sqrt(n)


def load(kind):
    d = np.load(DATA / f"{kind}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


# ---------------------------------------------------------------- 図3a（fig36）
g = load("grid")
keys = [m[0] for m in g["meta"]]
Ps, As = (10, 20, 30, 40), (0.1, 0.2, 0.3, 0.4, 0.5)
fig, axes = plt.subplots(4, 5, figsize=(12, 10.5))
SUMMARY["fig3a"] = {}
for i, P in enumerate(Ps):
    for j, a in enumerate(As):
        idx = [k for k, m in enumerate(g["meta"]) if m[0] == f"P{P}_a{a}"]
        pol = g["pol"][idx][:, 1000:].mean(1)
        mp, cp = ci(pol)
        SUMMARY["fig3a"][f"P{P}_a{a}"] = [round(float(mp), 3), round(float(cp), 3)]
        k0 = [k for k in idx if g["meta"][k][4] == 0][0]
        last = g["last"][k0]                          # (4, N, 2)
        ax = axes[i, j]
        for n in range(last.shape[1]):
            tr = last[:, n, :]
            for t in range(3):
                if np.abs(tr[t + 1] - tr[t]).max() < S / 2:
                    ax.plot(tr[t:t + 2, 0], tr[t:t + 2, 1], color="0.55", lw=0.6)
        ax.scatter(last[-1, :, 0], last[-1, :, 1], s=8, marker="s", color="k")
        ax.set_xlim(-0.5, S - 0.5); ax.set_ylim(-0.5, S - 0.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        frame = {(10, 0.1): "tab:blue", (40, 0.2): "tab:red", (40, 0.5): "tab:green"}.get((P, a))
        if frame:
            for s_ in ax.spines.values():
                s_.set_color(frame); s_.set_linewidth(3)
        ax.set_xlabel(f"{mp:.2f} ± {cp:.2f}", fontsize=8)
        if i == 0:
            ax.set_title(f"α={a}")
        if j == 0:
            ax.set_ylabel(f"P={P}")
fig.suptitle(f"fig{NO[0]}　図3a と同じ格子（{SETTING}、seed 0 の t=1500、3ステップの軌跡）\n下の数字：整列度（t=1000〜1500、5シードの平均 ± 95% 信頼区間）", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / f"fig{NO[0]}_{STEM}_fig3a.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- 図3b,c・図S2a（fig37）
tm = load("time")
names = ["青", "赤", "緑"]
t = np.arange(1, 1501)
fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
SUMMARY["fig3bc"], SUMMARY["figS2"] = {}, {}
for nm in names:
    idx = [k for k, m in enumerate(tm["meta"]) if m[0] == nm]
    pol, dens, rules = tm["pol"][idx], tm["dens"][idx], tm["rules"][idx].astype(float)
    s0 = [k for k, m in enumerate(tm["meta"]) if m[0] == nm and m[4] == 0][0]
    for a_, y, y0, ref in ((ax[0, 0], pol, tm["pol"][s0], FIG3[nm][0]), (ax[0, 1], dens, tm["dens"][s0], FIG3[nm][1])):
        m_, c_ = ci(y)
        a_.fill_between(t, m_ - c_, m_ + c_, color=COL[nm], alpha=0.2, lw=0)
        a_.plot(t, y0, color=COL[nm], lw=0.8)
        a_.axhline(ref, color=COL[nm], ls=":", lw=1.5)
    tp = pol[:, 1000:].mean(1); td = dens[:, 1000:].mean(1)
    # そろう時刻（整列度の 50 ステップ移動平均が、t=1000〜1500 の平均の 90% に初めて届く時刻）
    t90 = []
    for y in pol:
        ma = np.convolve(y, np.ones(50) / 50, mode="valid")
        hit = np.flatnonzero(ma >= 0.9 * y[1000:].mean())
        t90.append(int(hit[0] + 50) if hit.size else None)
    SUMMARY["fig3bc"][nm] = dict(pol=[round(float(np.median(tp)), 3), round(float(tp.min()), 3), round(float(tp.max()), 3)],
                                 dens=[round(float(np.median(td)), 3), round(float(td.min()), 3), round(float(td.max()), 3)],
                                 pol_ci=[round(float(v), 3) for v in ci(tp)], dens_ci=[round(float(v), 3) for v in ci(td)],
                                 t90=t90)
    share = rules[:, 1000:, :3].mean(1) * 100 / 100        # 個体数（N=100）なので % と同じ
    SUMMARY["figS2"][nm] = [[round(float(v), 1) for v in ci(share[:, k])] for k in range(3)]
    for k in range(3):
        m_, c_ = ci(rules[:, :, k])
        sm = np.convolve(m_, np.ones(10) / 10, mode="valid")     # 端で値が下がって見えないよう valid にする
        tv, cv = t[9:], c_[9:]
        ax[1, k].plot(tv, sm, color=COL[nm], lw=1.2)
        ax[1, k].fill_between(tv, sm - cv, sm + cv, color=COL[nm], alpha=0.2, lw=0)
        ax[1, k].axhline(FIGS2[nm][k], color=COL[nm], ls=":", lw=1.5)
ax[0, 0].set_title("図3b：整列度（細線＝seed 0、帯＝10シードの 95% 信頼区間、点線＝論文）"); ax[0, 0].set_ylim(0, 1)
ax[0, 1].set_title("図3c：密度（周り3×3 の個体数）")
ax[0, 2].axis("off")
txt = "t=1000〜1500 の平均（10シードの中央値［最小〜最大］）\n"
for nm in names:
    s_ = SUMMARY["fig3bc"][nm]
    txt += f"\n{nm}：整列度 {s_['pol'][0]:.2f}［{s_['pol'][1]:.2f}〜{s_['pol'][2]:.2f}］（論文 {FIG3[nm][0]}）\n　　密度 {s_['dens'][0]:.2f}［{s_['dens'][1]:.2f}〜{s_['dens'][2]:.2f}］（論文 {FIG3[nm][1]}）\n"
ax[0, 2].text(0, 0.95, txt, va="top", fontsize=10)
for k, nm_ in enumerate(("図S2a1：相互予期の個体数", "図S2a2：追従の個体数", "図S2a3：自由移動の個体数")):
    ax[1, k].set_title(nm_ + "（10 ステップ移動平均）"); ax[1, k].set_ylim(-2, 102); ax[1, k].set_xlabel("time step")
fig.suptitle(f"fig{NO[1]}　図3b,c と図S2a（{SETTING}）", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / f"fig{NO[1]}_{STEM}_fig3bc_S2a.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- 図S3（fig38）
om = load("omega")
ws = list(range(11))
pe, de, pa, da = [], [], [], []
for w in ws:
    idx = [k for k, m in enumerate(om["meta"]) if m[0] == f"w{w}"]
    pe.append(ci(om["pol"][idx][:, -1])); de.append(ci(om["dens"][idx][:, -1]))
    pa.append(ci(om["pol"][idx][:, 1000:].mean(1))); da.append(ci(om["dens"][idx][:, 1000:].mean(1)))
SUMMARY["figS3"] = {w: dict(pol_t1500=[round(float(v), 3) for v in pe[w]], pol_avg=[round(float(v), 3) for v in pa[w]],
                            dens_t1500=[round(float(v), 3) for v in de[w]]) for w in ws}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for a_, P_, D_, ttl in ((ax[0], pe, de, "t=1500 の値（論文と同じ）"), (ax[1], pa, da, "t=1000〜1500 の時間平均")):
    a_.errorbar(ws, [p[0] for p in P_], yerr=[p[1] for p in P_], color="tab:red", marker="o", capsize=3, label="整列度")
    a2 = a_.twinx()
    a2.errorbar(np.array(ws) + 0.1, [d[0] for d in D_], yerr=[d[1] for d in D_], color="tab:blue", marker="s", capsize=3, label="密度")
    a_.plot(list(FIGS3), list(FIGS3.values()), "k*", ms=12, label="論文の整列度")
    a_.set_ylim(0, 1.05); a2.set_ylim(1.5, 5); a_.set_xlabel("ω"); a_.set_ylabel("整列度", color="tab:red"); a2.set_ylabel("密度", color="tab:blue")
    a_.set_title(ttl); a_.legend(loc="lower center", fontsize=8); a_.grid(alpha=0.3)
fig.suptitle(f"fig{NO[2]}　図S3：ω を変えたとき（P=40、α=0.2、{SHORT}、10シードの平均 ± 95% 信頼区間）", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / f"fig{NO[2]}_{STEM}_figS3.png", dpi=110); plt.close(fig)


# ---------------------------------------------------------------- 2体・3体の力
def unwrap(pos):
    d = (np.diff(pos.astype(float), axis=0) + S / 2) % S - S / 2
    return np.concatenate([pos[:1].astype(float), pos[0] + np.cumsum(d, axis=0)])


def force_samples(pos):
    """focal ごと・相手ごとの (focal, 相手, t, 左右の距離, 前後の距離, 相対的な向き[度、0〜180], 旋回の力, 加速の力)"""
    X = unwrap(pos)                         # (T+1, N, 2)
    v = np.diff(X, axis=0)                  # v[t] = X[t+1] − X[t]
    acc = v[1:] - v[:-1]                    # 時刻 t=1.. の加速度
    head = v[:-1]                           # 向き＝1つ前の速度
    Xt = X[1:-1]
    N = pos.shape[1]
    rows = []
    for f in range(N):
        h = head[:, f]; nh = np.linalg.norm(h, axis=1)
        u = np.zeros_like(h); ok = nh > 1e-9; u[ok] = h[ok] / nh[ok, None]
        nrm = np.stack([-u[:, 1], u[:, 0]], 1)      # 左向きの法線
        turn = -(acc[:, f] * nrm).sum(1)            # 右向きを正
        speed = (acc[:, f] * u).sum(1)
        for g_ in range(N):
            if g_ == f:
                continue
            hg = head[:, g_]; ng = np.linalg.norm(hg, axis=1)
            r = (Xt[:, g_] - Xt[:, f] + S / 2) % S - S / 2
            lr = -(r * nrm).sum(1); fb = (r * u).sum(1)
            cosang = np.clip((u * hg).sum(1) / np.maximum(ng, 1e-9), -1, 1)
            ang = np.degrees(np.arccos(cosang))
            good = ok & (ng > 1e-9)
            tt = np.arange(len(r))
            rows.append(np.stack([np.full(len(r), f), np.full(len(r), g_), tt, lr, fb, ang, turn, speed], 1)[good])
    return np.concatenate(rows)


def overlap_map(x, y, val, xlo, xw, ylo, yw):
    M = np.full((len(ylo), len(xlo)), np.nan)
    for i, y0 in enumerate(ylo):
        sy = (y >= y0) & (y < y0 + yw)
        for j, x0 in enumerate(xlo):
            s_ = sy & (x >= x0) & (x < x0 + xw)
            if s_.sum() >= 20:
                M[i, j] = val[s_].mean()
    return M


def col_share(M):
    """地図のばらつきのうち、列（距離）の違いで説明できる割合"""
    ok = ~np.isnan(M)
    tot = np.nanvar(M)
    cm = np.nanmean(M, axis=0, keepdims=True)
    return float(1 - np.nanvar(np.where(ok, M - cm, np.nan)) / tot) if tot > 0 else np.nan


pair = load("pair")
PS = np.concatenate([force_samples(p) for p in pair["pos"]])
pol_pair, dist_pair = [], []
for p in pair["pos"]:
    X = unwrap(p); v = np.diff(X, axis=0)
    n = np.linalg.norm(v, axis=2); ok = (n > 0).all(1)
    uv = v[ok] / n[ok][:, :, None]
    pol_pair.append(np.linalg.norm(uv.mean(1), axis=1).mean())
    r = (p[:, 1].astype(float) - p[:, 0] + S / 2) % S - S / 2
    dist_pair.append(np.linalg.norm(r, axis=1).mean())
rules_pair = pair["rules"][:, :3]
SUMMARY["pair"] = dict(pol=[round(float(v), 3) for v in ci(pol_pair)], dist=[round(float(v), 2) for v in ci(dist_pair)],
                       rules=[[round(float(v), 3) for v in ci(rules_pair[:, k])] for k in range(3)])

XLO = np.arange(-4, 2.01, 0.5); YLO = np.arange(0, 26, 5)
lr, fb, ang, turn, speed = PS[:, 3], PS[:, 4], PS[:, 5], PS[:, 6], PS[:, 7]
Mt = overlap_map(lr, ang, turn, XLO, 2, YLO, 20)
Ms = overlap_map(fb, ang, speed, XLO, 2, YLO, 20)
SUMMARY["fig4"] = dict(turn_share=round(col_share(Mt), 3), speed_share=round(col_share(Ms), 3),
                       turn_range=[round(float(np.nanmin(Mt)), 2), round(float(np.nanmax(Mt)), 2)],
                       speed_range=[round(float(np.nanmin(Ms)), 2), round(float(np.nanmax(Ms)), 2)])
fig = plt.figure(figsize=(16, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])
for k, (M, ttl, xl) in enumerate(((Mt, "図4b：旋回の力（赤＝右へ）", "左右の距離（右が正）"),
                                  (Ms, "図4c：加速の力（赤＝速くなる）", "前後の距離（前が正）"))):
    a_ = fig.add_subplot(gs[k])
    vmax = np.nanmax(np.abs(M))
    im = a_.imshow(M, origin="lower", aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax),
                   extent=[XLO[0] + 1 - 0.25, XLO[-1] + 1 + 0.25, YLO[0] + 10 - 2.5, YLO[-1] + 10 + 2.5])
    plt.colorbar(im, ax=a_); a_.set_title(ttl); a_.set_xlabel(xl); a_.set_ylabel("相対的な向き（度）")
a_ = fig.add_subplot(gs[2])
x = np.arange(3)
mr = [ci(rules_pair[:, k]) for k in range(3)]
a_.bar(x - 0.2, [m[0] for m in mr], 0.4, yerr=[m[1] for m in mr], capsize=4, color="0.4", label="モデル（5シード、95% 信頼区間）")
a_.bar(x + 0.2, FIGS2B, 0.4, color="0.8", edgecolor="k", label="論文 図S2b")
a_.set_xticks(x); a_.set_xticklabels(["相互予期", "追従", "自由移動"]); a_.set_ylim(0, 1); a_.legend(fontsize=8)
a_.set_title(f"図S2b：2体のルールの割合\n整列度 {SUMMARY['pair']['pol'][0]:.2f}（論文 約 0.9）、平均距離 {SUMMARY['pair']['dist'][0]:.1f}")
fig.suptitle(f"fig{NO[3]}　2体（N=2、P=40、α=0.2、{SHORT}、30000 ステップ × 5シード）：図4b,c と図S2b", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / f"fig{NO[3]}_{STEM}_pair.png", dpi=110); plt.close(fig)

# 3体（図S4）：両隣の位置を focal の向きに射影した 2次元の地図。予測は2体の力（距離だけの関数）の平均
trio = load("trio")
TS = np.concatenate([np.c_[np.full(len(s_), k), s_] for k, s_ in enumerate(force_samples(p) for p in trio["pos"])])
# TS 列：seed, focal, 相手, t, lr, fb, ang, turn, speed
B = np.arange(-4, 4.01, 1.0)        # ビンの中心（幅 1）


def pair_1d(x, val):
    out = np.full(len(B), np.nan)
    for i, c in enumerate(B):
        s_ = np.abs(x - c) < 0.5
        if s_.sum() >= 20:
            out[i] = val[s_].mean()
    return out


F2_speed = pair_1d(fb, speed); F2_turn = pair_1d(lr, turn)
maps = {}
for comp, col_pos, col_val, F2 in (("speed", 5, 8, F2_speed), ("turn", 4, 7, F2_turn)):
    sim = np.full((len(B), len(B)), np.nan); cnt = np.zeros_like(sim)
    # 同じ (seed, focal, t) の2行（相手2体）を組にする
    key = TS[:, 0] * 1e7 + TS[:, 1] * 1e6 + TS[:, 3]
    order = np.argsort(key, kind="stable"); Ts = TS[order]; ks = key[order]
    first = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    first = first[(first + 1 < len(Ts))]
    first = first[ks[first] == ks[first + 1]]
    d1 = Ts[first, col_pos]; d2 = Ts[first + 1, col_pos]; val = Ts[first, col_val]
    i1 = np.round(d1).astype(int) + 4; i2 = np.round(d2).astype(int) + 4
    okk = (i1 >= 0) & (i1 < len(B)) & (i2 >= 0) & (i2 < len(B))
    acc_ = np.zeros_like(sim); np.add.at(acc_, (i1[okk], i2[okk]), val[okk]); np.add.at(cnt, (i1[okk], i2[okk]), 1)
    sim = np.where(cnt >= 20, acc_ / np.maximum(cnt, 1), np.nan)
    sim = (sim + sim.T) / 2                           # 2体の入れ替えで対称にする（論文の予測図と同じ扱い）
    pred = (F2[:, None] + F2[None, :]) / 2
    maps[comp] = (sim, pred, sim - pred)
# 「focal が両隣の間にいるとき、残差が強い」か：d1·d2<0 のマスと、同じ側のマスの |残差| の平均
SUMMARY["trio"] = {}
for comp in maps:
    sim, pred, diff = maps[comp]
    bx = B[:, None] * B[None, :]
    SUMMARY["trio"][comp] = dict(between=round(float(np.nanmean(np.abs(diff[bx < 0]))), 3),
                                 same_side=round(float(np.nanmean(np.abs(diff[bx > 0]))), 3),
                                 corr_sim_pred=round(float(np.corrcoef(sim[~np.isnan(sim)], pred[~np.isnan(sim)])[0, 1]), 3))
fig, ax = plt.subplots(2, 3, figsize=(14, 9))
for r_, (comp, lab) in enumerate((("speed", "前後の距離"), ("turn", "左右の距離"))):
    vmax = np.nanmax(np.abs(np.stack(maps[comp][:2])))
    for c_, (M, ttl) in enumerate(zip(maps[comp], ("シミュレーション（b）", "2体の力の平均で予測（c）", "差 b − c（d）"))):
        im = ax[r_, c_].imshow(M.T, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), extent=[-4.5, 4.5, -4.5, 4.5])
        plt.colorbar(im, ax=ax[r_, c_])
        ax[r_, c_].set_title(f"{'加速' if comp == 'speed' else '旋回'}の力：{ttl}")
        ax[r_, c_].set_xlabel(f"{lab} N1"); ax[r_, c_].set_ylabel(f"{lab} N2")
fig.suptitle(f"fig{NO[4]}　3体（図S4、付録の説明では Figure S3）：N=3、P=40、α=0.2、{SHORT}、30000 ステップ × 5シード", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / f"fig{NO[4]}_{STEM}_trio.png", dpi=110); plt.close(fig)

json.dump(SUMMARY, open(DATA / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(SUMMARY, ensure_ascii=False, indent=1))
