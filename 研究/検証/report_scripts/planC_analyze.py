"""計画Cの集計。引数: planC_pair.py の出力 npz、集計結果の出力 json、図の出力 png
論文2の Methods（Force map）どおりの力の地図を作り、計算方法を変えて比べる
  論文の書き方：加速度は速度 v_i,t から求める（F=ma, m=1）。旋回の力は速度に垂直な成分、加速の力は平行な成分
  地図の縦軸は「相対的な向き」0〜45°、横軸は「左右の距離」-4〜4（図4b）
  ビンは重ねる：角度は幅 20° を 5° ずつ、距離は幅 2 を 0.5 ずつずらす
計算方法の候補
  backward : 向き = v(t-1)、加速度 = v(t) - v(t-1)（11/19 の解析と同じ）
  central  : 向き = (v(t-1) + v(t)) / 2、加速度 = v(t) - v(t-1)
  smooth w : 位置を幅 w の移動平均でならしてから central と同じ計算（w = 3, 5, 10）
「向きの差の符号を残した」地図（-45〜45°）も作り、旋回の力が左右の距離と向きの差のどちらで決まるかを回帰で比べる
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = 50
D = np.load(sys.argv[1])
TAGS = {"principal_x1": "ω 式どおり・α そのまま", "principal_x2": "ω 式どおり・α 2倍", "velocity_x1": "ω 移動側（v1 既定）・α そのまま"}
METHODS = ["backward", "central", "smooth3", "smooth5", "smooth10"]
TH_LO = np.arange(0, 26, 5)                 # 0-20, 5-25, ..., 25-45
D_LO = np.arange(-4, 2.01, 0.5)             # -4〜-2, ..., 2〜4
TH_S_LO = np.arange(-45, 26, 5)             # 符号つき：-45〜-25, ..., 25〜45


def unwrap(pos):
    d = (np.diff(pos.astype(float), axis=0) + S / 2) % S - S / 2
    return np.concatenate([pos[:1].astype(float), pos[0] + np.cumsum(d, axis=0)])


def samples(pos, method):
    """全員分の (左右の距離, 前後の距離, 符号つきの向きの差[度], 旋回の力, 加速の力) を返す"""
    X = unwrap(pos)                                    # (T+1, 2, 2)
    if method.startswith("smooth"):
        w = int(method[6:]); k = np.ones(w) / w
        X = np.stack([np.stack([np.convolve(X[:, i, j], k, mode="valid") for j in range(2)], -1) for i in range(2)], 1)
    v = np.diff(X, axis=0)                             # v[t] = X[t+1] - X[t]
    acc = v[1:] - v[:-1]                               # t=1.. : v(t) - v(t-1)
    head = v[:-1] if method == "backward" else (v[:-1] + v[1:]) / 2
    pos_t = X[1:-1]
    out = []
    for f, g in [(0, 1), (1, 0)]:
        h = head[:, f]; hg = head[:, g]
        nh = np.linalg.norm(h, axis=1); ng = np.linalg.norm(hg, axis=1)
        ok = (nh > 1e-9) & (ng > 1e-9)
        u = np.zeros_like(h); u[ok] = h[ok] / nh[ok, None]
        n = np.stack([-u[:, 1], u[:, 0]], 1)            # 左向きの法線
        r = (pos_t[:, g] - pos_t[:, f] + S / 2) % S - S / 2
        lr = -(r * n).sum(1)                           # 図4 にあわせて「右」を正にする
        fb = (r * u).sum(1)
        ang = np.degrees(np.arctan2(u[:, 0] * hg[:, 1] - u[:, 1] * hg[:, 0], (u * hg).sum(1)))
        turn = -(acc[:, f] * n).sum(1)                 # 右向きを正
        speed = (acc[:, f] * u).sum(1)
        out.append(np.stack([lr, fb, ang, turn, speed], 1)[ok])
    return np.concatenate(out)


def fmap(x, y, val, xlo, xw, ylo, yw):
    M = np.full((len(ylo), len(xlo)), np.nan)
    for i, a in enumerate(ylo):
        sy = (y >= a) & (y < a + yw)
        for j, b in enumerate(xlo):
            s = sy & (x >= b) & (x < b + xw)
            if s.sum() >= 30:
                M[i, j] = val[s].mean()
    return M


def share_lr(M):
    """地図のばらつきのうち、左右の距離（列）で説明できる割合"""
    col = np.nanmean(M, 0); row = np.nanmean(M, 1)
    vc, vr = np.nanvar(col), np.nanvar(row)
    return float(vc / (vc + vr))


def regress(Z):
    """旋回の力 = b1·(左右の距離) + b2·sin(向きの差)。標準化した係数を返す"""
    sel = (np.abs(Z[:, 0]) <= 4) & (np.abs(Z[:, 1]) <= 4) & (np.abs(Z[:, 2]) <= 45)
    x1 = Z[sel, 0]; x2 = np.sin(np.radians(Z[sel, 2])); y = Z[sel, 3]
    A = np.stack([(x1 - x1.mean()) / x1.std(), (x2 - x2.mean()) / x2.std(), np.ones_like(x1)], 1)
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(b[0]), float(b[1])


res = {}
for tag in TAGS:
    keys = [k for k in D.files if k.startswith(tag) and not k.endswith("rules")]
    for meth in METHODS:
        Z = np.concatenate([samples(D[k], meth) for k in keys])
        near = (np.abs(Z[:, 0]) <= 4) & (np.abs(Z[:, 1]) <= 4)
        Zn = Z[near]
        absang = np.abs(Zn[:, 2])
        Mt = fmap(Zn[:, 0], absang, Zn[:, 3], D_LO, 2, TH_LO, 20)      # 図4b
        Ms = fmap(absang, Zn[:, 1], Zn[:, 4], TH_LO, 20, D_LO, 2)      # 図4c（縦が前後の距離）
        Mts = fmap(Zn[:, 0], Zn[:, 2], Zn[:, 3], D_LO, 2, TH_S_LO, 20)  # 符号つき
        b_lr, b_ang = regress(Z)
        res[f"{tag}|{meth}"] = dict(turn=Mt.tolist(), speed=Ms.tolist(), turn_signed=Mts.tolist(),
                                    share_lr=share_lr(Mt), share_lr_signed=share_lr(Mts),
                                    share_fb_speed=1 - share_lr(Ms),
                                    b_lr=b_lr, b_ang=b_ang, n=int(len(Zn)))
    rules = np.mean([D[k] for k in D.files if k.startswith(tag) and k.endswith("rules")], 0)
    res[f"{tag}|rules"] = rules.tolist()
json.dump(res, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)

print("| 設定 | 計算方法 | 図4b：左右の距離で決まる割合 | 符号つきの地図での割合 | 標準化係数 左右の距離 / 向きの差 | 図4c：前後の距離で決まる割合 |")
print("|---|---|---|---|---|---|")
for tag, nm in TAGS.items():
    for meth in METHODS:
        r = res[f"{tag}|{meth}"]
        print(f"| {nm} | {meth} | {r['share_lr']:.2f} | {r['share_lr_signed']:.2f} | {r['b_lr']:+.2f} / {r['b_ang']:+.2f} | {r['share_fb_speed']:.2f} |")

# ---------------- 図 ----------------
plt.rcParams.update({"font.family": ["Yu Gothic", "MS Gothic"], "font.size": 9.5, "savefig.dpi": 150, "savefig.bbox": "tight"})
cols = [("principal_x2", "backward", "図4b の方法（向きの差は 0〜45°）"),
        ("principal_x2", "backward", "向きの差に符号を残す（−45〜45°）"),
        ("principal_x2", "central", "向きを前後の平均にする（符号つき）"),
        ("principal_x1", "backward", "α そのまま・図4b の方法")]
fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), gridspec_kw={"wspace": 0.45})
d_c = D_LO + 1
for ax, (tag, meth, title) in zip(axes, cols):
    r = res[f"{tag}|{meth}"]
    signed = "符号" in title
    M = np.array(r["turn_signed"] if signed else r["turn"], float)
    th_c = (TH_S_LO if signed else TH_LO) + 10
    vmax = np.nanmax(np.abs(M))
    im = ax.pcolormesh(d_c, th_c, M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
    ax.set_xlabel("左右の距離（＋が右）"); ax.set_ylabel("相対的な向き（度）")
    ax.set_title(title, fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.05)
fig.suptitle("2体の旋回の力（赤：右へ、青：左へ。ω 式どおり、N=2、P=40、α=0.2、30000 ステップ × 3シード）", y=1.04)
fig.savefig(sys.argv[3]); print("fig:", sys.argv[3])
