"""計画K：論文1の図15（スケールフリー相関）の傾きを、計画I で試していない読み方で確かめる。引数: 出力 json
論文の文面（p.22）
  「u_i = v_i − (1/N)Σv_k, where v_i is the velocity of the i-th individual and N is the number of
    individuals in a swarm」
  「For each snapshot of the transitions ... we obtained the correlation length」
  「the same slope as that of our model and that of a starling flock (Cavagna et al. 2010)」
計画I では、速度＝移動（1〜20 ステップ）、トーラス全体の100個体で計算した。ここでは次の2つを足す。
  速度 v_i の取り方（vel）
    disp1：1ステップの移動（計画I と同じ）
    disp5：5ステップの移動 / 5（計画I と同じ）
    theta：基本ベクトル L·(cosθ, sinθ)（モデルの個体が「向かおうとしている速度」。速度一致でそろえる量）
  計算する範囲（dom）
    all：トーラス全体の N 個体（計画I と同じ）
    clus：一番大きい群れだけ（距離 2 以下でつながる個体の集まり。R(Nf)=2 と同じ距離）。
          文面の「N is the number of individuals in a swarm」を、1つの群れの中の個体数と読む。
          平均速度もその群れの中で引く。群れが 20 個体未満のスナップショットは使わない
  規格化は Cavagna et al. 2010 と同じ「自分との積の平均で割る」だけ（計画I で一番まともだったもの）
  ほかは計画I と同じ（距離ビン幅 1、ペア 10 未満のビンは除く、ξ は C が最初に 0 を切る距離、r/ξ=0〜1.2 で直線）
  スケールフリーなら ξ が群れの大きさに比例する（Cavagna et al. 2010）。clus では群れの大きさ（重心からの平均距離）も記録する
条件：L=4、R(Nf)=2、R(M)=2。設定3通り（v1、α 2倍・round、α そのまま・round＋E2）× P=10/15/20 × α=0.3/0.5 × 3シード
  500 ステップ回してから 600 ステップ、3 ステップごとに解析
"""
import os, sys, json, itertools
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 研究/
from model.MAM_basic_v1 import SoldierCrabSwarm_v1

S = 50
CONFIGS = {"v1": dict(), "as2_round": dict(alpha_scale=2, int_mode="round"),
           "as1_round_E2": dict(alpha_scale=1, int_mode="round", vm_source="disp")}
VELS = ["disp1", "disp5", "theta"]
DOMS = ["all", "clus"]
XB = np.arange(0, 1.65, 0.05)


def tor(d):
    return (d + S / 2) % S - S / 2


def largest_cluster(pos, link=2.0):
    d = tor(pos[:, None, :] - pos[None, :, :])
    adj = (np.sqrt((d ** 2).sum(-1)) <= link)
    N = len(pos)
    lab = -np.ones(N, int)
    best = None
    for s in range(N):
        if lab[s] >= 0:
            continue
        stack, mem = [s], [s]
        lab[s] = s
        while stack:
            k = stack.pop()
            for j in np.nonzero(adj[k] & (lab < 0))[0]:
                lab[j] = s; stack.append(j); mem.append(j)
        if best is None or len(mem) > len(best):
            best = mem
    return np.array(best)


def corr(pos, vel, bin_width=1.0, min_pairs=10):
    N = len(pos)
    u = vel - vel.mean(0)
    c0 = np.mean((u ** 2).sum(1))
    if c0 == 0:
        return None
    dist = np.sqrt((tor(pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(N, 1)
    r, c = dist[iu], (u @ u.T)[iu]
    edges = np.arange(0, S / 2 + bin_width, bin_width)
    idx = np.digitize(r, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    sums = np.bincount(idx[ok], weights=c[ok], minlength=len(edges) - 1)
    cnts = np.bincount(idx[ok], minlength=len(edges) - 1)
    good = cnts >= min_pairs
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers[good], sums[good] / cnts[good] / c0


def xi_of(r, C):
    if len(C) < 2 or C[0] <= 0:
        return np.nan
    for i in range(len(C) - 1):
        if C[i] > 0 and C[i + 1] <= 0:
            return r[i] + (r[i + 1] - r[i]) * C[i] / (C[i] - C[i + 1])
    return np.nan


def one(job):
    tag, P, a, s = job
    m = SoldierCrabSwarm_v1(model="paper1", P=P, alpha=a, seed=s, R_M=2, **CONFIGS[tag])
    m.run(500)
    T = 600
    pos = np.zeros((T + 1, m.N, 2)); th = np.zeros((T + 1, m.N))
    pos[0] = m.position; th[0] = m.theta
    for t in range(1, T + 1):
        m.step(); pos[t] = m.position; th[t] = m.theta
    out = dict(tag=tag, P=P, alpha=a, seed=s, fits={})
    acc = {(v, d): ([], [], [], []) for v in VELS for d in DOMS}   # x, y, xi, size
    for t in range(0, T - 5, 3):
        # 位置は t、移動は t→t+1（または t→t+5）。theta は移動のあとの向き（t+1）
        vel = {"disp1": tor(pos[t + 1] - pos[t]), "disp5": tor(pos[t + 5] - pos[t]) / 5,
               "theta": m.L * np.stack([np.cos(th[t + 1]), np.sin(th[t + 1])], 1)}
        cl = largest_cluster(pos[t])
        for d in DOMS:
            if d == "clus" and len(cl) < 20:
                continue
            sel = slice(None) if d == "all" else cl
            p = pos[t][sel]
            if d == "clus":
                p = p[0] + tor(p - p[0])
                size = float(np.mean(np.sqrt(((p - p.mean(0)) ** 2).sum(1))))
            else:
                size = np.nan
            for v in VELS:
                res = corr(p, vel[v][sel])
                if res is None:
                    continue
                r, C = res
                xi = xi_of(r, C)
                if np.isnan(xi):
                    continue
                X = acc[(v, d)]
                X[0].append(r / xi); X[1].append(C); X[2].append(xi); X[3].append(size)
    for (v, d), (xs, ys, xis, sz) in acc.items():
        key = f"{v}|{d}"
        if len(xs) < 3:
            out["fits"][key] = None
            continue
        x = np.concatenate(xs); y = np.concatenate(ys)
        sel = x <= 1.2
        slope, icpt = np.polyfit(x[sel], y[sel], 1)
        pred = slope * x[sel] + icpt
        r2 = 1 - np.sum((y[sel] - pred) ** 2) / np.sum((y[sel] - y[sel].mean()) ** 2)
        bi = np.digitize(x, XB) - 1
        prof = [float(y[bi == k].mean()) if np.sum(bi == k) >= 5 else None for k in range(len(XB) - 1)]
        out["fits"][key] = dict(slope=float(slope), icpt=float(icpt), r2=float(r2), n=len(xs),
                                xi=float(np.mean(xis)), prof=prof,
                                xi_list=[float(q) for q in xis], size_list=[float(q) for q in sz])
    return out


if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, [20, 15, 10], [0.3, 0.5], range(3)))
    if len(sys.argv) > 2:
        jobs = jobs[:int(sys.argv[2])]
    with Pool(min(14, os.cpu_count() or 1)) as p:
        res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
