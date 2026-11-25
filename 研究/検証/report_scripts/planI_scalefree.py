"""計画I（検証/04_計画.md）：論文1の図15（スケールフリー相関）を確かめる。引数: 出力 json
論文の定義（p.22）
  揺らぎベクトル u_i = v_i − (1/N)Σv_k（v_i は i 番目の個体の速度＝スナップショットの移動）
  C(r) = Σ u_i·u_j δ(r − r_ij) / Σ δ(r − r_ij)。規格化は論文に書かれていないので3通り比べる
    self：自分自身との積の平均 <u_i·u_i> で割る（ノートブック・Cavagna et al. 2010 と同じ）
    first：一番近い距離ビン（r<1.5）の C で割る（図15 で r→0 の C がほぼ 1 になる読み方）
    raw：割らない（速度は格子の単位のまま）
  相関長 ξ は C(r)=0 になる距離（最初に正から負に変わる所を線形補間）
  C を r/ξ に対して描くと1本の直線（図15d：r/ξ=0 で約 1.05、1.3 で約 −0.2、傾き約 −1。実際のカニ 図15c は約 −0.9）
計算はノートブック MAM_valid.ipynb の「スケールフリー相関」の関数と同じ（距離ビン幅 1、ペアが 10 未満のビンは除く、
直線は r/ξ=0〜1.2 の点で引く）
条件：図15 と同じ L=4、R(Nf)=2、R(M)=2（P と α は論文に書かれていないので振る）
  設定4通り：v1 の既定（α そのまま・floor）、α 2倍・round、α そのまま・round＋移動で速度一致（E2）、α 2倍・round＋E2
  P=10 / 15 / 20、α=0.3 / 0.5 / 0.7、3シード。500 ステップ回してから 600 ステップ記録し、3 ステップごとのスナップショットで解析
  速度の取り方：1ステップの移動（論文の「スナップショットの移動」）と、3・5・10・20 ステップの移動を割ったもの
"""
import os, sys, json, itertools
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 研究/
from model.MAM_basic_v1 import SoldierCrabSwarm_v1

S = 50
TAUS = [1, 3, 5, 10, 20]
CONFIGS = {"v1": dict(), "as2_round": dict(alpha_scale=2, int_mode="round"),
           "as1_round_E2": dict(alpha_scale=1, int_mode="round", vm_source="disp"),
           "as2_round_E2": dict(alpha_scale=2, int_mode="round", vm_source="disp")}
XB = np.arange(0, 1.65, 0.05)          # r/ξ のビン（図用）


def corr(pos, vel, bin_width=1.0, min_pairs=10):
    N = len(pos)
    u = vel - vel.mean(0)
    c0 = np.mean((u ** 2).sum(1))
    if c0 == 0:
        return None
    diff = pos[:, None, :] - pos[None, :, :]
    diff = (diff + S / 2) % S - S / 2
    dist = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(N, 1)
    r, c = dist[iu], (u @ u.T)[iu]
    edges = np.arange(0, S / 2 + bin_width, bin_width)
    idx = np.digitize(r, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    sums = np.bincount(idx[ok], weights=c[ok], minlength=len(edges) - 1)
    cnts = np.bincount(idx[ok], minlength=len(edges) - 1)
    good = cnts >= min_pairs
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers[good], sums[good] / cnts[good], c0


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
    H = m.run(600, record_history=True)
    out = dict(tag=tag, P=P, alpha=a, seed=s, fits={})
    for tau in TAUS:
      for norm in ("self", "first", "raw"):
          xs, ys, xis = [], [], []
          for t in range(0, H.shape[0] - tau, 3):
              d = (H[t + tau] - H[t]).astype(float)
              d = (d + S / 2) % S - S / 2
              res = corr(H[t].astype(float), d / tau)
              if res is None:
                  continue
              r, C, c0 = res
              if norm == "self":
                  C = C / c0
              elif norm == "first":
                  if r[0] > 1.5 or C[0] <= 0:
                      continue
                  C = C / C[0]
              xi = xi_of(r, C)
              if np.isnan(xi):
                  continue
              xs.append(r / xi); ys.append(C); xis.append(xi)
          key = f"{norm}|{tau}"
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
                                  xi=float(np.mean(xis)), prof=prof)
    return out


if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, [20, 15, 10], [0.3, 0.5, 0.7], range(3)))
    if len(sys.argv) > 2:
        jobs = jobs[:int(sys.argv[2])]
    with Pool(min(14, os.cpu_count() or 1)) as p:
        res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
