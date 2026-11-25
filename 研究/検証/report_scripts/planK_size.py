"""計画K（その2）：スケールフリーかどうか＝相関長 ξ が群れの大きさに比例するか（Cavagna et al. 2010）を確かめる。引数: 出力 json
論文1 p.21〜22「a correlated domain in a bird flock is scaled to flock size」「the correlation function exhibited
linear decay independent of swarm size」。図15 は1つの N しか描いていないので、N を変えて ξ と群れの大きさを比べる。
密度をそろえるため、S を √N に比例させる（N=100 で S=50 が基準）。
  N=50/100/200/400（S=35/50/71/100）、設定2通り（α 2倍・round、α そのまま・round＋E2）、P=15、α=0.3、3シード
  ξ と群れの大きさ（一番大きい群れの、重心からの平均距離）は planK_scalefree2.py と同じ計算（範囲は clus）
"""
import os, sys, json, itertools
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 研究/
import planK_scalefree2 as K
from model.MAM_basic_v1 import SoldierCrabSwarm_v1

SIZES = {50: 35, 100: 50, 200: 71, 400: 100}
CONFIGS = {"as2_round": K.CONFIGS["as2_round"], "as1_round_E2": K.CONFIGS["as1_round_E2"]}


def one(job):
    tag, N, s = job
    S = SIZES[N]
    K.S = S   # 距離の補正に使うトーラスの大きさ
    m = SoldierCrabSwarm_v1(model="paper1", N=N, S=S, P=15, alpha=0.3, seed=s, R_M=2, **CONFIGS[tag])
    m.run(500)
    T = 600
    pos = np.zeros((T + 1, N, 2)); pos[0] = m.position
    for t in range(1, T + 1):
        m.step(); pos[t] = m.position
    out = dict(tag=tag, N=N, S=S, seed=s, rows=[])
    for t in range(0, T - 5, 6):
        cl = K.largest_cluster(pos[t])
        if len(cl) < 20:
            continue
        p = pos[t][cl]; p = p[0] + K.tor(p - p[0])
        size = float(np.mean(np.sqrt(((p - p.mean(0)) ** 2).sum(1))))
        row = dict(n=len(cl), size=size)
        for v, vel in (("disp1", K.tor(pos[t + 1] - pos[t])), ("disp5", K.tor(pos[t + 5] - pos[t]) / 5)):
            res = K.corr(p, vel[cl])
            xi = np.nan if res is None else K.xi_of(*res)
            row[v] = None if np.isnan(xi) else float(xi)
        out["rows"].append(row)
    return out


if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, SIZES, range(3)))
    with Pool(min(14, os.cpu_count() or 1)) as p:
        res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
