"""計画B（検証/04_計画.md）：論文1の密度を、論文の測り方（区画に分ける）で測り直すための位置を保存する。
引数: 出力 json [動作確認用の件数]
条件：図10右・図11 と同じく P=1〜30、α=0.3〜0.6（L=4, R(Nf)=2, R(M)=3）、3シード、900 ステップ
設定：v1 の既定（α そのまま・floor・sequential）と、計画2で一番近かった α 2倍・round・sequential
記録：t=300〜900 の 10 ステップごとの全員の位置、毎ステップの整列度 pL の平均
"""
import os, sys, json, itertools
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 研究/
from model.MAM_basic_v1 import SoldierCrabSwarm_v1, polarization1

T, T0 = 900, 300
CONFIGS = {"v1": dict(), "as2_round": dict(alpha_scale=2, int_mode="round")}
PS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
ALPHAS = [0.3, 0.4, 0.5, 0.6]


def one(job):
    tag, P, a, s = job
    m = SoldierCrabSwarm_v1(model="paper1", P=P, alpha=a, seed=s, **CONFIGS[tag])
    pos, pL = [], []
    for t in range(T):
        m.step()
        if t >= T0:
            pL.append(polarization1(m.last_displacement, m.L))
            if t % 10 == 9:
                pos.append(m.position.tolist())
    return dict(tag=tag, P=P, alpha=a, seed=s, pL=float(np.mean(pL)), pos=pos)


if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, sorted(PS, reverse=True), ALPHAS, range(3)))
    if len(sys.argv) > 2:
        jobs = jobs[:int(sys.argv[2])]
    with Pool(min(14, os.cpu_count() or 1)) as p:
        res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
