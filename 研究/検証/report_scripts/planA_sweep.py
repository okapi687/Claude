"""計画A（検証/04_計画.md）：論文1で α=0.3 → 0.4 の急な落ち方を調べる。引数: 出力 json [動作確認用の件数]
設定：計画2で一番近かった「α 2倍 × round × sequential」（alpha_scale=2, int_mode="round", ma_rule="sequential"）
条件：P=10/15/20、論文の α=0.25〜0.45（0.025 刻み）、5シード、3000 ステップ
記録：毎ステップの整列度 pL（|平均変位| / L）と基本ベクトルのそろい φ（|θ の単位ベクトルの平均|）、
      10 ステップごとの密度（Ls=10 の窓をずらした最大個体数 / N）、ルールの割合
"""
import os, sys, json, itertools
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 研究/
from model.MAM_basic_v1 import SoldierCrabSwarm_v1, polarization1, density_sliding

T = 3000
ALPHAS = [round(0.25 + 0.025 * i, 3) for i in range(9)]


def one(job):
    P, a, s = job
    m = SoldierCrabSwarm_v1(model="paper1", P=P, alpha=a, seed=s,
                            alpha_scale=2, int_mode="round", ma_rule="sequential")
    pL, phi, dens, rules = [], [], [], np.zeros(4)
    for t in range(T):
        th = m.theta.copy()
        m.step()
        pL.append(round(float(polarization1(m.last_displacement, m.L)), 4))
        phi.append(round(float(abs(np.mean(np.exp(1j * th)))), 4))
        if t % 10 == 9:
            dens.append(round(float(density_sliding(m.position, m.S, 10)), 4))
        if t >= 300:
            rules += np.array(m.last_rule_counts) / m.N
    return dict(P=P, alpha=a, seed=s, pL=pL, phi=phi, dens=dens,
                rules=(rules / (T - 300)).round(4).tolist())


if __name__ == "__main__":
    jobs = list(itertools.product([20, 15, 10], ALPHAS, range(5)))
    if len(sys.argv) > 2:
        jobs = jobs[:int(sys.argv[2])]
    with Pool(min(14, os.cpu_count() or 1)) as p:
        res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
