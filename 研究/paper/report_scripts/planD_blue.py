"""計画D（検証/04_計画.md）：論文2の青（P=10）で α を振り、ルールの割合・整列度・密度を見る。引数: 出力 json
条件：P=10（と比較用に P=8・12）、論文の α=0.025〜0.3、alpha_scale 1 / 2、ω 式どおり、round、1個体1票、5シード、1500 ステップ
測り方：t=1000〜1500 の平均（図3・図S2 と同じ）
論文の青（P=10, α=0.1）：整列度 0.2、密度 0.65、相互予期 / 追従 / 自由移動 = 63 / 5 / 32%
"""
import os, sys, json, itertools
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 研究/
from model.MAM_basic_v1 import SoldierCrabSwarm_v1, polarization2, density_paper2

T, T0 = 1500, 1000
ALPHAS = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3]


def one(job):
    P, sc, a, s = job
    m = SoldierCrabSwarm_v1(model="paper2", P=P, alpha=a, alpha_scale=sc, omega_on="principal", seed=s)
    pol, dens, rules, ov = [], [], np.zeros(4), []
    for t in range(T):
        m.step()
        if t >= T0:
            pol.append(polarization2(m.last_displacement)); dens.append(density_paper2(m.position, m.S))
            rules += np.array(m.last_rule_counts) / m.N
    return dict(P=P, scale=sc, alpha=a, seed=s, pol=float(np.mean(pol)), dens=float(np.mean(dens)),
                rules=(rules / (T - T0)).round(4).tolist())


if __name__ == "__main__":
    jobs = list(itertools.product([10], [1, 2], ALPHAS, range(5)))
    jobs += list(itertools.product([8, 12], [1, 2], [0.05, 0.1, 0.15], range(5)))
    with Pool(min(14, os.cpu_count() or 1)) as p:
        res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
