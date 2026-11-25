"""計画C（検証/04_計画.md）：論文2の2体のシミュレーション（図4・図S2b）を回して、位置を保存する。
引数: 出力 npz
条件：本文 p.6 と同じ N=2、P=40、L=6、ω=5、α=0.2、50×50、30000 ステップ。3シード
設定：ω 式どおり・α そのまま／ω 式どおり・α 2倍／ω 移動側（v1 の既定）・α そのまま。人気度は1個体1票（v1 の既定）
"""
import os, sys, itertools
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 研究/
from model.MAM_basic_v1 import SoldierCrabSwarm_v1

T = 30000
CONFIGS = {"principal_x1": dict(omega_on="principal", alpha_scale=1),
           "principal_x2": dict(omega_on="principal", alpha_scale=2),
           "velocity_x1": dict(omega_on="velocity", alpha_scale=1)}


def one(job):
    tag, s = job
    m = SoldierCrabSwarm_v1(model="paper2", N=2, P=40, alpha=0.2, seed=s, **CONFIGS[tag])
    pos = np.zeros((T + 1, 2, 2), np.int16); pos[0] = m.position
    rules = np.zeros(4)
    for t in range(T):
        m.step(); pos[t + 1] = m.position; rules += m.last_rule_counts
    return tag, s, pos, rules / (2 * T)


if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, range(3)))
    with Pool(min(9, os.cpu_count() or 1)) as p:
        res = p.map(one, jobs)
    np.savez_compressed(sys.argv[1], **{f"{t}_s{s}": pos for t, s, pos, _ in res},
                        **{f"{t}_s{s}_rules": r for t, s, _, r in res})
    print("done", len(res))
