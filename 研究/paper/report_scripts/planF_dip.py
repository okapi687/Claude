"""計画F（検証/04_計画.md）：論文1の密度の谷（P=8〜12）を、シードを増やして確かめる。引数: 出力 json
密度は計画B で図11 に合った「区画 2×2（Ls=25）」で測る。比較用に今の実装（窓をずらす Ls=10）と回転半径も測る
条件：v1 の既定（α そのまま・floor）と α 2倍・round、α=0.3〜0.6、P=4〜20（1刻み）、10シード、900 ステップ（t=300〜900 の 10 ステップごと）
"""
import os, sys, json, itertools
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 研究/
from model.MAM_basic_v1 import SoldierCrabSwarm_v1, density_sliding

S, N = 50, 100
CONFIGS = {"v1": dict(), "as2_round": dict(alpha_scale=2, int_mode="round")}


def grid(pos, k):
    e = np.floor(np.arange(k + 1) * S / k).astype(int)
    ix = np.searchsorted(e, pos[:, 0], side="right") - 1
    iy = np.searchsorted(e, pos[:, 1], side="right") - 1
    h = np.zeros((k, k), int); np.add.at(h, (ix, iy), 1)
    return h.max() / N


def rg(pos):
    th = pos * 2 * np.pi / S
    c = np.angle(np.exp(1j * th).mean(0)) * S / (2 * np.pi)
    d = (pos - c + S / 2) % S - S / 2
    return float(np.sqrt((d ** 2).sum(1).mean()))


def one(job):
    tag, P, a, s = job
    m = SoldierCrabSwarm_v1(model="paper1", P=P, alpha=a, seed=s, **CONFIGS[tag])
    g2, sl, r = [], [], []
    for t in range(900):
        m.step()
        if t >= 300 and t % 10 == 9:
            g2.append(grid(m.position, 2)); sl.append(density_sliding(m.position, S, 10)); r.append(rg(m.position))
    return dict(tag=tag, P=P, alpha=a, seed=s, grid2=float(np.mean(g2)), slide10=float(np.mean(sl)), Rg=float(np.mean(r)))


if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, range(20, 3, -1), [0.3, 0.4, 0.5, 0.6], range(10)))
    with Pool(8) as p:
        res = p.map(one, jobs, chunksize=2)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
