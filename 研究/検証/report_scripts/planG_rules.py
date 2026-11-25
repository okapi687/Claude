"""計画G（検証/04_計画.md）：論文1で相互予期が多すぎる（群れが詰まりすぎる）かを調べる。引数: 出力 json
P を変えたときのルールの割合（相互予期 / 追従 / 自由移動 / 留まる）と、密度（区画 Ls=10・Ls=25）、整列度を設定ごとに測る
設定：v1（α そのまま・floor）、α 2倍・round、α そのまま・round＋E2（速度一致で移動を平均）、文字どおりの競合（ma_rule="literal"）
条件：P=1〜30、論文の α=0.5、5シード、900 ステップ（t=300〜900 の平均、密度は 10 ステップごと）
"""
import os, sys, json, itertools
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 研究/
from model.MAM_basic_v1 import SoldierCrabSwarm_v1, polarization1

S, N = 50, 100
CONFIGS = {"v1": dict(), "as2_round": dict(alpha_scale=2, int_mode="round"),
           "as1_round_E2": dict(alpha_scale=1, int_mode="round", vm_source="disp"),
           "literal": dict(ma_rule="literal")}


def grid(pos, k):
    e = np.floor(np.arange(k + 1) * S / k).astype(int)
    ix = np.searchsorted(e, pos[:, 0], side="right") - 1
    iy = np.searchsorted(e, pos[:, 1], side="right") - 1
    h = np.zeros((k, k), int); np.add.at(h, (ix, iy), 1)
    return h.max() / N


def one(job):
    tag, P, s = job
    m = SoldierCrabSwarm_v1(model="paper1", P=P, alpha=0.5, seed=s, **CONFIGS[tag])
    rules, pl, g5, g2 = np.zeros(4), [], [], []
    for t in range(900):
        m.step()
        if t >= 300:
            rules += np.array(m.last_rule_counts) / N
            pl.append(polarization1(m.last_displacement, m.L))
            if t % 10 == 9:
                g5.append(grid(m.position, 5)); g2.append(grid(m.position, 2))
    return dict(tag=tag, P=P, seed=s, rules=(rules / 600).tolist(), pL=float(np.mean(pl)),
                dens10=float(np.mean(g5)), dens25=float(np.mean(g2)))


if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, [30, 25, 20, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1], range(5)))
    with Pool(14) as p:
        res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
