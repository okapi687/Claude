"""計画E2（検証/04_計画.md）：速度一致で「移動の単位ベクトル」を平均する読み方（vm_source="unit_disp"）を round で試す。引数: 出力 json
式(8) の「近傍の平均の速度の向き」の、もう1つの読み方。4.3 節では floor で試して効かなかったので、round で試し直す
α の読み方は「そのまま・round」と「2倍・round」。条件と測り方は計画E と同じ（cand_p1.one、P=10/15/20、α=0.2〜0.7、5シード）
"""
import os, sys, json, itertools
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 研究/
from cand_p1 import one

CONFIGS = {"as1_round|E2u": dict(alpha_scale=1, int_mode="round", vm_source="unit_disp"),
           "as2_round|E2u": dict(alpha_scale=2, int_mode="round", vm_source="unit_disp")}

if __name__ == "__main__":
    jobs = [(tag, dict(P=P, alpha=a, seed=s, **kw))
            for (tag, kw), P, a, s in itertools.product(CONFIGS.items(), [10, 15, 20], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], range(5))]
    with Pool(12) as p:
        res = p.map(one, jobs, chunksize=1)
    for r in res:
        for k in [k for k in r if k.endswith("_ts")]:
            del r[k]
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
