"""計画Eの追加：E2（移動ベクトルで速度一致）を「α そのまま・round」でも回す（floor の偏りが θ に入らない条件）。引数: 出力 json
条件と測り方は planE_sweep.py と同じ
"""
import os, sys, json, itertools
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 研究/
from cand_p1 import one

CONFIGS = {"as1_round|base": dict(alpha_scale=1, int_mode="round"),
           "as1_round|E2": dict(alpha_scale=1, int_mode="round", vm_source="disp")}

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
