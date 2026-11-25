"""計画E（検証/04_計画.md）：論文1で θ がばらつく仕組みを、式(4)(6)(8) の読み方から探す。引数: 出力 json [動作確認用の件数]
E1 choose_by="site"  ：同率の最大人気サイト（式(4)）と自由移動（式(6)）の行き先を、サイト単位で一様に選ぶ
E2 vm_source="disp"  ：速度一致（式(8)）で、近傍の移動ベクトルそのもの（長さ込み）を平均する
α の読み方は「そのまま・floor（v1 の既定）」と「2倍・round（計画2で一番近い）」の2通り
条件と測り方は計画2と同じ（P=10/15/20、α=0.2〜0.7、5シード、900 ステップで t=300〜900 の平均。cand_p1.one）
"""
import os, sys, json, itertools
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 研究/
from cand_p1 import one

READING = {"as1_floor": dict(alpha_scale=1, int_mode="floor"), "as2_round": dict(alpha_scale=2, int_mode="round")}
VARIANT = {"base": dict(), "E1": dict(choose_by="site"), "E2": dict(vm_source="disp"),
           "E1E2": dict(choose_by="site", vm_source="disp")}
CONFIGS = {f"{r}|{v}": {**rk, **vk} for (r, rk), (v, vk) in itertools.product(READING.items(), VARIANT.items())}

if __name__ == "__main__":
    jobs = [(tag, dict(P=P, alpha=a, seed=s, **kw))
            for (tag, kw), P, a, s in itertools.product(
                CONFIGS.items(), [10, 15, 20], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], range(5))]
    if len(sys.argv) > 2:
        jobs = jobs[:int(sys.argv[2])]
    with Pool(min(14, os.cpu_count() or 1)) as p:
        res = p.map(one, jobs, chunksize=1)
    for r in res:                       # 時系列は大きいので落とす
        for k in [k for k in r if k.endswith("_ts")]:
            del r[k]
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
