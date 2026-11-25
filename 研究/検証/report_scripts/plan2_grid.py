"""計画2（レポート7章）：論文1で alpha_scale × int_mode × ma_rule の8通りを比べる。引数: 出力 json
条件：P=10/15/20、α=0.2〜0.7、5シード。測り方は cand_p1.one と同じ（900ステップ、t=300〜900 の平均）
目標：論文の α=0.3 / 0.4 / 0.5 で整列度 0.50 / 0.27 / 0.12 の ±0.1
"""
import os, sys, json, itertools
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 研究/
from cand_p1 import one

CONFIGS = {
    f"as{a_s}_{im}_{mr}": dict(alpha_scale=a_s, int_mode=im, ma_rule=mr)
    for a_s, im, mr in itertools.product([1, 2], ["floor", "round"], ["sequential", "literal"])
}

if __name__ == "__main__":
    jobs = [(tag, dict(P=P, alpha=a, seed=s, **kw))
            for (tag, kw), P, a, s in itertools.product(
                CONFIGS.items(), [10, 15, 20], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], range(5))]
    if len(sys.argv) > 2:  # 動作確認用：先頭 n 件だけ
        jobs = jobs[:int(sys.argv[2])]
    with Pool(14) as p:
        res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
