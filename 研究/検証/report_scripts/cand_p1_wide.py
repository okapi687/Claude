import sys, json, itertools
from multiprocessing import Pool
from cand_p1 import one
if __name__ == "__main__":
    jobs = [(tag, dict(P=P, alpha=a, seed=s, **kw)) for (tag, kw), P, a, s in itertools.product(
        [("v1", {}), ("v1_round", dict(int_mode="round"))], [10, 15, 20], [0.4, 0.5, 0.6, 0.8, 0.9, 1.0], range(5))]
    with Pool(10) as p:
        res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w")); print("done", len(res))
