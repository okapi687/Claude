import sys, json, itertools
from multiprocessing import Pool
from cand_p1 import one
if __name__ == "__main__":
    jobs = [("literal" + ("_round" if im == "round" else ""), dict(P=P, alpha=a, seed=s, ma_rule="literal", int_mode=im))
            for im, s in itertools.product(["floor", "round"], range(5))
            for P, a in [(15, x) for x in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]] + [(1, 0.7), (2, 0.7), (5, 0.7)]]
    with Pool(20) as p:
        res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w")); print("done", len(res))
