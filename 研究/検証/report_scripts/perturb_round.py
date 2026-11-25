import sys, json, itertools
from multiprocessing import Pool
from perturb import one
if __name__ == "__main__":
    jobs = [("round_noise", dict(P=15, alpha=a, ext_noise=e, seed=s, int_mode="round"), 900, 300)
            for a, e, s in itertools.product([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [0.0, 0.3, 0.5, 1.0], range(5))]
    with Pool(20) as p:
        res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w")); print("done", len(res))
