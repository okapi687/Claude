import sys, json, itertools
from multiprocessing import Pool
from alpha_dyn import one
if __name__ == "__main__":
    jobs = []
    for (tag, kw), a, s in itertools.product({"p1_base": {}, "p1_count_all": dict(count_all=True)}.items(),
                                             [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], range(3, 13)):
        jobs.append((tag, dict(model="paper1", P=15, alpha=a, seed=s, **kw), 900, 300))
    for (tag, kw), s in itertools.product({"p2_principal_long": dict(omega_on="principal"), "p2_velocity_long": {}}.items(), range(5)):
        jobs.append((tag, dict(model="paper2", P=40, alpha=0.2, seed=s, **kw), 4000, 3500))
    with Pool(20) as pool:
        res = pool.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w")); print("done", len(res))
