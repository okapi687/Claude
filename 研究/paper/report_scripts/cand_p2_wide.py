import sys, json, itertools
from multiprocessing import Pool
from cand_p2 import one
if __name__ == "__main__":
    cf = {"v1": {}, "v1_principal": dict(omega_on="principal"), "v1_floor": dict(int_mode="floor"),
          "v1_principal_floor": dict(omega_on="principal", int_mode="floor")}
    jobs = [(t, dict(P=40, alpha=a, seed=s, **kw)) for (t, kw), a, s in itertools.product(cf.items(), [0.4, 0.6, 0.8, 1.0], range(3))]
    jobs += [(t, dict(P=10, alpha=0.2, seed=s, **kw)) for (t, kw), s in itertools.product(cf.items(), range(3))]
    jobs += [(t, dict(P=40, alpha=a, seed=s, **kw)) for (t, kw), a, s in itertools.product(list(cf.items())[2:], [0.2, 0.5], range(3))]
    jobs += [(t, dict(P=10, alpha=0.1, seed=s, **kw)) for (t, kw), s in itertools.product(list(cf.items())[2:], range(3))]
    with Pool(20) as p:
        res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w")); print("done", len(res))
