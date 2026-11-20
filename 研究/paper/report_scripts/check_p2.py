"""論文2（MAM_basic_Murakami）の再現チェック"""
import sys, json, itertools
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, r"C:\Users\PC_User\ClaudeCode\研究")
from model.MAM_basic_Murakami import simulate

T = 1500


def one(args):
    tag, kw = args
    dens, pol, rules = simulate(T, **kw)
    return dict(tag=tag, kw=kw,
                dens=[float(x) for x in dens],
                pol=[None if np.isnan(x) else float(x) for x in pol],
                rules=rules.tolist())


if __name__ == "__main__":
    jobs = []
    settings = {"blue": dict(P=10, alpha=0.1), "red": dict(P=40, alpha=0.2), "green": dict(P=40, alpha=0.5)}
    # 図3b,c と 図S2：既定（round）と floor
    for (name, kw), s, im in itertools.product(settings.items(), range(6), ["round", "floor"]):
        jobs.append((f"fig3_{name}_{im}", dict(kw, seed=s, int_mode=im)))
    # 図S3：ω = 0..10
    for om, s in itertools.product(range(11), range(5)):
        jobs.append((f"S3_w{om}", dict(P=40, alpha=0.2, omega=float(om), seed=s)))
    # 図3a：P×α の格子
    for P, a, s in itertools.product([10, 20, 30, 40], [0.1, 0.2, 0.3, 0.4, 0.5], range(3)):
        jobs.append((f"3a_P{P}_a{a}", dict(P=P, alpha=a, seed=s)))
    with Pool(22) as pool:
        res = pool.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
