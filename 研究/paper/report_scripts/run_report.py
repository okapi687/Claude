"""レポート用：v1 で論文2（ωの2通りの読み方）と論文1（追従の連鎖あり/なし）を回す"""
import sys, json, itertools, numpy as np
from multiprocessing import Pool
sys.path.insert(0, r"C:\Users\PC_User\ClaudeCode\研究")
from model.MAM_basic_v1 import simulate

def one(args):
    tag, kw, T = args
    pol, dens, rules = simulate(T, **kw)
    return dict(tag=tag, kw=kw, pol=pol.tolist(), dens=dens.tolist(), rules=rules.mean(0).tolist(),
                rules_late=rules[T//2:].mean(0).tolist())

if __name__ == "__main__":
    jobs = []
    st = {"blue": dict(P=10, alpha=0.1), "red": dict(P=40, alpha=0.2), "green": dict(P=40, alpha=0.5)}
    for (n, kw), s, om in itertools.product(st.items(), range(6), ["velocity", "principal"]):
        jobs.append((f"p2_{n}_{om}", dict(model="paper2", seed=s, omega_on=om, **kw), 1500))
    for w, s, om in itertools.product(range(11), range(5), ["velocity", "principal"]):
        jobs.append((f"p2S3_{om}_{w}", dict(model="paper2", P=40, alpha=0.2, omega=float(w), omega_on=om, seed=s), 1500))
    for P, a, s in itertools.product([10, 20, 30, 40], [0.1, 0.2, 0.3, 0.4, 0.5], range(3)):
        jobs.append((f"p2grid_{P}_{a}", dict(model="paper2", P=P, alpha=a, seed=s), 1500))
    for P, s, ch in itertools.product([1, 2, 3, 4, 5, 6, 8, 10, 15, 20], range(3), [False, True]):
        jobs.append((f"p1chain_{ch}_{P}", dict(model="paper1", P=P, alpha=0.7, seed=s, follow_chain=ch), 600))
    with Pool(20) as pool:
        res = pool.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
