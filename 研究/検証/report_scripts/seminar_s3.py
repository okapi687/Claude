"""図S3（ω を 0〜10）を alpha_scale 1 / 2、ω の掛け方2通りで。引数: 出力 json"""
import sys, json, itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, r"C:\Users\PC_User\ClaudeCode\研究")
from model.MAM_basic_v1 import SoldierCrabSwarm_v1, polarization2, density_paper2

def one(a):
    on, sc, w, s = a
    m = SoldierCrabSwarm_v1(model="paper2", P=40, alpha=0.2, alpha_scale=sc, omega=float(w), omega_on=on, seed=s)
    p, d = [], []
    for t in range(1500):
        m.step()
        if t >= 1400:
            p.append(polarization2(m.last_displacement)); d.append(density_paper2(m.position, m.S))
    return dict(on=on, scale=sc, omega=w, seed=s, pol=float(np.mean(p)), dens=float(np.mean(d)))

if __name__ == "__main__":
    jobs = list(itertools.product(["principal", "velocity"], [1, 2], range(11), range(3)))
    with Pool(22) as pool:
        res = pool.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w")); print("done", len(res))
