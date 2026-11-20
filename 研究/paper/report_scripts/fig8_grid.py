"""論文1 図8 の格子（P × α）を、α をそのまま / 2倍 で回す。引数: 出力 json"""
import sys, json, itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, r"C:\Users\PC_User\ClaudeCode\研究")
from model.MAM_basic_v1 import SoldierCrabSwarm_v1, polarization1, density_sliding

def one(a):
    scale, im, P, al, s = a
    m = SoldierCrabSwarm_v1(model="paper1", P=P, alpha=al * scale, int_mode=im, seed=s)
    hist, pol, dens = [], [], []
    for t in range(600):
        m.step()
        if t >= 300:
            pol.append(polarization1(m.last_displacement, m.L)); dens.append(density_sliding(m.position, m.S, 10))
        if t >= 590 and s == 0:
            hist.append(m.position.tolist())
    return dict(scale=scale, im=im, P=P, alpha=al, seed=s, pol=float(np.mean(pol)), dens=float(np.mean(dens)), hist=hist)

if __name__ == "__main__":
    jobs = list(itertools.product([1, 2], ["floor", "round"], [1, 2, 5, 10, 20], [0.3, 0.5, 0.7, 0.9], range(3)))
    with Pool(20) as p:
        res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w")); print("done", len(res))
