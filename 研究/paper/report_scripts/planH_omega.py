"""計画H（検証/04_計画.md）：論文2の図S3（ω を変えたときの整列度）の ω≤1 を、ω の2つの読み方で調べる。引数: 出力 json
図S3：P=40、L=6、α=0.2、t=1500 での整列度と密度。論文は ω=2〜7 で整列度が高く、ω≤1 と ω≥8 で低い
ω の読み方：式(13)どおり q = V_u + ω·v_u（omega_on="principal"）と、説明文どおり q = v_u + ω·V_u（"velocity"）
α の読み方：α 2倍・round（論文2で一番多くの図に合う設定）と、比較用に α そのまま・round
ω=0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 10、5シード、1500 ステップ
記録：整列度・密度・θ のそろい φ の時系列（10 ステップごと）
"""
import os, sys, json, itertools
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 研究/
from model.MAM_basic_v1 import SoldierCrabSwarm_v1, polarization2, density_paper2

OMEGAS = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 10]


def one(job):
    on, sc, w, s = job
    m = SoldierCrabSwarm_v1(model="paper2", P=40, alpha=0.2, alpha_scale=sc, omega=float(w), omega_on=on, seed=s)
    pol, dens, phi = [], [], []
    pb, db, fb = [], [], []
    for t in range(1500):
        m.step()
        pb.append(polarization2(m.last_displacement)); db.append(density_paper2(m.position, m.S))
        fb.append(abs(np.mean(np.exp(1j * m.theta))))
        if t % 10 == 9:
            pol.append(float(np.mean(pb))); dens.append(float(np.mean(db))); phi.append(float(np.mean(fb)))
            pb, db, fb = [], [], []
    return dict(on=on, scale=sc, omega=w, seed=s, pol=pol, dens=dens, phi=phi)


if __name__ == "__main__":
    jobs = list(itertools.product(["principal", "velocity"], [2, 1], OMEGAS, range(5)))
    with Pool(14) as p:
        res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
