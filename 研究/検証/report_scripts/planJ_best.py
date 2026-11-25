"""計画J：今のベストのモデルの挙動を確かめる。引数: 出力 json
(1) 論文2 図3b,c：青（P=10, α=0.1）・赤（P=40, α=0.2）・緑（P=40, α=0.5）の整列度・密度の時系列（毎ステップ）
    設定：best（α 2倍・ω 式どおり・round・1個体1票）、murakami（α そのまま・ω 式どおり・round）、floor（best の Int だけ floor）
    10シード（floor は5シード）、1500 ステップ。best は t=1500 の位置と、最後の3ステップの軌跡も保存
(2) 論文1：P=15、α=0.3 / 0.5 で、候補 A（α 2倍・round・θ で速度一致）、B（α そのまま・round・移動で速度一致）、
    v1 の既定（α そのまま・floor・θ で速度一致）。5シード、900 ステップ。整列度・θ のそろい φ・区画の密度・ルールの割合と、
    t=900 の位置と最後の10ステップの軌跡（seed 0 だけ）
"""
import os, sys, json, itertools
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 研究/
from model.MAM_basic_v1 import SoldierCrabSwarm_v1, polarization1, polarization2, density_paper2

P2_COND = {"blue": (10, 0.1), "red": (40, 0.2), "green": (40, 0.5)}
P2_SET = {
    "best": dict(alpha_scale=2, omega_on="principal", int_mode="round"),
    "murakami": dict(alpha_scale=1, omega_on="principal", int_mode="round"),
    "floor": dict(alpha_scale=2, omega_on="principal", int_mode="floor"),
}
P1_SET = {
    "A_as2_round_theta": dict(alpha_scale=2, int_mode="round", vm_source="theta"),
    "B_as1_round_disp": dict(alpha_scale=1, int_mode="round", vm_source="disp"),
    "v1_default": dict(alpha_scale=1, int_mode="floor", vm_source="theta"),
}


def density_grid(pos, S, Ls):
    """論文1 p.17：空間を S/Ls × S/Ls の区画に分け、区画の中の最大個体数 / N"""
    n = S // Ls
    h = np.zeros((n, n), int)
    np.add.at(h, (np.minimum(pos[:, 0] // Ls, n - 1), np.minimum(pos[:, 1] // Ls, n - 1)), 1)
    return float(h.max() / len(pos))


def run_p2(job):
    cond, sname, s = job
    P, a = P2_COND[cond]
    m = SoldierCrabSwarm_v1(model="paper2", P=P, alpha=a, seed=s, **P2_SET[sname])
    pol, dens, ma = [], [], []
    tail = []
    for t in range(1500):
        m.step()
        pol.append(polarization2(m.last_displacement)); dens.append(density_paper2(m.position, m.S))
        ma.append(m.last_rule_counts[0] / m.N)
        if t >= 1497:
            tail.append(m.position.tolist())
    out = dict(part="p2", cond=cond, set=sname, seed=s, pol=pol, dens=dens, ma=ma)
    if sname == "best" and s == 0:
        out["tail"] = tail
    return out


def run_p1(job):
    sname, a, s = job
    m = SoldierCrabSwarm_v1(model="paper1", P=15, alpha=a, seed=s, **P1_SET[sname])
    rec = {k: [] for k in ["pol", "phi", "d10", "d25", "ma", "fol", "free"]}
    tail = []
    for t in range(900):
        m.step()
        rec["pol"].append(polarization1(m.last_displacement, m.L))
        rec["phi"].append(float(abs(np.mean(np.exp(1j * m.theta)))))
        rec["d10"].append(density_grid(m.position, m.S, 10)); rec["d25"].append(density_grid(m.position, m.S, 25))
        c = m.last_rule_counts
        rec["ma"].append(c[0] / m.N); rec["fol"].append(c[1] / m.N); rec["free"].append(c[2] / m.N)
        if t >= 890:
            tail.append(m.position.tolist())
    out = dict(part="p1", set=sname, alpha=a, seed=s, **rec)
    if s == 0:
        out["tail"] = tail
    return out


def run(job):
    return run_p2(job[1:]) if job[0] == "p2" else run_p1(job[1:])


if __name__ == "__main__":
    jobs = [("p2", c, sn, s) for c in P2_COND for sn in P2_SET for s in range(5 if sn == "floor" else 10)]
    jobs += [("p1", sn, a, s) for sn in P1_SET for a in (0.3, 0.5) for s in range(5)]
    jobs.sort(key=lambda j: 0 if j[0] == "p2" and j[1] != "blue" else 1)  # 重いものを先に
    with Pool(14) as p:
        res = p.map(run, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
