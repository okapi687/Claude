"""論文1（MAM_basic_ex_v2）の再現チェック"""
import sys, json, itertools
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, r"C:\Users\PC_User\ClaudeCode\研究")
from model.MAM_basic_ex_v2 import SoldierCrabSwarm_ex_v2

LS_LIST = [10, 15, 20, 25]


def density_sliding(pos, S, Ls):
    g = np.zeros((S, S), dtype=int)
    g[pos[:, 0], pos[:, 1]] = 1
    c = np.pad(np.tile(g, (2, 2)), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    win = c[Ls:Ls + S, Ls:Ls + S] - c[:S, Ls:Ls + S] - c[Ls:Ls + S, :S] + c[:S, :S]
    return win.max() / len(pos)


def density_blocks(pos, S, Ls):
    """論文の文面どおり：X/Ls × X/Ls の重ならないブロックに分けた最大値 / N"""
    g = np.zeros((S, S), dtype=int)
    g[pos[:, 0], pos[:, 1]] = 1
    n = S // Ls
    g = g[:n * Ls, :n * Ls]
    return g.reshape(n, Ls, n, Ls).sum(axis=(1, 3)).max() / len(pos)


def one(args):
    P, alpha, seed, int_mode, n_tr, n_me = args
    m = SoldierCrabSwarm_ex_v2(N=100, S=50, L=4, P=P, alpha=alpha, R_Nf=2, R_M=3, c=2,
                               seed=seed, int_mode=int_mode)
    m.run(n_tr)
    m.move_counts = {k: 0 for k in m.move_counts}
    pol, pol_unit = [], []
    dens = {Ls: [] for Ls in LS_LIST}
    dblk = {Ls: [] for Ls in LS_LIST}
    for _ in range(n_me):
        m.step()
        d = m.last_displacement.astype(float)
        pol.append(np.linalg.norm(d.mean(0)) / m.L)
        n = np.linalg.norm(d, axis=1); mv = n > 0
        u = np.zeros_like(d); u[mv] = d[mv] / n[mv, None]
        pol_unit.append(np.linalg.norm(u.mean(0)))
        for Ls in LS_LIST:
            dens[Ls].append(density_sliding(m.position, m.S, Ls))
            dblk[Ls].append(density_blocks(m.position, m.S, Ls))
    tot = sum(m.move_counts.values())
    return dict(P=P, alpha=alpha, seed=seed, int_mode=int_mode,
                pol=float(np.mean(pol)), pol_unit=float(np.mean(pol_unit)),
                pol_series=[float(x) for x in pol],
                dens={Ls: float(np.mean(v)) for Ls, v in dens.items()},
                dens_series={Ls: [float(x) for x in v] for Ls, v in dens.items()},
                dblk={Ls: float(np.mean(v)) for Ls, v in dblk.items()},
                rules={k: v / tot for k, v in m.move_counts.items()})


if __name__ == "__main__":
    jobs = []
    # 図10・11：P=1..30, α=0.3..0.7
    for a, P, s, im in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7], range(1, 31), range(3), ["floor", "round"]):
        jobs.append((P, a, s, im, 300, 300))
    # 図8・13・14：α=0.2..0.9, P=1,2,5,10..20
    extra_P = [1, 2, 5, 10, 12, 14, 16, 18, 20]
    for a, P, s, im in itertools.product([0.2, 0.25, 0.35, 0.45, 0.55, 0.8, 0.9], extra_P, range(3), ["floor", "round"]):
        jobs.append((P, a, s, im, 300, 300))
    with Pool(22) as pool:
        res = pool.map(one, jobs, chunksize=4)
    out = sys.argv[1]
    json.dump(res, open(out, "w"))
    print("done", len(res))
