"""論文1：θ がばらつく仕組みを足した版で α の効き方を見る。引数: 出力 json
- ext_noise=ε : 速度一致のあとに θ += ε·π·U[-1,1]（図10左の「外部摂動 [-1,1]」を Vicsek 型の角度ノイズと読む）
- vm_mean="arith" : 式(8) <θ>_M を角度の数値そのものの算術平均と読む（[0,2π) で平均）
"""
import sys, json, itertools
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, r"C:\Users\PC_User\ClaudeCode\研究")
from model.MAM_basic_v1 import SoldierCrabSwarm_v1, polarization1, density_sliding


class Perturbed(SoldierCrabSwarm_v1):
    def __init__(self, ext_noise=0.0, vm_mean="vector", **kw):
        self.ext_noise, self.vm_mean = ext_noise, vm_mean
        super().__init__(**kw)

    def _velocity_matching(self):
        if self.vm_mean == "vector":
            super()._velocity_matching()
        else:
            S = self.S
            th = np.mod(self.theta, 2 * np.pi)
            new = th.copy()
            for k in range(self.N):
                x0, y0 = self.position[k]
                occ = self.occupancy[np.mod(x0 + self._dx_M, S), np.mod(y0 + self._dy_M, S)]
                occ = occ[occ != -1]
                if occ.size:
                    new[k] = th[occ].mean()
            self.theta = new
        if self.ext_noise > 0:
            self.theta = self.theta + self.ext_noise * np.pi * self.rng.uniform(-1, 1, self.N)


def one(args):
    tag, kw, T, burn = args
    m = Perturbed(model="paper1", **kw)
    pol, phi, fwd, dens, ma = [], [], [], [], []
    for t in range(T):
        th0 = m.theta.copy()
        m.step()
        if t < burn:
            continue
        d = m.last_displacement.astype(float)
        pol.append(polarization1(m.last_displacement, m.L)); dens.append(density_sliding(m.position, m.S, 10))
        phi.append(abs(np.mean(np.exp(1j * th0))))
        nrm = np.linalg.norm(d, axis=1); mv = nrm > 0
        fwd.append(float(np.mean(np.cos(np.arctan2(d[mv, 1], d[mv, 0]) - th0[mv]))) if mv.any() else 0.0)
        ma.append(m.last_rule_counts[0] / m.N)
    return dict(tag=tag, alpha=kw["alpha"], P=kw["P"], eps=kw.get("ext_noise", 0.0), seed=kw["seed"],
                pol=float(np.mean(pol)), phi=float(np.mean(phi)), fwd=float(np.mean(fwd)),
                dens=float(np.mean(dens)), ma=float(np.mean(ma)))


if __name__ == "__main__":
    jobs = []
    EPS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for a, e, s in itertools.product([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], EPS, range(5)):
        jobs.append(("ma_noise", dict(P=15, alpha=a, ext_noise=e, seed=s), 900, 300))
    for e, s in itertools.product(EPS, range(5)):        # 図10左：相互予期なし（P=1）
        jobs.append(("p1_noise", dict(P=1, alpha=0.7, ext_noise=e, seed=s), 900, 300))
    for e, s in itertools.product(EPS, range(5)):        # 図10右と同じ α=0.7, P=30
        jobs.append(("p30_noise", dict(P=30, alpha=0.7, ext_noise=e, seed=s), 900, 300))
    for a, s in itertools.product([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], range(5)):
        jobs.append(("arith", dict(P=15, alpha=a, vm_mean="arith", seed=s), 900, 300))
    jobs.sort(key=lambda j: -j[1]["P"])
    with Pool(20) as pool:
        res = pool.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
