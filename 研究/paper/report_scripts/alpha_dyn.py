"""α の効き方を、1か所だけ変えた版どうしで比べる。引数: 出力 json"""
import sys, json, itertools
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, r"C:\Users\PC_User\ClaudeCode\研究")
from model.MAM_basic_v1 import SoldierCrabSwarm_v1, polarization1, polarization2, density_sliding, density_paper2


class Variant(SoldierCrabSwarm_v1):
    """実験用：v1 を1か所だけ変える"""

    def __init__(self, vm="theta", eta_mode="uniform", count_all=False, **kw):
        self.vm, self.eta_mode, self.count_all = vm, eta_mode, count_all
        super().__init__(**kw)

    def _generate_potential_transitions(self):
        if self.eta_mode == "uniform":
            return super()._generate_potential_transitions()
        N, P, S, L, alpha = self.N, self.P, self.S, self.L, self.alpha
        eta = self.rng.uniform(0.0, 1.0, size=(N, P))
        xi = self.rng.uniform(-alpha * np.pi, alpha * np.pi, size=(N, P))
        if self.eta_mode == "sqrt":           # 扇形の中に一様（面積あたり一様）
            eta = np.sqrt(eta)
        elif self.eta_mode == "one":          # 長さは全部 L
            eta = np.ones_like(eta)
        eta[:, 0] = 1.0; xi[:, 0] = 0.0
        th = self.theta[:, None]
        vx = self._to_int(L * eta * np.cos(th + xi)); vy = self._to_int(L * eta * np.sin(th + xi))
        tx = np.mod(self.position[:, 0:1] + vx, S); ty = np.mod(self.position[:, 1:2] + vy, S)
        return np.stack([tx, ty], axis=-1), np.stack([vx, vy], axis=-1)

    def _compute_popularity(self, targets, dedupe=True):
        return super()._compute_popularity(targets, dedupe=(False if self.count_all else dedupe))

    def _velocity_matching(self):
        if self.vm == "theta":                # v1：近傍の基本ベクトルの向きの平均
            return super()._velocity_matching()
        # "velocity"：近傍の実際の移動（単位ベクトル）の向きの平均（式(8)「average velocity direction」）
        S = self.S
        d = self.last_displacement.astype(float)
        nrm = np.linalg.norm(d, axis=1)
        u = np.zeros_like(d); mv = nrm > 0; u[mv] = d[mv] / nrm[mv, None]
        new = self.theta.copy()
        for k in range(self.N):
            x0, y0 = self.position[k]
            occ = self.occupancy[np.mod(x0 + self._dx_M, S), np.mod(y0 + self._dy_M, S)]
            occ = occ[occ != -1]
            m = u[occ].sum(0)
            if np.hypot(*m) > 1e-9:
                new[k] = np.arctan2(m[1], m[0])
        self.theta = new


def one(args):
    tag, kw, T, burn = args
    m = Variant(**kw)
    pol, phi, fwd, dens, ma = [], [], [], [], []
    for t in range(T):
        th0 = m.theta.copy()
        m.step()
        if t < burn:
            continue
        d = m.last_displacement.astype(float)
        if m.model == "paper1":
            pol.append(polarization1(m.last_displacement, m.L)); dens.append(density_sliding(m.position, m.S, 10))
        else:
            pol.append(polarization2(m.last_displacement)); dens.append(density_paper2(m.position, m.S))
        phi.append(abs(np.mean(np.exp(1j * th0))))                      # 基本ベクトルのそろい具合
        nrm = np.linalg.norm(d, axis=1); mv = nrm > 0
        rel = np.arctan2(d[mv, 1], d[mv, 0]) - th0[mv]
        fwd.append(float(np.mean(np.cos(rel))))                           # 実際の移動が基本ベクトルの向きにどれだけ沿うか
        ma.append(m.last_rule_counts[0] / m.N)
    return dict(tag=tag, alpha=kw["alpha"], seed=kw["seed"], pol=float(np.mean(pol)), phi=float(np.mean(phi)),
                fwd=float(np.mean(fwd)), dens=float(np.mean(dens)), ma=float(np.mean(ma)))


if __name__ == "__main__":
    jobs = []
    v1 = {"p1_base": {}, "p1_vm_velocity": dict(vm="velocity"), "p1_count_all": dict(count_all=True),
          "p1_eta_sqrt": dict(eta_mode="sqrt"), "p1_eta_one": dict(eta_mode="one")}
    for (tag, kw), a, s in itertools.product(v1.items(), [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], range(3)):
        jobs.append((tag, dict(model="paper1", P=15, alpha=a, seed=s, **kw), 600, 300))
    v2 = {"p2_base_velocity": {}, "p2_base_principal": dict(omega_on="principal"),
          "p2_tiebreak_random": dict(tiebreak="random"), "p2_tiebreak_random_principal": dict(tiebreak="random", omega_on="principal"),
          "p2_count_all": dict(dedupe_votes=False), "p2_eta_sqrt": dict(eta_mode="sqrt"),
          "p2_spv_angle": dict(spv_mode="angle")}
    for (tag, kw), a, s in itertools.product(v2.items(), [0.1, 0.2, 0.3, 0.4, 0.5], range(3)):
        jobs.append((tag, dict(model="paper2", P=40, alpha=a, seed=s, **kw), 1500, 1000))
    with Pool(20) as pool:
        res = pool.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
