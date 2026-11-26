"""計画S：論文1 で、速度一致（式(8)）をする順番を変える（11/26）
今の一番よい版（α そのまま・round・sequential・移動で速度一致・遷移の本数）をもとに、次を比べる。
  今の一番よい版        ：全員が動く → 全員で速度一致（ステップの最後）
  速度一致を前に（一斉）：全員で速度一致 → 全員が動く。ループの区切りが変わるだけで、同じ結果になるはず
  1体ずつ（速度一致は最後）：計画Q の「1体ずつ」。順番の効果だけを見るための比較
  1体ずつ・速度一致→判定 ：ランダムな順に1体ずつ、①自分の θ を近傍の「最新の移動」の平均にそろえ、
                          ②その θ で可能的遷移を作り直し（人気度も自分の票を入れ替える）、
                          ③その場で (4)相互予期 → (5)追従 → (6)自由移動 を判定して動く
  θ で速度一致（一斉）/（1体ずつ）：速度一致で θ を平均する版。1体ずつは、先にそろえた隣の θ をすぐ使う
使い方: python planS_paper1_vm_order.py [check]
"""
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))          # 研究/
from model.MAM_paper1 import CrabSwarmPaper1, polarization1, density_grid  # noqa: E402
from planQ_paper1_readings import Q, wrap  # noqa: E402

BEST = dict(alpha_scale=1, int_mode="round", ma_rule="sequential", vm_source="disp", popularity="transition")
VARIANTS = {
    "今の一番よい版": dict(),
    "速度一致を前に（一斉）": dict(vm_when="before"),
    "1体ずつ（速度一致は最後）": dict(order="agent"),
    "1体ずつ・速度一致→判定": dict(vm_when="agent"),
    "θ で速度一致（一斉）": dict(vm_source="theta"),
    "θ で速度一致（1体ずつ）": dict(vm_source="theta", vm_when="async"),
}
PS, ALPHAS, SEEDS = (1, 2, 10, 30), (0.3, 0.4, 0.5, 0.6), range(3)
STEPS, BURN = 900, 300


class S(Q):
    def __init__(self, vm_when="after", **kw):
        self.vm_when = vm_when
        super().__init__(**kw)

    def step(self):
        if self.vm_when == "before":
            self._velocity_matching()
        old = self.position.copy()
        counts = self._move_all() if self.vm_when != "agent" else self._agent_vm_move()
        d = self.position - old
        self.last_displacement = (d + self.S // 2) % self.S - self.S // 2
        self.last_rule_counts = counts
        for key, n in zip(("MA", "follow", "free", "stay"), counts):
            self.move_counts[key] += n
        if self.vm_when == "after":
            self._velocity_matching()
        elif self.vm_when == "async":
            self._async_theta_vm()
        self.t += 1

    # ---- θ を1体ずつそろえる（先にそろえた隣の θ をすぐ使う）----
    def _async_theta_vm(self):
        S = self.S
        for k in self.rng.permutation(self.N):
            x0, y0 = self.position[k]
            occ = self.occupancy[np.mod(x0 + self._dx_M, S), np.mod(y0 + self._dy_M, S)]
            a = self.theta[occ[occ != -1]]
            mv = np.array([np.cos(a).mean(), np.sin(a).mean()])
            if not np.allclose(mv, 0.0):
                self.theta[k] = np.arctan2(mv[1], mv[0])

    # ---- 1体の可能的遷移（式(1)）----
    def _targets_one(self, k):
        P, S, L = self.P, self.S, self.L
        al = self.alpha * self.alpha_scale
        eta = self.rng.uniform(0.0, 1.0, P); xi = self.rng.uniform(-al * np.pi, al * np.pi, P)
        eta[0], xi[0] = 1.0, 0.0
        vx = self._to_int(L * eta * np.cos(self.theta[k] + xi))
        vy = self._to_int(L * eta * np.sin(self.theta[k] + xi))
        return np.stack([np.mod(self.position[k, 0] + vx, S), np.mod(self.position[k, 1] + vy, S)], axis=-1)

    def _vote(self, z, kt, sign):
        u = kt if self.popularity == "transition" else np.unique(kt, axis=0)
        np.add.at(z, (u[:, 0], u[:, 1]), sign)

    def _agent_vm_move(self):
        N, S = self.N, self.S
        self.w[:, :] = 0
        self.w_ma[:, :] = 0
        targets = self._generate_potential_transitions()
        z = self._compute_popularity(targets)
        occupancy = self.occupancy.copy()
        pos = self.position.copy()                       # その時点の位置（動いた個体は新しい位置）
        cur_d = self.last_displacement.astype(float).copy()   # 最新の移動（このステップで動いた個体は今の移動）
        cnt = [0, 0, 0, 0]
        flag = self.w if self.follow_chain else self.w_ma
        for k in self.rng.permutation(N):
            x0, y0 = pos[k]
            # ① 速度一致（式(8)）：近傍（自分を含む）の最新の移動の平均の向き
            occ = occupancy[np.mod(x0 + self._dx_M, S), np.mod(y0 + self._dy_M, S)]
            mv = cur_d[occ[occ != -1]].mean(axis=0)
            if not np.allclose(mv, 0.0):
                self.theta[k] = np.arctan2(mv[1], mv[0])
            # ② 新しい θ で可能的遷移を作り直し、人気度の自分の票を入れ替える
            self._vote(z, targets[k], -1)
            targets[k] = self._targets_one(k)
            self._vote(z, targets[k], +1)
            # ③ その場で判定
            dest = self._best_MA_target(targets[k], z, occupancy)
            r = 0
            if dest is None:
                nbx, nby = np.mod(x0 + self._dx_Nf, S), np.mod(y0 + self._dy_Nf, S)
                ok = (flag[nbx, nby] == 1) & (occupancy[nbx, nby] == -1)
                if ok.any():
                    i = self.rng.integers(ok.sum())
                    dest, r = (nbx[ok][i], nby[ok][i]), 1
                else:
                    kt = targets[k]
                    vac = occupancy[kt[:, 0], kt[:, 1]] == -1
                    dest, r = (self._pick(kt, vac), 2) if vac.any() else ((x0, y0), 3)
            cnt[r] += 1
            if r == 3:
                cur_d[k] = 0.0
                continue
            occupancy[x0, y0] = -1
            occupancy[dest[0], dest[1]] = k
            self.w[x0, y0] = 1
            if r == 0:
                self.w_ma[x0, y0] = 1
            dd = (np.array(dest) - pos[k] + S // 2) % S - S // 2
            cur_d[k] = dd
            pos[k] = dest
        self.position = pos
        self.occupancy = occupancy
        return tuple(cnt)


def run(job):
    var, P, alpha, seed = job
    m = S(P=P, alpha=alpha, seed=seed, **{**BEST, **VARIANTS[var]})
    pol, d25, phi, fan, rules = [], [], [], [], []
    for t in range(STEPS):
        th0 = m.theta.copy()
        m.step()
        if t < BURN:
            continue
        d = m.last_displacement.astype(float)
        pol.append(polarization1(m.last_displacement, m.L))
        d25.append(density_grid(m.position, m.S, 25))
        phi.append(float(np.hypot(np.cos(m.theta).mean(), np.sin(m.theta).mean())))
        mv = np.linalg.norm(d, axis=1) > 0
        fan.append(wrap(np.arctan2(d[mv, 1], d[mv, 0]) - th0[mv]))
        rules.append(np.array(m.last_rule_counts) / m.N)
    return dict(var=var, P=P, alpha=alpha, seed=seed, pol=float(np.mean(pol)), dens25=float(np.mean(d25)),
                phi=float(np.mean(phi)), fan=float(np.degrees(np.sqrt(np.mean(np.concatenate(fan) ** 2)))),
                rules=np.mean(rules, 0).tolist())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        a = CrabSwarmPaper1(P=10, alpha=0.4, seed=1, **BEST); a.run(100)
        b = S(P=10, alpha=0.4, seed=1, **BEST); b.run(100)
        print("今の一番よい版が同じ:", np.array_equal(a.position, b.position) and np.allclose(a.theta, b.theta))
        sys.exit()
    out = HERE.parent / "report_data" / "planS_paper1_vm_order.json"
    jobs = [(v, P, a, s) for v in VARIANTS for P in PS for a in ALPHAS for s in SEEDS]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run, jobs, chunksize=4))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    g = collections.defaultdict(list)
    for r in res:
        g[(r["var"], r["P"])].append(r)
    print("（論文 図11、P が大きい所：整列度 α=0.3/0.4/0.5/0.6 で 0.53/0.30/0.13/0.10、密度 Ls=25 で 0.8〜0.9）")
    for v in VARIANTS:
        for P in PS:
            rs = g[(v, P)]
            pa = [np.mean([r["pol"] for r in rs if r["alpha"] == a]) for a in ALPHAS]
            da = np.mean([r["dens25"] for r in rs]); fa = np.mean([r["fan"] for r in rs]); ph = np.mean([r["phi"] for r in rs])
            ru = 100 * np.mean([r["rules"] for r in rs], axis=0)
            print(f"{v:14s} P={P:2d} 整列度 " + "/".join(f"{x:.2f}" for x in pa) +
                  f"  密度25 {da:.2f}  φ {ph:.2f}  扇 {fa:.0f}  ルール {ru[0]:.0f}/{ru[1]:.0f}/{ru[2]:.0f}")
