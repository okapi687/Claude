"""計画Q（11/26）：論文1 の文面で分かれる読み方を、今の一番よい版（計画L・M）で試し直す。
今の一番よい版：α そのまま・round・sequential・移動で速度一致・人気度は遷移の本数（2.16 節）。
残っているずれ：P が大きいと整列度が上がる（P=30・α=0.5 で 0.34、論文 図11 は 0.13）、密度（Ls=25）が低い（論文 0.8〜0.9）。
下の読み方は、前に古い設定（θ で速度一致・1個体1票）で試して「変化なし」「不採用」になったもの。今の版では試していない。
  1体ずつ         : p.9「更新する個体を w=0 のサイトからランダムに選ぶ」と式(4)〜(6)。選ばれた個体が、その場で (4)相互予期→(5)追従→(6)自由移動 を判定する（変更一覧 1-8）
  追従先は w 全部 : 式(5) の w=1（相互予期に限らず、更新された個体が空けたサイト）（1-9、follow_chain=True）
  R(Nf)=5         : p.10「追従者近傍の半径は基本ベクトルの長さより大きい」（L=4。図の R(Nf)=2 とは食い違う）（1-10）
  サイト単位      : 式(4) の下「最大人気のサイトが複数あれば1つをランダムに」、式(6) の Rd{t(k,t;i)}（1-16、choose_by="site"）
  floor           : 式(1) の下「X ≤ x < X+1」（1-5、int_mode="floor"。round は今の版に残る唯一の拡大解釈）
  自分を含めない  : 式(8) の近傍 M に自分を含めるかは書かれていない（1-14）
P=1・2・10・30 × α=0.3〜0.6 × 3シード、900 ステップ（t=300〜900 の平均）。L=4、R(Nf)=2、R(M)=3（図10・11 と同じ）。
扇＝実際の移動の向きと θ の差（二乗平均の平方根、度）、φ＝θ のそろい（|平均の (cosθ, sinθ)|）も出す。
使い方: python planQ_paper1_readings.py [出力のjson]   既定は ../report_data/planQ_paper1_readings.json
"""
import sys, json, collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # 研究/
from model.MAM_paper1 import CrabSwarmPaper1, polarization1, density_grid  # noqa: E402

BEST = dict(alpha_scale=1, int_mode="round", ma_rule="sequential", vm_source="disp", popularity="transition")
VARIANTS = {
    "今の一番よい版": dict(),
    "1体ずつ": dict(order="agent"),
    "追従先は w 全部": dict(follow_chain=True),
    "R(Nf)=5": dict(R_Nf=5),
    "サイト単位": dict(choose_by="site"),
    "floor": dict(int_mode="floor"),
    "自分を含めない": dict(vm_self=False),
}
PS, ALPHAS, SEEDS = (1, 2, 10, 30), (0.3, 0.4, 0.5, 0.6), range(3)
STEPS, BURN = 900, 300


class Q(CrabSwarmPaper1):
    def __init__(self, order="phase", vm_self=True, **kw):
        self.order, self.vm_self = order, vm_self
        super().__init__(**kw)

    def _move_all(self):
        if self.order == "phase":
            return super()._move_all()
        # 1体ずつ：ランダムな順に、選ばれた個体がその場で (4)→(5)→(6) を判定する（各個体1回）
        N, S = self.N, self.S
        self.w[:, :] = 0
        self.w_ma[:, :] = 0
        targets = self._generate_potential_transitions()
        z = self._compute_popularity(targets)
        occupancy = self.occupancy.copy()
        new_position = self.position.copy()
        cnt = [0, 0, 0, 0]
        flag = self.w if self.follow_chain else self.w_ma
        for k in self.rng.permutation(N):
            x0, y0 = self.position[k]
            dest = self._best_MA_target(targets[k], z, occupancy)
            r = 0
            if dest is None:
                nbx = np.mod(x0 + self._dx_Nf, S)
                nby = np.mod(y0 + self._dy_Nf, S)
                ok = (flag[nbx, nby] == 1) & (occupancy[nbx, nby] == -1)
                if ok.any():
                    i = self.rng.integers(ok.sum())
                    dest, r = (nbx[ok][i], nby[ok][i]), 1
                else:
                    kt = targets[k]
                    vac = occupancy[kt[:, 0], kt[:, 1]] == -1
                    if vac.any():
                        dest, r = self._pick(kt, vac), 2
                    else:
                        dest, r = (x0, y0), 3
            cnt[r] += 1
            if r == 3:
                continue
            occupancy[x0, y0] = -1
            occupancy[dest[0], dest[1]] = k
            self.w[x0, y0] = 1
            if r == 0:
                self.w_ma[x0, y0] = 1
            new_position[k] = dest
        self.position = new_position
        self.occupancy = occupancy
        return tuple(cnt)

    def _velocity_matching(self):
        if self.vm_self:
            return super()._velocity_matching()
        S = self.S
        d = self.last_displacement.astype(float)
        new_theta = self.theta.copy()
        for k in range(self.N):
            x0, y0 = self.position[k]
            occ = self.occupancy[np.mod(x0 + self._dx_M, S), np.mod(y0 + self._dy_M, S)]
            occ = occ[(occ != -1) & (occ != k)]
            if occ.size:
                mv = d[occ].mean(axis=0)
                if not np.allclose(mv, 0.0):
                    new_theta[k] = np.arctan2(mv[1], mv[0])
        self.theta = new_theta


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run(job):
    var, P, alpha, seed = job
    kw = {**BEST, **VARIANTS[var]}
    m = Q(P=P, alpha=alpha, seed=seed, **kw)
    pol, d25, d10, phi, fan, rules = [], [], [], [], [], []
    for t in range(STEPS):
        th0 = m.theta.copy()
        m.step()
        if t < BURN:
            continue
        d = m.last_displacement.astype(float)
        pol.append(polarization1(m.last_displacement, m.L))
        d25.append(density_grid(m.position, m.S, 25)); d10.append(density_grid(m.position, m.S, 10))
        phi.append(float(np.hypot(np.cos(m.theta).mean(), np.sin(m.theta).mean())))
        mv = np.linalg.norm(d, axis=1) > 0
        fan.append(wrap(np.arctan2(d[mv, 1], d[mv, 0]) - th0[mv]))
        rules.append(np.array(m.last_rule_counts) / m.N)
    return dict(var=var, P=P, alpha=alpha, seed=seed, pol=float(np.mean(pol)), dens25=float(np.mean(d25)),
                dens10=float(np.mean(d10)), phi=float(np.mean(phi)),
                fan=float(np.degrees(np.sqrt(np.mean(np.concatenate(fan) ** 2)))), rules=np.mean(rules, 0).tolist())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":   # 「今の一番よい版」が CrabSwarmPaper1 と同じ軌跡か
        a = CrabSwarmPaper1(P=10, alpha=0.4, seed=1, **BEST); a.run(100)
        b = Q(P=10, alpha=0.4, seed=1, **BEST); b.run(100)
        print("同じ:", np.array_equal(a.position, b.position) and np.allclose(a.theta, b.theta))
        sys.exit()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "report_data" / "planQ_paper1_readings.json"
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
            print(f"{v:10s} P={P:2d} 整列度 " + "/".join(f"{x:.2f}" for x in pa) +
                  f"  密度25 {da:.2f}  φ {ph:.2f}  扇 {fa:.0f}  ルール {ru[0]:.0f}/{ru[1]:.0f}/{ru[2]:.0f}")
