"""計画T（11/27）：動画から出た2つの手がかりを、論文1 の条件（N=100・S=50・L=4・R(Nf)=2・R(M)=3）で試す。
 留まりあり : 長さ 0 の遷移（自分のサイト）を、自分にとっては空きとして扱う（相互予期・自由移動とも）
 速度一致なし : 論文2 のモデル（θ を自分の移動と ω=5 で更新）を、α そのまま・round で論文1 の条件で回す
P=2・10・30 × α=0.3〜0.6 × 3シード、900 ステップ（t=300〜900 の平均）。
使い方: python planT_leads.py   出力は ../report_data/planT_leads.json
"""
import sys, json, collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # 研究/
from model.MAM_paper1 import CrabSwarmPaper1, polarization1, density_grid
from model.MAM_paper2 import CrabSwarmPaper2

BEST = dict(alpha_scale=1, int_mode="round", ma_rule="sequential", vm_source="disp", popularity="transition")

class Stay(CrabSwarmPaper1):
    def _move_all(self):
        N, S = self.N, self.S
        self.w[:, :] = 0; self.w_ma[:, :] = 0
        targets = self._generate_potential_transitions()
        z = self._compute_popularity(targets)
        occ = self.occupancy.copy(); newp = self.position.copy(); moved = np.zeros(N, bool); cnt = [0, 0, 0, 0]
        def go(k, nx, ny, ma=False):
            x0, y0 = self.position[k]
            if (nx, ny) != (x0, y0):
                occ[x0, y0] = -1; occ[nx, ny] = k; self.w[x0, y0] = 1
                if ma: self.w_ma[x0, y0] = 1
            newp[k] = (nx, ny); moved[k] = True
        def free_mask(k, kt):
            return (occ[kt[:, 0], kt[:, 1]] == -1) | (occ[kt[:, 0], kt[:, 1]] == k)   # 自分のサイトも空きとして扱う
        while True:
            any_moved = False
            for k in self.rng.permutation(np.flatnonzero(~moved)):
                kt = targets[k]; vac = free_mask(k, kt)
                if not vac.any(): continue
                cc = np.where(vac, z[kt[:, 0], kt[:, 1]], -1); m = cc.max()
                if m < self.c: continue
                nx, ny = self._pick(kt, cc == m); go(k, nx, ny, ma=True); cnt[0] += 1; any_moved = True
            if not any_moved: break
        for k in self.rng.permutation(np.flatnonzero(~moved)):
            x0, y0 = self.position[k]
            nbx = np.mod(x0 + self._dx_Nf, S); nby = np.mod(y0 + self._dy_Nf, S)
            ok = (self.w_ma[nbx, nby] == 1) & (occ[nbx, nby] == -1)
            if ok.any():
                i = self.rng.integers(ok.sum()); go(k, nbx[ok][i], nby[ok][i]); cnt[1] += 1
        for k in self.rng.permutation(np.flatnonzero(~moved)):
            kt = targets[k]; vac = free_mask(k, kt)
            if vac.any():
                nx, ny = self._pick(kt, vac); go(k, nx, ny); cnt[2] += 1
            else:
                go(k, *self.position[k]); cnt[3] += 1
        self.position = newp; self.occupancy = occ
        return tuple(cnt)

def run(job):
    var, P, alpha, seed = job
    if var == "今の一番よい版":
        m = CrabSwarmPaper1(P=P, alpha=alpha, seed=seed, **BEST)
    elif var == "留まりあり":
        m = Stay(P=P, alpha=alpha, seed=seed, **BEST)
    else:
        m = CrabSwarmPaper2(N=100, S=50, L=4.0, P=P, alpha=alpha, c=2, omega=5.0, seed=seed, alpha_scale=1, int_mode="round")
    pol, d25, stay = [], [], []
    for t in range(900):
        m.step()
        if t < 300: continue
        pol.append(polarization1(m.last_displacement, 4.0))
        d25.append(density_grid(m.position, 50, 25))
        stay.append(float((np.abs(m.last_displacement).sum(1) == 0).mean()))
    return dict(var=var, P=P, alpha=alpha, seed=seed, pol=float(np.mean(pol)), d25=float(np.mean(d25)), stay=float(np.mean(stay)))

if __name__ == "__main__":
    VARS = ("今の一番よい版", "留まりあり", "速度一致なし（論文2型）")
    jobs = [(v, P, a, s) for v in VARS for P in (2, 10, 30) for a in (0.3, 0.4, 0.5, 0.6) for s in range(3)]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run, jobs, chunksize=3))
    (HERE.parent / "report_data" / "planT_leads.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    g = collections.defaultdict(list)
    for r in res: g[(r["var"], r["P"])].append(r)
    print("論文 図11（P が大きい所）：整列度 0.53/0.30/0.13/0.10、密度 0.8〜0.9（Ls=25）")
    for v in VARS:
        for P in (2, 10, 30):
            rs = g[(v, P)]
            pa = [np.mean([r["pol"] for r in rs if r["alpha"] == a]) for a in (0.3, 0.4, 0.5, 0.6)]
            print(f"{v} P={P}: 整列度 " + "/".join(f"{x:.2f}" for x in pa) +
                  f"  密度25 {np.mean([r['d25'] for r in rs]):.2f}  留まる {np.mean([r['stay'] for r in rs]):.2f}")
