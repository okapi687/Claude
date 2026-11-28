"""計画U（11/28）：計画T の「留まりあり」を、留まり方の読み方で分けて試す。
長さ 0 の遷移（η が小さくて丸めると (0,0) になる遷移）を、どの場面で、どう数えるかを変える。
  留まりあり（両方）     : 相互予期・自由移動とも、自分のサイトを自分にとっては空きとする（計画T と同じ）
  自由移動だけ           : 相互予期では自分のサイトは空きでない（式(6) の文字どおり）。自由移動でだけ留まれる
  自分の票を数えない     : 両方で留まれるが、自分のサイトの人気度から自分の遷移の本数を引く
  1個体1票               : 両方で留まれるが、人気度を「そのサイトを含む個体の数」で数える（同じ個体は1票）
(1) 論文1 の条件（N=100・S=50）：P=2・10・30 × α=0.3〜0.6 × 3シード、900 ステップ（t=300〜900 の平均）
(2) 動画の条件（S=100・N=200・P=10）：α=0.3・0.5・0.7 × 2シード。planT_video_model.py と同じ量（離れた個体の留まる割合、塊）
使い方: python planU_stay.py   出力は ../report_data/planU_stay.json
"""
import sys, json, collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # 研究/
sys.path.insert(0, str(HERE))
from model.MAM_paper1 import CrabSwarmPaper1, polarization1, density_grid  # noqa: E402
from planT_video_model import metrics  # noqa: E402  （S=100 用の測り方）

BEST = dict(alpha_scale=1, int_mode="round", ma_rule="sequential", vm_source="disp", popularity="transition")
# 読み方 → (相互予期で留まれる, 自由移動で留まれる, 自分の票を引く, 人気度の数え方)
VARS = {
    "今の一番よい版":     None,
    "留まりあり（両方）": (True, True, False, "transition"),
    "自由移動だけ":       (False, True, False, "transition"),
    "自分の票を数えない": (True, True, True, "transition"),
    "1個体1票":           (True, True, False, "individual"),
}


class Stay(CrabSwarmPaper1):
    def __init__(self, ma_self, free_self, excl_self, **kw):
        super().__init__(**kw)
        self.ma_self, self.free_self, self.excl_self = ma_self, free_self, excl_self

    def _move_all(self):
        N, S = self.N, self.S
        self.w[:, :] = 0; self.w_ma[:, :] = 0
        targets = self._generate_potential_transitions()
        z = self._compute_popularity(targets)
        occ = self.occupancy.copy(); newp = self.position.copy(); moved = np.zeros(N, bool); cnt = [0, 0, 0, 0]
        own = (targets == self.position[:, None, :]).all(-1)          # (N,P) 長さ 0 の遷移
        nself = own.sum(1)

        def go(k, nx, ny, ma=False):
            x0, y0 = self.position[k]
            if (nx, ny) != (x0, y0):
                occ[x0, y0] = -1; occ[nx, ny] = k; self.w[x0, y0] = 1
                if ma: self.w_ma[x0, y0] = 1
            newp[k] = (nx, ny); moved[k] = True

        def vacant(k, kt, allow_self):
            o = occ[kt[:, 0], kt[:, 1]]
            return (o == -1) | ((o == k) & allow_self)

        while True:
            any_moved = False
            for k in self.rng.permutation(np.flatnonzero(~moved)):
                kt = targets[k]; vac = vacant(k, kt, self.ma_self)
                if not vac.any(): continue
                zz = z[kt[:, 0], kt[:, 1]].copy()
                if self.excl_self:
                    zz[own[k]] -= nself[k]
                cc = np.where(vac, zz, -1); m = cc.max()
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
            kt = targets[k]; vac = vacant(k, kt, self.free_self)
            if vac.any():
                nx, ny = self._pick(kt, vac); go(k, nx, ny); cnt[2] += 1
            else:
                go(k, *self.position[k]); cnt[3] += 1
        self.position = newp; self.occupancy = occ
        return tuple(cnt)


def make(var, **kw):
    v = VARS[var]
    if v is None:
        return CrabSwarmPaper1(**BEST, **kw)
    b = dict(BEST, popularity=v[3])
    return Stay(v[0], v[1], v[2], **b, **kw)


def run_paper(job):
    var, P, alpha, seed = job
    m = make(var, P=P, alpha=alpha, seed=seed)
    pol, d25, stay = [], [], []
    for t in range(900):
        m.step()
        if t < 300: continue
        pol.append(polarization1(m.last_displacement, 4.0))
        d25.append(density_grid(m.position, 50, 25))
        stay.append(float((np.abs(m.last_displacement).sum(1) == 0).mean()))
    return dict(cond="論文1", var=var, P=P, alpha=alpha, seed=seed,
                pol=float(np.mean(pol)), d25=float(np.mean(d25)), stay=float(np.mean(stay)))


def run_video(job):
    var, alpha, seed = job
    m = make(var, N=200, S=100, L=4.0, P=10, alpha=alpha, seed=seed)
    marks = {130: [], 680: [], 1220: []}
    for t in range(1, 1261):
        prev = m.position.copy(); m.step()
        for t0 in marks:
            if t0 <= t < t0 + 40:
                marks[t0].append(metrics(prev, m.position.copy(), m.last_displacement.astype(float)))
    seg = []
    for t0, ms in marks.items():
        iso = np.concatenate([x[1] for x in ms])
        seg.append(dict(t0=t0, pol=round(float(np.mean([x[0] for x in ms])), 2),
                        stayAll=round(float(np.mean([x[2] for x in ms])), 2),
                        stayIso=round(float((iso == 0).mean()) if len(iso) else -1, 2),
                        clusters=ms[-1][3], big=[int(b) for b in ms[-1][4]]))
    return dict(cond="動画", var=var, alpha=alpha, seed=seed, seg=seg)


if __name__ == "__main__":
    jp = [(v, P, a, s) for v in VARS for P in (2, 10, 30) for a in (0.3, 0.4, 0.5, 0.6) for s in range(3)]
    jv = [(v, a, s) for v in VARS for a in (0.3, 0.5, 0.7) for s in range(2)]
    with ProcessPoolExecutor() as ex:
        rv = list(ex.map(run_video, jv))
        rp = list(ex.map(run_paper, jp, chunksize=3))
    (HERE.parent / "report_data" / "planU_stay.json").write_text(json.dumps(rp + rv, ensure_ascii=False, indent=1), encoding="utf-8")
    g = collections.defaultdict(list)
    for r in rp: g[(r["var"], r["P"])].append(r)
    print("論文 図11（P が大きい所）：整列度 0.53/0.30/0.13/0.10、密度 0.8〜0.9（Ls=25）")
    for v in VARS:
        for P in (2, 10, 30):
            rs = g[(v, P)]
            pa = [np.mean([r["pol"] for r in rs if r["alpha"] == a]) for a in (0.3, 0.4, 0.5, 0.6)]
            print(f"{v} P={P}: 整列度 " + "/".join(f"{x:.2f}" for x in pa) +
                  f"  密度25 {np.mean([r['d25'] for r in rs]):.2f}  留まる {np.mean([r['stay'] for r in rs]):.2f}")
    print("動画 P=10：整列度 0.14〜0.18、離れた個体の留まる 7〜9%、一番大きい塊 46 → 75 → 110")
    for r in rv:
        print(r["var"], f"α={r['alpha']} s{r['seed']}", " | ".join(
            f"t{x['t0']}: 整列度 {x['pol']} 留まる {x['stayAll']}/{x['stayIso']} 塊 {x['clusters']} {x['big'][:3]}" for x in r["seg"]))
