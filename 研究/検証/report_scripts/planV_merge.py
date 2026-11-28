"""計画V（11/28）：動画（本の図3-8、P=10）の「整列度が低いのに、塊が時間とともに合わさって大きくなる」を出す読み方を探す。
動画と同じ条件（S=100・N=200・P=10・L=4）で、論文に根拠のある読み方を1つずつ今の一番よい版に入れて回す。
  一番よい版 : α そのまま・round・sequential・移動で速度一致・人気度は遷移の本数
  候補は VARS を見る。どれも論文の文面・図・流れ図、またはこれまでの計画で出てきた読み方
測る量は planT_video_model.py と同じ（整列度、離れた個体の留まる割合、塊の数、一番大きい塊）。
α=0.3・0.5・0.7・0.9 × 2シード、1260 ステップ（t=130・680・1220 から 40 ステップずつ）。
後半（引数 paper）：有望だった読み方を論文1 の条件（N=100・S=50）で、P=2・10・20・30 × α=0.3〜0.9 × 3シード、900 ステップ。
  整列度・密度（Ls=25）・一番大きい塊の割合（t=300〜900 の平均）を出す。
使い方: python planV_merge.py         出力は ../report_data/planV_merge.json
        python planV_merge.py paper   出力は ../report_data/planV_merge_paper.json
"""
import sys, json, heapq, collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # 研究/
sys.path.insert(0, str(HERE))
from model.MAM_paper1 import CrabSwarmPaper1  # noqa: E402
from model.MAM_paper2 import CrabSwarmPaper2  # noqa: E402
from planT_video_model import metrics  # noqa: E402
from planU_stay import Stay  # noqa: E402

BEST = dict(alpha_scale=1, int_mode="round", ma_rule="sequential", vm_source="disp", popularity="transition")


class SiteMA(CrabSwarmPaper1):
    """相互予期をサイト中心のループにする（論文2 本文 図2 の流れ図、計画R と同じ）。
    人気度が一番高い空きサイトから順に、そこへ遷移を向けている、まだ動いていない個体のうち
    向きが基本ベクトルに一番近い個体が入る。人気度 ≥ c のサイトがなくなるまで。
    論文1 は「Gunji 2012・Murakami 2017 のモデルは速度一致がないほかは同じ」と書いているので、論文1 にも入れてみる。
    追従・自由移動は今のモデルのまま"""

    def _move_all(self):
        N, S, c = self.N, self.S, self.c
        self.w[:, :] = 0; self.w_ma[:, :] = 0
        targets = self._generate_potential_transitions()
        z = self._compute_popularity(targets)
        occ = self.occupancy.copy(); newp = self.position.copy(); moved = np.zeros(N, bool); cnt = [0, 0, 0, 0]
        d = targets - self.position[:, None, :]
        d = (d + S // 2) % S - S // 2
        a = np.arctan2(d[..., 1], d[..., 0])
        ang = np.abs((a - self.theta[:, None] + np.pi) % (2 * np.pi) - np.pi)

        def go(k, nx, ny, ma=False):
            x0, y0 = self.position[k]
            occ[x0, y0] = -1; occ[nx, ny] = k; self.w[x0, y0] = 1
            if ma: self.w_ma[x0, y0] = 1
            newp[k] = (nx, ny); moved[k] = True

        vacant0 = self.occupancy == -1
        reach = collections.defaultdict(dict)
        for k in range(N):
            tx, ty = targets[k, :, 0], targets[k, :, 1]
            for i in np.flatnonzero(vacant0[tx, ty]):
                s = (int(tx[i]), int(ty[i])); dd = reach[s]
                if k not in dd or ang[k, i] < dd[k]:
                    dd[k] = ang[k, i]
        heap = [(-int(z[s]), min(dd.values()), self.rng.random(), s) for s, dd in reach.items() if z[s] >= c]
        heapq.heapify(heap)
        while heap:
            negp, amin0, r, s = heapq.heappop(heap)
            if occ[s] != -1:
                continue
            cand = [(v, k) for k, v in reach[s].items() if not moved[k]]
            if not cand:
                continue
            a_now = min(v for v, _ in cand)
            if a_now > amin0 + 1e-9:
                heapq.heappush(heap, (negp, a_now, r, s)); continue
            cand = [k for v, k in cand if v <= a_now + 1e-9]
            go(cand[int(self.rng.integers(len(cand)))], *s, ma=True); cnt[0] += 1
        for k in self.rng.permutation(np.flatnonzero(~moved)):
            x0, y0 = self.position[k]
            nbx = np.mod(x0 + self._dx_Nf, S); nby = np.mod(y0 + self._dy_Nf, S)
            ok = (self.w_ma[nbx, nby] == 1) & (occ[nbx, nby] == -1)
            if ok.any():
                i = self.rng.integers(ok.sum()); go(k, nbx[ok][i], nby[ok][i]); cnt[1] += 1
        for k in self.rng.permutation(np.flatnonzero(~moved)):
            kt = targets[k]; vac = occ[kt[:, 0], kt[:, 1]] == -1
            if vac.any():
                nx, ny = self._pick(kt, vac); go(k, nx, ny); cnt[2] += 1
            else:
                go(k, *self.position[k]); cnt[3] += 1
        self.position = newp; self.occupancy = occ
        return tuple(cnt)


# 読み方 → (クラス, 一番よい版から変える引数)。根拠は 04_本の動画.md 2.22 の表
VARS = {
    "今の一番よい版":           (CrabSwarmPaper1, {}),
    "1個体1票":                 (CrabSwarmPaper1, dict(popularity="individual")),
    "θ で速度一致":             (CrabSwarmPaper1, dict(vm_source="theta")),
    "単位ベクトルで速度一致":   (CrabSwarmPaper1, dict(vm_source="unit_disp")),
    "R(M)=2":                   (CrabSwarmPaper1, dict(R_M=2)),
    "R(Nf)=5":                  (CrabSwarmPaper1, dict(R_Nf=5)),
    "追従の連鎖":               (CrabSwarmPaper1, dict(follow_chain=True)),
    "自由移動で待つ":           (CrabSwarmPaper1, dict(free_wait=True)),
    "サイト単位で選ぶ":         (CrabSwarmPaper1, dict(choose_by="site")),
    "競合を文字どおり":         (CrabSwarmPaper1, dict(ma_rule="literal")),
    "相互予期をサイト中心に":   (SiteMA, {}),
    "サイト中心・1個体1票":     (SiteMA, dict(popularity="individual")),
    "1個体1票・留まりあり":     ("stay", dict(popularity="individual")),
    "θ・1個体1票":              (CrabSwarmPaper1, dict(vm_source="theta", popularity="individual")),
    "速度一致なし（論文2型）":  ("p2", {}),
}


def make(var, alpha, seed, N=200, S=100, P=10):
    cls, kw = VARS[var]
    base = dict(N=N, S=S, L=4.0, P=P, alpha=alpha, seed=seed)
    if cls == "p2":
        return CrabSwarmPaper2(c=2, omega=5.0, alpha_scale=1, int_mode="round", **base)
    if cls == "stay":
        return Stay(True, True, False, **dict(BEST, **kw), **base)
    return cls(**dict(BEST, **kw), **base)


def run(job):
    var, alpha, seed = job
    m = make(var, alpha, seed)
    marks = {130: [], 680: [], 1220: []}
    for t in range(1, 1261):
        prev = m.position.copy(); m.step()
        for t0 in marks:
            if t0 <= t < t0 + 40:
                marks[t0].append(metrics(prev, m.position.copy(), m.last_displacement.astype(float)))
    seg = []
    for t0, ms in marks.items():
        iso = np.concatenate([x[1] for x in ms])
        seg.append(dict(t0=t0, pol=round(float(np.mean([x[0] for x in ms])), 3),
                        stayIso=round(float((iso == 0).mean()) if len(iso) else -1, 3),
                        clusters=int(np.mean([x[3] for x in ms])), big=int(np.mean([x[4][0] for x in ms]))))
    return dict(var=var, alpha=alpha, seed=seed, seg=seg)


def biggest(pos, S):
    """チェビシェフ距離 2 以内でつながる塊のうち、一番大きいものの個体数"""
    d = np.abs(pos[None, :, :] - pos[:, None, :]); d = np.minimum(d, S - d).max(-1) <= 2
    n = len(pos); seen = np.zeros(n, bool); best = 0
    for i in range(n):
        if seen[i]: continue
        st = [i]; seen[i] = True; sz = 0
        while st:
            j = st.pop(); sz += 1
            for k in np.flatnonzero(d[j] & ~seen):
                seen[k] = True; st.append(k)
        best = max(best, sz)
    return best


PAPER_VARS = ("今の一番よい版", "θ で速度一致", "θ・1個体1票", "1個体1票・留まりあり")


def run_paper(job):
    from model.MAM_paper1 import polarization1, density_grid
    var, P, alpha, seed = job
    m = make(var, alpha, seed, N=100, S=50, P=P)
    pol, d25, big = [], [], []
    for t in range(900):
        m.step()
        if t < 300: continue
        pol.append(polarization1(m.last_displacement, 4.0)); d25.append(density_grid(m.position, 50, 25))
        if t % 20 == 0: big.append(biggest(m.position, 50) / 100)
    return dict(var=var, P=P, alpha=alpha, seed=seed, pol=float(np.mean(pol)), d25=float(np.mean(d25)), big=float(np.mean(big)))


def main_paper():
    jobs = [(v, P, a, s) for v in PAPER_VARS for P in (2, 10, 20, 30) for a in (0.3, 0.4, 0.5, 0.6, 0.7, 0.9) for s in range(3)]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run_paper, jobs, chunksize=3))
    (HERE.parent / "report_data" / "planV_merge_paper.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    g = collections.defaultdict(list)
    for r in res: g[(r["var"], r["P"])].append(r)
    print("論文 図11：整列度 α=0.3〜0.6 で 0.53/0.30/0.13/0.10、密度 0.8〜0.9")
    for (v, P), rs in g.items():
        q = lambda key: "/".join(f"{np.mean([r[key] for r in rs if r['alpha'] == a]):.2f}" for a in (0.3, 0.4, 0.5, 0.6, 0.7, 0.9))
        print(f"{v} P={P}: 整列度 {q('pol')}  密度 {q('d25')}  一番大きい塊 {q('big')}")


if __name__ == "__main__" and sys.argv[1:] == ["paper"]:
    main_paper()
elif __name__ == "__main__":
    jobs = [(v, a, s) for v in VARS for a in (0.3, 0.5, 0.7, 0.9) for s in range(2)]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run, jobs))
    (HERE.parent / "report_data" / "planV_merge.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    g = collections.defaultdict(list)
    for r in res: g[(r["var"], r["alpha"])].append(r)
    print("動画 P=10：整列度 0.14〜0.18、塊 55 → 41 → 39、一番大きい塊 46 → 75 → 110")
    for (v, a), rs in g.items():
        segs = list(zip(*[r["seg"] for r in rs]))
        f = lambda key: " → ".join(str(round(np.mean([x[key] for x in s]))) for s in segs)
        print(f"{v} α={a}: 整列度 {np.mean([x['pol'] for s in segs for x in s]):.2f}  塊 {f('clusters')}  一番大きい {f('big')}  "
              f"留まる {np.mean([x['stayIso'] for s in segs for x in s]):.2f}")
