"""計画R（11/26）：論文2 本文の図2（流れ図）どおりに更新の順番を組み直して、ベスト（α 2倍・round）で比べる。

流れ図（本文 図2）の相互予期の部分は
    「人気度が1より大きいサイトはあるか？」→ yes →「1体が一番人気のサイトへ動く」→ もう一度問う（ループ）
と、**サイトを中心にしたループ**になっている。本文 p.4 にも「サイトの人気度の分布が更新の順番を決める」とある。
今のモデル（MAM_paper2）は「ランダムな順に1体ずつ、自分の遷移先で一番人気のサイトへ」なので、ここが違う。
追従は「wanderer が空けたサイトはあるか？」→「ランダムに選ばれた1体がそのサイトへ」、
自由移動は「ランダムに選ばれた1体が、遷移の中から1本をランダムに選んで動く」。

ma_mode（相互予期）
  "seq"     : 今のモデル。ランダムな順に1体ずつ、人気度 ≥ c の空きサイトのうち一番人気へ
  "site"    : 流れ図どおり。まだ取られていない空きサイトのうち、人気度（ステップの最初に数えたもの）が一番高いサイトを選び、
              そこへ遷移を向けている、まだ動いていない個体の1体が入る。これを人気度 ≥ c のサイトがなくなるまでくり返す。
              同じ人気度のサイトの順はランダム
  "site_recount" : "site" で、人気度を「まだ動いていない個体の票」だけで数え直す
              （動いた個体の残りの遷移は、もう人気度に入れない、と読む）
ma_pick（"site" のとき、どの個体が入るか）
  "angle"   : そのサイトへの遷移の向きが、基本ベクトルに一番近い個体（本文 p.4。付録の式(8) の後ろの文）
  "random"  : ランダムに1体（流れ図は「An agent moves」としか書いていない）
follow_mode（追従）
  "indiv"   : 今のモデル。ランダムな順に1体ずつ、自分の遷移先のうち空いたサイトからランダムに1つ
  "site"    : 流れ図どおり。空いたサイトごとに、そこへ遷移を向けている個体からランダムに1体
free_mode（自由移動）
  "site"    : 今のモデル。空いている遷移先のサイトから、サイト単位でランダムに1つ（付録の式(11) は集合 Rd{τ}）
  "vector"  : 流れ図どおり「遷移の中から1本をランダムに選ぶ」。同じサイトへの遷移が多いほど選ばれやすい

青・赤・緑 × 10シード、1500 ステップ。整列度・密度は t=1000〜1500 の平均。ルールの割合と、ルールごとの扇（移動 V の向きと θ の差、度）も出す。
使い方: python planR_flowchart.py [出力のjson]   既定は ../report_data/planR_flowchart.json
        python planR_flowchart.py check          「今のモデル」が MAM_paper2 と同じ軌跡になるか
"""
import sys, json, heapq, collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # 研究/
from model.MAM_paper2 import CrabSwarmPaper2, PRESETS, CONDITIONS, polarization2, density_paper2  # noqa: E402

BASE = dict(ma_mode="seq", ma_pick="angle", follow_mode="indiv", free_mode="site")
VARIANTS = {
    "今のモデル": dict(),
    "相互予期をサイト中心に": dict(ma_mode="site"),
    "サイト中心・入る個体はランダム": dict(ma_mode="site", ma_pick="random"),
    "サイト中心・人気度を数え直す": dict(ma_mode="site_recount"),
    "自由移動を遷移単位で": dict(free_mode="vector"),
    "流れ図どおり（全部）": dict(ma_mode="site", follow_mode="site", free_mode="vector"),
    "流れ図どおり・数え直す": dict(ma_mode="site_recount", follow_mode="site", free_mode="vector"),
}


class Flow(CrabSwarmPaper2):
    def __init__(self, ma_mode="seq", ma_pick="angle", follow_mode="indiv", free_mode="site", **kw):
        self.ma_mode, self.ma_pick, self.follow_mode, self.free_mode = ma_mode, ma_pick, follow_mode, free_mode
        super().__init__(**kw)
        self.rule = np.full(self.N, 3)

    def _move_all(self):
        N, S, c = self.N, self.S, self.c
        self.w_ma[:, :] = 0
        targets, vecs = self._generate_potential_transitions()
        z = self._compute_popularity(targets)
        self._pv_int = vecs[:, 0, :].copy()
        occ = self.occupancy.copy()
        vacant0 = self.occupancy == -1
        taken = np.zeros((S, S), dtype=bool)
        newp = self.position.copy()
        rule = np.full(N, 3)
        a = np.arctan2(vecs[:, :, 1], vecs[:, :, 0])
        ang = np.abs((a - a[:, :1] + np.pi) % (2 * np.pi) - np.pi)

        def move(k, nx, ny):
            x0, y0 = self.position[k]
            occ[x0, y0] = -1; occ[nx, ny] = k; newp[k] = (nx, ny); taken[nx, ny] = True

        def ma(k, x, y):
            self.w_ma[self.position[k][0], self.position[k][1]] = 1
            move(k, x, y); rule[k] = 0

        order = [int(k) for k in self.rng.permutation(N)]

        # ---- 相互予期 ----
        if self.ma_mode == "seq":                    # MAM_paper2 と同じ（乱数の順番も同じ）
            for k in order:
                tx, ty = targets[k, :, 0], targets[k, :, 1]
                ok = vacant0[tx, ty] & ~taken[tx, ty]
                if not ok.any():
                    continue
                zz = np.where(ok, z[tx, ty], -1)
                m = zz.max()
                if m < c:
                    continue
                cand = np.flatnonzero(zz == m)
                a_c = ang[k, cand]
                cand = cand[a_c <= a_c.min() + 1e-9]
                i = int(self.rng.choice(cand))
                ma(k, tx[i], ty[i])
        else:
            # サイト → {個体: そのサイトへの遷移のうち基本ベクトルに一番近い角度}（ステップ開始時に空いているサイトだけ）
            reach = collections.defaultdict(dict)
            for k in range(N):
                tx, ty = targets[k, :, 0], targets[k, :, 1]
                for i in np.flatnonzero(vacant0[tx, ty]):
                    s = (int(tx[i]), int(ty[i]))
                    d = reach[s]
                    if k not in d or ang[k, i] < d[k]:
                        d[k] = ang[k, i]
            recount = self.ma_mode == "site_recount"
            heap = []
            for s, d in reach.items():
                p = len(d) if recount else int(z[s])
                if p >= c:
                    heap.append((-p, self.rng.random(), s))
            heapq.heapify(heap)
            moved = np.zeros(N, dtype=bool)
            while heap:
                negp, r, s = heapq.heappop(heap)
                if taken[s]:
                    continue
                cand = [(v, k) for k, v in reach[s].items() if not moved[k]]
                if recount and len(cand) < -negp:    # 票が減っていたら、数え直した人気度で入れ直す
                    if len(cand) >= c:
                        heapq.heappush(heap, (-len(cand), r, s))
                    continue
                if not cand:
                    continue
                if self.ma_pick == "angle":
                    amin = min(v for v, _ in cand)
                    cand = [(v, k) for v, k in cand if v <= amin + 1e-9]
                k = cand[int(self.rng.integers(len(cand)))][1]
                ma(k, *s)
                moved[k] = True

        # ---- 追従 ----
        rest = []
        if self.follow_mode == "indiv":
            for k in order:
                if rule[k] == 0:
                    continue
                tx, ty = targets[k, :, 0], targets[k, :, 1]
                ok = (self.w_ma[tx, ty] == 1) & (occ[tx, ty] == -1)
                if ok.any():
                    s = int(self.rng.choice(np.unique(tx[ok] * S + ty[ok])))
                    move(k, s // S, s % S); rule[k] = 1
                else:
                    rest.append(k)
        else:                                        # 空いたサイトごとに、そこへ遷移を向けている個体からランダムに1体
            sites = np.argwhere(self.w_ma == 1)
            for j in self.rng.permutation(len(sites)):
                x, y = sites[j]
                if occ[x, y] != -1:
                    continue
                ks = [k for k in order if rule[k] == 3 and
                      np.any((targets[k, :, 0] == x) & (targets[k, :, 1] == y))]
                if ks:
                    k = ks[int(self.rng.integers(len(ks)))]
                    move(k, x, y); rule[k] = 1
            rest = [k for k in order if rule[k] == 3]

        # ---- 自由移動 ----
        for k in rest:
            tx, ty = targets[k, :, 0], targets[k, :, 1]
            ok = occ[tx, ty] == -1
            if not ok.any():
                continue
            if self.free_mode == "site":
                s = int(self.rng.choice(np.unique(tx[ok] * S + ty[ok])))
                move(k, s // S, s % S)
            else:                                    # 遷移を1本ランダムに選ぶ
                i = int(self.rng.choice(np.flatnonzero(ok)))
                move(k, tx[i], ty[i])
            rule[k] = 2

        self.rule = rule
        self.position = newp
        self.occupancy = occ
        return tuple(int((rule == r).sum()) for r in range(4))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run(job):
    var, name, seed = job
    kw = dict(BASE, **VARIANTS[var])
    m = Flow(**CONDITIONS[name], omega=5.0, seed=seed, **PRESETS["best"], **kw)
    pol, den, cnt = [], [], np.zeros(4)
    fan = collections.defaultdict(list)
    for t in range(1500):
        th0 = m.theta.copy()
        m.step()
        if t < 1000:
            continue
        pol.append(polarization2(m.last_displacement)); den.append(density_paper2(m.position, m.S))
        cnt += np.bincount(m.rule, minlength=4)
        d = m.last_displacement.astype(float)
        f = np.degrees(wrap(np.arctan2(d[:, 1], d[:, 0]) - th0))
        for r in range(3):
            fan[r].append(f[m.rule == r])
    rms = {r: (float(np.sqrt(np.mean(np.concatenate(fan[r]) ** 2))) if sum(map(len, fan[r])) else None) for r in range(3)}
    return dict(var=var, name=name, seed=seed, pol=float(np.mean(pol)), dens=float(np.mean(den)),
                rules=(cnt / cnt.sum()).tolist(), fan=rms)


LONG_VARIANTS = ["今のモデル", "相互予期をサイト中心に", "流れ図どおり（全部）"]
LONG_T = 3000


def run_long(job):
    """長く回す（3000 ステップ）。α 2倍と 1倍。整列度・密度の時系列（50 ステップごとの平均）と最後の位置を返す"""
    var, name, seed, ascale = job
    kw = dict(BASE, **VARIANTS[var])
    m = Flow(**CONDITIONS[name], omega=5.0, seed=seed, alpha_scale=ascale, int_mode="round", **kw)
    pol, den, cnt = [], [], np.zeros(4)
    for t in range(LONG_T):
        m.step()
        pol.append(polarization2(m.last_displacement)); den.append(density_paper2(m.position, m.S))
        if t >= LONG_T - 500:
            cnt += np.bincount(m.rule, minlength=4)
    pol, den = np.array(pol), np.array(den)
    return dict(var=var, name=name, seed=seed, ascale=ascale,
                pol_ts=pol.reshape(-1, 50).mean(1).tolist(), dens_ts=den.reshape(-1, 50).mean(1).tolist(),
                pol=float(pol[-500:].mean()), dens=float(den[-500:].mean()),
                rules=(cnt / cnt.sum()).tolist(), pos=m.position.tolist(), theta=m.theta.tolist())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "long":
        out = HERE.parent / "report_data" / "planR_flowchart_long.json"
        jobs = [(v, n, s, a) for a in (2, 1) for v in LONG_VARIANTS for n in ("青", "赤", "緑") for s in range(10)]
        with ProcessPoolExecutor() as ex:
            res = list(ex.map(run_long, jobs))
        out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        g = collections.defaultdict(list)
        for r in res:
            g[(r["ascale"], r["var"], r["name"])].append(r)
        for (a, v, n), rs in g.items():
            ru = 100 * np.mean([r["rules"] for r in rs], axis=0)
            print(f"α{a}倍 {v:12s} {n} 整列度 {np.mean([r['pol'] for r in rs]):.2f} 密度 {np.mean([r['dens'] for r in rs]):.2f} "
                  f"ルール {ru[0]:.1f}/{ru[1]:.1f}/{ru[2]:.1f}  シードごと {[round(r['pol'], 2) for r in rs]}")
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        a = CrabSwarmPaper2(**CONDITIONS["赤"], omega=5.0, seed=3, **PRESETS["best"]); a.run(200)
        b = Flow(**CONDITIONS["赤"], omega=5.0, seed=3, **PRESETS["best"]); b.run(200)
        print("同じ:", np.array_equal(a.position, b.position) and np.allclose(a.theta, b.theta))
        sys.exit()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "report_data" / "planR_flowchart.json"
    jobs = [(v, n, s) for v in VARIANTS for n in ("青", "赤", "緑") for s in range(10)]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run, jobs))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    g = collections.defaultdict(list)
    for r in res:
        g[(r["var"], r["name"])].append(r)
    print("（論文：青 0.2 / 63-5-32、赤 0.93 / 4.0 / 99-1-0、緑 0.35 / 98-2-0）")
    for (v, n), rs in g.items():
        p = np.mean([r["pol"] for r in rs]); d = np.mean([r["dens"] for r in rs])
        ru = 100 * np.mean([r["rules"] for r in rs], axis=0)
        fa = [np.mean([r["fan"][k] for r in rs if r["fan"][k] is not None] or [np.nan]) for k in (0, 1, 2)]
        print(f"{v:16s} {n} 整列度 {p:.2f} 密度 {d:.2f} ルール {ru[0]:.1f}/{ru[1]:.1f}/{ru[2]:.1f} 扇 {fa[0]:.0f}/{fa[1]:.0f}/{fa[2]:.0f}")
