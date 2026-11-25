"""計画P（11/26）：論文2 の「どの遷移が選ばれるか」（付録の式(6)〜(11)、本文 p.4）の読み方を、ベスト（α 2倍）で比べる。
計画O で、θ の揺らぎの大部分は選ばれた遷移（実際の移動 V）から来ていて、θ の入れ方（式(2)(13)）は理由ではなかった。
そこで、選び方の読み方のうち、本文・付録に根拠があるものを1つずつ入れる。

ma_mode（相互予期）
  "seq"        : 今のモデル（MAM_paper2）。ランダムな順に1体ずつ、人気度 ≥ c の空きサイトのうち一番人気へ（先に決めた個体が取る）
  "bid"        : 本文 p.4「同じサイトに行きたい個体が複数いたら、そのサイトへの遷移の向きが基本ベクトルに一番近い個体が入り、
                 他の個体は2番目に人気のサイトへ動く」。全員が同時に1番人気を狙う（変更一覧 2-7。α そのままでしか試していなかった）
  "poporder"   : 本文 p.4「サイトの人気度の分布が更新の順番を決める」。狙うサイトの人気度が高い個体から順に更新（2-8。同上）
  "second_any" : 本文 p.4 の「2番目に人気のサイトへ動く」を、2番目の人気度が c 未満でも動く、と読む
                 （今のモデルは、人気のサイトが先に取られたら、人気度 ≥ c の残りがなければ追従・自由移動に回る）
follow_mode（追従）
  "indiv"      : 今のモデル。ランダムな順に1体ずつ、自分の遷移先のうち wanderer が空けたサイトからランダムに1つ
  "site"       : 本文 p.4「新しく空いたサイトに遷移を向けている個体の中から、ランダムに選ばれた個体が動く」。サイトごとに選ぶ
青・赤・緑 × 10シード、1500 ステップ。整列度・密度は t=1000〜1500 の平均。ルールの割合と、ルールごとの扇（移動 V の向きと θ の差、度）も出す。
使い方: python planP_selection.py [出力のjson]   既定は ../report_data/planP_selection.json
"""
import sys, json, collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # 研究/
from model.MAM_paper2 import CrabSwarmPaper2, PRESETS, CONDITIONS, polarization2, density_paper2  # noqa: E402

VARIANTS = {
    "今のモデル": dict(ma_mode="seq", follow_mode="indiv"),
    "競合（bid）": dict(ma_mode="bid", follow_mode="indiv"),
    "人気度の順": dict(ma_mode="poporder", follow_mode="indiv"),
    "2番目は c 未満でも": dict(ma_mode="second_any", follow_mode="indiv"),
    "追従はサイトごと": dict(ma_mode="seq", follow_mode="site"),
}


class Sel(CrabSwarmPaper2):
    def __init__(self, ma_mode="seq", follow_mode="indiv", **kw):
        self.ma_mode, self.follow_mode = ma_mode, follow_mode
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

        def ma(k, site):
            self.w_ma[self.position[k][0], self.position[k][1]] = 1
            move(k, *site); rule[k] = 0

        order = [int(k) for k in self.rng.permutation(N)]

        # ---- 相互予期 ----
        if self.ma_mode in ("seq", "second_any"):
            for k in order:
                tx, ty = targets[k, :, 0], targets[k, :, 1]
                ok = vacant0[tx, ty] & ~taken[tx, ty]
                if not ok.any():
                    continue
                zz = np.where(ok, z[tx, ty], -1)
                m = zz.max()
                if m < c:
                    had_popular = (vacant0[tx, ty] & (z[tx, ty] >= c)).any()
                    if not (self.ma_mode == "second_any" and had_popular):
                        continue
                cand = np.flatnonzero(zz == m)
                a_c = ang[k, cand]
                cand = cand[a_c <= a_c.min() + 1e-9]
                i = int(self.rng.choice(cand))
                ma(k, (tx[i], ty[i]))
        else:
            ranked = []
            for k in range(N):
                tx, ty = targets[k, :, 0], targets[k, :, 1]
                best = {}
                for i in np.flatnonzero(vacant0[tx, ty] & (z[tx, ty] >= c)):
                    s = (int(tx[i]), int(ty[i]))
                    if s not in best or ang[k, i] < best[s]:
                        best[s] = ang[k, i]
                ranked.append(sorted(best.items(), key=lambda kv: (-z[kv[0]], kv[1], self.rng.random())))
            if self.ma_mode == "bid":
                bidders = list(range(N))
                for rank in (0, 1):                  # 1番人気を同時に狙い、負けたら2番目へ
                    bids = {}
                    for k in bidders:
                        if len(ranked[k]) > rank and not taken[ranked[k][rank][0]]:
                            bids.setdefault(ranked[k][rank][0], []).append((ranked[k][rank][1], self.rng.random(), k))
                    bidders = []
                    for site, ks in bids.items():
                        ks.sort()
                        ma(ks[0][2], site)
                        bidders += [k for _, _, k in ks[1:]]
            else:                                    # poporder
                key = [(-(z[r[0][0]] if r else 0), self.rng.random(), k) for k, r in enumerate(ranked)]
                for _, _, k in sorted(key):
                    for site, _ in ranked[k]:
                        if not taken[site]:
                            ma(k, site)
                            break

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
        else:                                        # サイトごとに、そのサイトへ遷移を向けている個体からランダムに1体
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
            if ok.any():
                s = int(self.rng.choice(np.unique(tx[ok] * S + ty[ok])))
                move(k, s // S, s % S); rule[k] = 2

        self.rule = rule
        self.position = newp
        self.occupancy = occ
        return tuple(int((rule == r).sum()) for r in range(4))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run(job):
    var, name, seed = job
    m = Sel(**CONDITIONS[name], omega=5.0, seed=seed, **PRESETS["best"], **VARIANTS[var])
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



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":   # 「今のモデル」が MAM_paper2 と同じ結果になるか
        a = CrabSwarmPaper2(**CONDITIONS["赤"], omega=5.0, seed=3, **PRESETS["best"]); a.run(200)
        b = Sel(**CONDITIONS["赤"], omega=5.0, seed=3, **PRESETS["best"]); b.run(200)
        print("同じ:", np.array_equal(a.position, b.position) and np.allclose(a.theta, b.theta))
        sys.exit()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "report_data" / "planP_selection.json"
    jobs = [(v, n, s) for v in VARIANTS for n in ("青", "赤", "緑") for s in range(10)]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run, jobs))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    g = collections.defaultdict(list)
    for r in res:
        g[(r["var"], r["name"])].append(r)
    print("（論文：青 0.2 / 0.65 / 63-5-32、赤 0.93 / 4.0、緑 0.35 / 3.2）")
    for (v, n), rs in g.items():
        p = np.mean([r["pol"] for r in rs]); d = np.mean([r["dens"] for r in rs])
        ru = 100 * np.mean([r["rules"] for r in rs], axis=0)
        fa = [np.mean([r["fan"][k] for r in rs if r["fan"][k] is not None] or [np.nan]) for k in (0, 1, 2)]
        print(f"{v:12s} {n} 整列度 {p:.2f} 密度 {d:.2f} ルール {ru[0]:.1f}/{ru[1]:.1f}/{ru[2]:.1f} 扇 {fa[0]:.0f}/{fa[1]:.0f}/{fa[2]:.0f}")
