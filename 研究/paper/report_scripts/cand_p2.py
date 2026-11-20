"""論文2：本文・付録の文面と v1 の違いの候補を1つずつ入れて比べる。引数: 出力 json
候補
 ma_mode="bid"     : 本文 p.4「同じサイトを狙う個体が複数いたら、そのサイトへの可能的遷移の向きが基本ベクトルに
                     一番近い個体が入り、他の個体は2番目に人気のサイトへ動く」。全員が同時に1番人気を狙う形
                     （v1 はランダム順に1体ずつ決め、先に決めた個体が取る）
 ma_mode="poporder": 本文 p.4「サイトの人気度の分布が更新の順番を決める」。狙うサイトの人気度が高い個体から順に更新
 整列度に「留まった個体（速度0）も平均に入れる」版を同時に測る（本文「平均単位速度のノルム」）
"""
import sys, json, itertools
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, r"C:\Users\PC_User\ClaudeCode\研究")
from model.MAM_basic_v1 import SoldierCrabSwarm_v1, polarization2, density_paper2


class Cand2(SoldierCrabSwarm_v1):
    def __init__(self, ma_mode="sequential", **kw):
        self.ma_mode = ma_mode
        super().__init__(**kw)

    def _step_paper2(self):
        if self.ma_mode == "sequential":
            return super()._step_paper2()
        N, S, c = self.N, self.S, self.c
        self.w_ma[:, :] = 0
        targets, vecs = self._generate_potential_transitions()
        z = self._compute_popularity(targets, dedupe=self.dedupe_votes)
        self._pv_int = vecs[:, 0, :].copy()
        a = np.arctan2(vecs[:, :, 1], vecs[:, :, 0])
        ang = np.abs((a - a[:, :1] + np.pi) % (2 * np.pi) - np.pi)
        ang[np.all(vecs == 0, axis=2)] = np.pi          # (0,0) は向きなし
        vacant0 = self.occupancy == -1
        occ = self.occupancy.copy(); newp = self.position.copy(); cnt = [0, 0, 0, 0]
        taken = np.zeros((S, S), bool); wand = np.zeros(N, bool)

        def move(k, nx, ny):
            x0, y0 = self.position[k]
            occ[x0, y0] = -1; occ[nx, ny] = k; newp[k] = (nx, ny); taken[nx, ny] = True

        # 各個体の候補サイト：人気度 ≥ c の空きサイトを人気度の高い順（同率は基本ベクトルに近い順）
        ranked = []
        for k in range(N):
            tx, ty = targets[k, :, 0], targets[k, :, 1]
            ok = vacant0[tx, ty] & (z[tx, ty] >= c)
            best = {}
            for i in np.flatnonzero(ok):
                s = (int(tx[i]), int(ty[i]))
                if s not in best or ang[k, i] < best[s]:
                    best[s] = ang[k, i]
            items = sorted(best.items(), key=lambda kv: (-z[kv[0]], kv[1], self.rng.random()))
            ranked.append(items)

        if self.ma_mode == "bid":
            bidders = list(range(N))
            for rank in (0, 1):
                bids = {}
                for k in bidders:
                    if len(ranked[k]) > rank and not taken[ranked[k][rank][0]]:
                        bids.setdefault(ranked[k][rank][0], []).append((ranked[k][rank][1], self.rng.random(), k))
                bidders = []
                for site, ks in bids.items():
                    ks.sort()
                    w = ks[0][2]
                    self.w_ma[tuple(self.position[w])] = 1
                    move(w, *site); wand[w] = True; cnt[0] += 1
                    bidders += [k for _, _, k in ks[1:]]
        else:  # poporder：狙うサイトの人気度が高い個体から
            key = [(-(z[r[0][0]] if r else 0), self.rng.random(), k) for k, r in enumerate(ranked)]
            for _, _, k in sorted(key):
                for site, _ in ranked[k]:
                    if not taken[site]:
                        self.w_ma[tuple(self.position[k])] = 1
                        move(k, *site); wand[k] = True; cnt[0] += 1
                        break

        order = [int(k) for k in self.rng.permutation(N) if not wand[k]]
        rest = []
        for k in order:
            tx, ty = targets[k, :, 0], targets[k, :, 1]
            ok = (self.w_ma[tx, ty] == 1) & (occ[tx, ty] == -1)
            if ok.any():
                s = int(self.rng.choice(np.unique(tx[ok] * S + ty[ok]))); move(k, s // S, s % S); cnt[1] += 1
            else:
                rest.append(k)
        for k in rest:
            tx, ty = targets[k, :, 0], targets[k, :, 1]
            ok = occ[tx, ty] == -1
            if ok.any():
                s = int(self.rng.choice(np.unique(tx[ok] * S + ty[ok]))); move(k, s // S, s % S); cnt[2] += 1
            else:
                cnt[3] += 1
        self.position = newp; self.occupancy = occ
        return tuple(cnt)


def pol_all(d):
    d = d.astype(float); n = np.linalg.norm(d, axis=1); u = np.zeros_like(d); mv = n > 0
    u[mv] = d[mv] / n[mv, None]
    return float(np.linalg.norm(u.mean(0)))


def one(args):
    tag, kw = args
    m = Cand2(model="paper2", **kw)
    T = 1500
    rec = {k: [] for k in ["pol", "polAll", "dens", "ma", "stay"]}
    for t in range(T):
        m.step()
        rec["pol"].append(polarization2(m.last_displacement)); rec["polAll"].append(pol_all(m.last_displacement))
        rec["dens"].append(density_paper2(m.position, m.S))
        c = m.last_rule_counts; rec["ma"].append(c[0] / m.N); rec["stay"].append(c[3] / m.N)
    out = dict(tag=tag, alpha=kw["alpha"], P=kw["P"], seed=kw["seed"])
    for k, v in rec.items():
        v = np.array(v); out[k] = float(v[1000:].mean()); out[k + "_ts"] = [float(x) for x in v.reshape(-1, 50).mean(1)]
    return out


CONFIGS = {
    "v1": {}, "v1_principal": dict(omega_on="principal"),
    "bid": dict(ma_mode="bid"), "bid_principal": dict(ma_mode="bid", omega_on="principal"),
    "poporder": dict(ma_mode="poporder"), "poporder_principal": dict(ma_mode="poporder", omega_on="principal"),
}

if __name__ == "__main__":
    jobs = []
    for (tag, kw), s in itertools.product(CONFIGS.items(), range(3)):
        for a in [0.1, 0.2, 0.3, 0.4, 0.5]:
            jobs.append((tag, dict(P=40, alpha=a, seed=s, **kw)))
        jobs.append((tag, dict(P=10, alpha=0.1, seed=s, **kw)))
    jobs.sort(key=lambda j: -j[1]["P"])
    with Pool(20) as pool:
        res = pool.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
