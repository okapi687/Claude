"""論文1：論文の文面と v1 の違いの候補を1つずつ入れて比べる。引数: 出力 json
候補
 order="agent"   : 本文 p.9-10 の読み方。ランダム順に1体ずつ、(4)相互予期→(5)追従→(6)自由移動 をその場で判定
                   （v1 は図2の説明どおり「全員の相互予期→全員の追従→全員の自由移動」）
 follow_flag     : 追従先。"w" = 式(5)どおり更新済みの個体が空けたサイト全部、"w_ma" = 相互予期で空いたサイトだけ（v1）
 R_Nf=5          : 本文「追従者近傍の半径は基本ベクトルの長さより大きい」（L=4 → 5）。図の R(Nf)=2 と矛盾
整列度は複数の定義で同時に測る
"""
import sys, json, itertools
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, r"C:\Users\PC_User\ClaudeCode\研究")
from model.MAM_basic_v1 import SoldierCrabSwarm_v1, density_sliding


class Cand(SoldierCrabSwarm_v1):
    def __init__(self, order="phase", follow_flag=None, **kw):
        self.order, self.follow_flag = order, follow_flag
        super().__init__(**kw)

    def _step_paper1(self):
        if self.order == "phase":
            return super()._step_paper1()
        N, S = self.N, self.S
        self.w[:, :] = 0; self.w_ma[:, :] = 0
        targets, _ = self._generate_potential_transitions()
        z = self._compute_popularity(targets, dedupe=True)
        occ = self.occupancy.copy(); newp = self.position.copy(); cnt = [0, 0, 0, 0]
        flag = self.w_ma if self.follow_flag == "w_ma" else self.w
        for k in self.rng.permutation(N):
            x0, y0 = self.position[k]; kt = targets[k]
            dest = self._best_MA_target(kt, z, occ); kind = 0
            if dest is None:
                nbx = np.mod(x0 + self._dx_Nf, S); nby = np.mod(y0 + self._dy_Nf, S)
                ok = (flag[nbx, nby] == 1) & (occ[nbx, nby] == -1)
                if ok.any():
                    i = self.rng.choice(np.flatnonzero(ok)); dest = (nbx[i], nby[i]); kind = 1
            if dest is None:
                vac = occ[kt[:, 0], kt[:, 1]] == -1
                if vac.any():
                    dest = kt[self.rng.choice(np.flatnonzero(vac))]; kind = 2
                else:
                    dest = (x0, y0); kind = 3
            nx, ny = dest
            occ[x0, y0] = -1; occ[nx, ny] = k
            if (nx, ny) != (x0, y0):
                self.w[x0, y0] = 1
                if kind == 0:
                    self.w_ma[x0, y0] = 1
            newp[k] = (nx, ny); cnt[kind] += 1
        self.position = newp; self.occupancy = occ
        return tuple(cnt)


def density_grid(pos, S, Ls):
    """本文どおり：空間を Ls×Ls の区画に分け、区画内の最大個体数 / N"""
    n = S // Ls
    h = np.zeros((n, n), int)
    np.add.at(h, (np.minimum(pos[:, 0] // Ls, n - 1), np.minimum(pos[:, 1] // Ls, n - 1)), 1)
    return h.max() / len(pos)


def pols(d, L):
    d = d.astype(float); nrm = np.linalg.norm(d, axis=1); mv = nrm > 0
    m = d.mean(0)
    unit = np.linalg.norm((d[mv] / nrm[mv, None]).mean(0)) if mv.any() else 0.0
    return dict(pL=np.linalg.norm(m) / L,                                   # v1：|平均変位| / L
                pUnit=unit,                                                  # 単位変位の平均の長さ
                pStep=np.linalg.norm(m) / nrm.mean() if nrm.mean() > 0 else 0.0,  # |平均変位| / 平均の歩幅
                pDebias=np.linalg.norm(m + 0.5) / L)                         # floor の (-0.5,-0.5) を足し戻す


def one(args):
    tag, kw = args
    m = Cand(model="paper1", **kw)
    T = 900
    rec = {k: [] for k in ["pL", "pUnit", "pStep", "pDebias", "phi", "dens", "densG", "ma", "fol", "free"]}
    for t in range(T):
        th0 = m.theta.copy(); m.step()
        for k, v in pols(m.last_displacement, m.L).items():
            rec[k].append(v)
        rec["phi"].append(abs(np.mean(np.exp(1j * th0))))
        rec["dens"].append(density_sliding(m.position, m.S, 10)); rec["densG"].append(density_grid(m.position, m.S, 10))
        c = m.last_rule_counts
        rec["ma"].append(c[0] / m.N); rec["fol"].append(c[1] / m.N); rec["free"].append(c[2] / m.N)
    out = dict(tag=tag, alpha=kw["alpha"], P=kw["P"], seed=kw["seed"])
    for k, v in rec.items():
        v = np.array(v)
        out[k] = float(v[300:].mean()); out[k + "_early"] = float(v[:100].mean())
        out[k + "_ts"] = [float(x) for x in v.reshape(-1, 30).mean(1)]
    return out


CONFIGS = {
    "v1": {},
    "agent_w": dict(order="agent", follow_flag="w"),
    "agent_wma": dict(order="agent", follow_flag="w_ma"),
    "v1_RNf5": dict(R_Nf=5),
    "agent_w_RNf5": dict(order="agent", follow_flag="w", R_Nf=5),
    "v1_round": dict(int_mode="round"),
    "agent_w_round": dict(order="agent", follow_flag="w", int_mode="round"),
}

if __name__ == "__main__":
    jobs = []
    for (tag, kw), s in itertools.product(CONFIGS.items(), range(5)):
        for a in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            jobs.append((tag, dict(P=15, alpha=a, seed=s, **kw)))
        for P in [1, 2, 5, 30]:
            jobs.append((tag, dict(P=P, alpha=0.7, seed=s, **kw)))
    jobs.sort(key=lambda j: -j[1]["P"])
    with Pool(20) as pool:
        res = pool.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
