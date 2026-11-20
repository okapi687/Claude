"""ゼミの指摘への対応（論文2）。引数: 出力 json
 sweep : P=40 で α を 0.05〜1.0 まで細かく振る（ω 式どおり / v1）。ルールの割合と「留まる」も記録
 grid  : 図3a と同じ P×α の格子（alpha_scale 1 / 2、ω 式どおり）。seed 0 は最後の4ステップの位置を保存
 pair  : 2体のシミュレーション（図4・図S2b、P=40, α=0.2, 30000 ステップ）。整列度・ルールの割合・社会的な力
"""
import sys, json, itertools
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, r"C:\Users\PC_User\ClaudeCode\研究")
from model.MAM_basic_v1 import SoldierCrabSwarm_v1, polarization2, density_paper2


def run_swarm(args):
    kind, tag, kw = args
    m = SoldierCrabSwarm_v1(model="paper2", **kw)
    T = 1500
    rec = {k: [] for k in ["pol", "dens", "ma", "fol", "free", "stay", "zero"]}
    hist = []
    for t in range(T):
        m.step()
        d = m.last_displacement
        rec["pol"].append(polarization2(d)); rec["dens"].append(density_paper2(m.position, m.S))
        c = m.last_rule_counts
        for k, v in zip(["ma", "fol", "free", "stay"], c):
            rec[k].append(v / m.N)
        rec["zero"].append(float(np.mean(np.all(d == 0, axis=1))))   # 変位が 0 の個体の割合
        if t >= T - 4:
            hist.append(m.position.tolist())
    out = dict(kind=kind, tag=tag, **{k: kw[k] for k in ["P", "alpha", "seed"]}, scale=kw.get("alpha_scale", 1))
    for k, v in rec.items():
        v = np.array(v); out[k] = float(v[1000:].mean()); out[k + "_ts"] = [float(x) for x in v.reshape(-1, 50).mean(1)]
    if kw["seed"] == 0:
        out["hist"] = hist
    return out


def run_pair(args):
    kind, tag, kw = args
    m = SoldierCrabSwarm_v1(model="paper2", N=2, P=40, alpha=0.2, **kw)
    T, S = 30000, m.S
    pos = np.zeros((T + 1, 2, 2), int); pos[0] = m.position
    rules = np.zeros(4)
    for t in range(T):
        m.step(); pos[t + 1] = m.position; rules += m.last_rule_counts
    V = (np.diff(pos, axis=0) + S // 2) % S - S // 2              # (T, 2, 2) 各ステップの速度
    n = np.linalg.norm(V, axis=2)
    u = np.where(n[..., None] > 0, V / np.maximum(n, 1e-9)[..., None], 0)
    pol = np.linalg.norm(u.mean(1), axis=1)
    dist_vec = (pos[:, 1] - pos[:, 0] + S // 2) % S - S // 2
    dist = np.linalg.norm(dist_vec, axis=1)
    # 社会的な力（Katz et al. と同じ考え方）：a = V(t) - V(t-1)、向きの基準は V(t-1)
    LR, FB, RH, TF, SF = [], [], [], [], []
    for f, g in [(0, 1), (1, 0)]:
        v0 = V[:-1, f].astype(float); a = (V[1:, f] - V[:-1, f]).astype(float)
        ok = np.linalg.norm(v0, axis=1) > 0
        h = np.zeros_like(v0); h[ok] = v0[ok] / np.linalg.norm(v0[ok], axis=1)[:, None]
        nrm = np.stack([-h[:, 1], h[:, 0]], 1)                       # 左向きの法線
        r = ((pos[1:-1, g] - pos[1:-1, f] + S // 2) % S - S // 2).astype(float)   # 時刻 t の相対位置
        vg = V[:-1, g].astype(float)
        ok &= np.linalg.norm(vg, axis=1) > 0
        rh = np.arctan2(h[:, 0] * vg[:, 1] - h[:, 1] * vg[:, 0], (h * vg).sum(1))
        LR.append((r * nrm).sum(1)[ok]); FB.append((r * h).sum(1)[ok]); RH.append(rh[ok])
        TF.append((a * nrm).sum(1)[ok]); SF.append((a * h).sum(1)[ok])
    LR, FB, RH, TF, SF = map(np.concatenate, (LR, FB, RH, TF, SF))
    near = (np.abs(LR) <= 6) & (np.abs(FB) <= 6)
    bins = np.arange(-6.5, 7.5, 1.0)
    hb = np.linspace(-np.pi, np.pi, 9)

    def binned(x, y, b):
        idx = np.digitize(x, b) - 1
        return [float(y[idx == i].mean()) if np.sum(idx == i) >= 30 else None for i in range(len(b) - 1)]

    tf_map = []
    for i in range(len(hb) - 1):
        sel = near & (RH >= hb[i]) & (RH < hb[i + 1])
        tf_map.append(binned(LR[sel], TF[sel], bins))
    sf_map = []
    for i in range(len(hb) - 1):
        sel = near & (RH >= hb[i]) & (RH < hb[i + 1])
        sf_map.append(binned(FB[sel], SF[sel], bins))
    return dict(kind=kind, tag=tag, seed=kw["seed"], scale=kw.get("alpha_scale", 1),
                pol=float(pol[1000:].mean()), dist=float(dist[1000:].mean()),
                near_frac=float(np.mean(dist[1000:] <= 6)),
                ma=rules[0] / (2 * T), fol=rules[1] / (2 * T), free=rules[2] / (2 * T), stay=rules[3] / (2 * T),
                tf_lr=binned(LR[near], TF[near], bins), tf_rh=binned(RH[near], TF[near], hb),
                sf_fb=binned(FB[near], SF[near], bins), tf_map=tf_map, sf_map=sf_map,
                pol_ts=[float(x) for x in pol[:30000].reshape(-1, 500).mean(1)])


def one(job):
    return run_pair(job) if job[0] == "pair" else run_swarm(job)


OMEGA = {"principal": dict(omega_on="principal"), "velocity": dict(omega_on="velocity")}

if __name__ == "__main__":
    jobs = []
    A = [round(0.05 * i, 2) for i in range(1, 21)]
    for (tag, kw), a, s in itertools.product(OMEGA.items(), A, range(3)):
        jobs.append(("sweep", tag, dict(P=40, alpha=a, seed=s, **kw)))
    for (tag, kw), a, s in itertools.product(OMEGA.items(), [0.1, 0.2], range(3)):
        jobs.append(("sweep", tag, dict(P=10, alpha=a, seed=s, **kw)))
    for P, a, sc, s in itertools.product([10, 20, 30], [0.1, 0.2, 0.3, 0.4, 0.5], [1, 2], range(2)):
        jobs.append(("grid", "principal", dict(P=P, alpha=a, alpha_scale=sc, seed=s, omega_on="principal")))
    for (otag, okw), sc, dd, s in itertools.product(OMEGA.items(), [1, 2], [True, False], range(3)):
        tag = f"{otag}_x{sc}_{'dedupe' if dd else 'count'}"
        jobs.append(("pair", tag, dict(alpha_scale=sc, dedupe_votes=dd, seed=s, **okw)))
    jobs.sort(key=lambda j: (j[0] != "pair", -j[2].get("P", 0)))
    with Pool(22) as pool:
        res = pool.map(one, jobs, chunksize=1)
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res))
