"""計画N：論文2（本文＋付録）のすべての図を、ベストの設定（α 2倍・round・ω 式どおり・1個体1票）で回す（11/24）
先生の答え（11/24）：α 2倍・round は、論文に合うならそれで進めてよい。

回すもの（どれも N=100（2体・3体は除く）、50×50、L=6、ω=5、PRESETS["best"]）
  grid  : 図3a  P=10/20/30/40 × α=0.1〜0.5、5シード、1500 ステップ。seed 0 は最後の4ステップの位置も保存
  time  : 図3b,c・図S2a  青・赤・緑 × 10シード、1500 ステップの時系列（整列度・密度・ルールの個体数）
  omega : 図S3  P=40・α=0.2・ω=0〜10、10シード、1500 ステップ
  pair  : 図4b・図S2b  N=2、P=40・α=0.2、30000 ステップ、5シード（位置とルール）
  trio  : 図S4（付録の説明では Figure S3）  N=3、同じ条件、30000 ステップ、5シード
使い方: python planN_paper2_sim.py [出力フォルダ]   （既定は report_data/planN/）
"""
import sys, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # 研究/
from model.MAM_paper2 import CrabSwarmPaper2, PRESETS, CONDITIONS, polarization2, density_paper2  # noqa: E402

BEST = PRESETS["best"]
T_MAIN, T_PAIR = 1500, 30000


def run_main(job):
    kind, key, P, alpha, omega, seed = job
    m = CrabSwarmPaper2(P=P, alpha=alpha, omega=omega, seed=seed, **BEST)
    pol, dens, rules, last = [], [], [], []
    for t in range(T_MAIN):
        m.step()
        pol.append(polarization2(m.last_displacement))
        dens.append(density_paper2(m.position, m.S))
        rules.append(m.last_rule_counts)
        if t >= T_MAIN - 4:
            last.append(m.position.copy())
    return dict(kind=kind, key=key, P=P, alpha=alpha, omega=omega, seed=seed,
                pol=np.array(pol, np.float32), dens=np.array(dens, np.float32),
                rules=np.array(rules, np.int16), last=np.array(last, np.int16))


def run_small(job):
    kind, N, seed = job
    m = CrabSwarmPaper2(N=N, P=40, alpha=0.2, seed=seed, **BEST)
    pos = np.zeros((T_PAIR + 1, N, 2), np.int16)
    pos[0] = m.position
    rules = np.zeros(4)
    for t in range(T_PAIR):
        m.step()
        pos[t + 1] = m.position
        rules += m.last_rule_counts
    return dict(kind=kind, seed=seed, pos=pos, rules=rules / (N * T_PAIR))


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "report_data" / "planN"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [("grid", f"P{P}_a{a}", P, a, 5.0, s) for P in (10, 20, 30, 40) for a in (0.1, 0.2, 0.3, 0.4, 0.5) for s in range(5)]
    jobs += [("time", name, c["P"], c["alpha"], 5.0, s) for name, c in CONDITIONS.items() for s in range(10)]
    jobs += [("omega", f"w{w}", 40, 0.2, float(w), s) for w in range(11) for s in range(10)]
    small = [("pair", 2, s) for s in range(5)] + [("trio", 3, s) for s in range(5)]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        fs = [ex.submit(run_small, j) for j in small]           # 長いものを先に
        res = list(ex.map(run_main, jobs, chunksize=2))
        sm = [f.result() for f in fs]
    for kind in ("grid", "time", "omega"):
        R = [r for r in res if r["kind"] == kind]
        np.savez_compressed(out / f"{kind}.npz",
                            meta=np.array([(r["key"], r["P"], r["alpha"], r["omega"], r["seed"]) for r in R], dtype=object),
                            pol=np.stack([r["pol"] for r in R]), dens=np.stack([r["dens"] for r in R]),
                            rules=np.stack([r["rules"] for r in R]), last=np.stack([r["last"] for r in R]))
    for kind in ("pair", "trio"):
        R = [r for r in sm if r["kind"] == kind]
        np.savez_compressed(out / f"{kind}.npz", pos=np.stack([r["pos"] for r in R]),
                            rules=np.stack([r["rules"] for r in R]), seeds=np.array([r["seed"] for r in R]))
    print("saved", out)
