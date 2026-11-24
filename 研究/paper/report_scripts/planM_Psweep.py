"""計画M：α 文面どおりの一番よい版（移動で速度一致・人気度は遷移の本数）を、P=1〜30 で図10・図11 と比べる（11/24）
比べる相手：候補B（移動で速度一致・1個体1票）。どちらも α そのまま・round・sequential
図10 右は α=0.7、横軸は P/30。図11 は α=0.3〜0.6。密度は区画 Ls=25（図11 で合った測り方）と Ls=10
使い方: python planM_Psweep.py [出力 json]   （既定は report_data/planM_Psweep.json）
"""
import sys, json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # 研究/
from model.MAM_paper1 import CrabSwarmPaper1, polarization1, density_grid  # noqa: E402

SETS = {
    "disp_tr": dict(vm_source="disp", popularity="transition"),     # 計画L で一番よかった版
    "disp_ind": dict(vm_source="disp", popularity="individual"),    # 候補B
}
COMMON = dict(alpha_scale=1, int_mode="round", ma_rule="sequential")
PS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30)
ALPHAS = (0.3, 0.4, 0.5, 0.6, 0.7)
STEPS, BURN = 900, 300


def run(job):
    tag, P, alpha, seed = job
    m = CrabSwarmPaper1(P=P, alpha=alpha, seed=seed, **COMMON, **SETS[tag])
    pol, d10, d25, rules = [], [], [], []
    for t in range(STEPS):
        m.step()
        if t < BURN:
            continue
        pol.append(polarization1(m.last_displacement, m.L))
        d10.append(density_grid(m.position, m.S, 10))
        d25.append(density_grid(m.position, m.S, 25))
        rules.append(np.array(m.last_rule_counts) / m.N)
    r = np.mean(rules, 0)
    return dict(tag=tag, P=P, alpha=alpha, seed=seed, pol=float(np.mean(pol)), pol_sd=float(np.std(pol)),
                dens10=float(np.mean(d10)), dens25=float(np.mean(d25)),
                ma=float(r[0]), fol=float(r[1]), free=float(r[2]))


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "report_data" / "planM_Psweep.json"
    jobs = [(tag, P, a, s) for tag in SETS for P in PS for a in ALPHAS for s in range(3)]
    jobs.sort(key=lambda j: -j[1])                 # P が大きい（遅い）ものから
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        res = list(ex.map(run, jobs))
    json.dump(res, open(out, "w", encoding="utf-8"))
    print("saved", out, len(res))
