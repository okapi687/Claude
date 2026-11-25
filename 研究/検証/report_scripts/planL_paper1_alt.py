"""計画L：論文1 を α 文面どおりで読んだときの、ほかの現実的な読み方を比べる（11/24）
- どれも α そのまま・round・sequential（p.6 の A,B,C の例どおり）
- 変えるのは、速度一致で平均するもの（vm_source）と、人気度の数え方（popularity）
  popularity="transition" は p.7「the number of transitions reaching that site is counted as the popularity」の文面どおり
使い方: python planL_paper1_alt.py [出力 json]   （既定は report_data/planL_paper1_alt.json）
"""
import sys, json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # 研究/
from model.MAM_paper1 import CrabSwarmPaper1, polarization1, density_grid  # noqa: E402

SETS = {
    "theta_ind": dict(vm_source="theta", popularity="individual"),
    "disp_ind": dict(vm_source="disp", popularity="individual"),        # 候補B
    "theta_tr": dict(vm_source="theta", popularity="transition"),
    "disp_tr": dict(vm_source="disp", popularity="transition"),
    "unit_tr": dict(vm_source="unit_disp", popularity="transition"),
}
COMMON = dict(alpha_scale=1, int_mode="round", ma_rule="sequential")
STEPS, BURN = 900, 300


def run(job):
    tag, P, alpha, seed = job
    m = CrabSwarmPaper1(P=P, alpha=alpha, seed=seed, **COMMON, **SETS[tag])
    pol, phi, rules = [], [], []
    dens = {Ls: [] for Ls in (5, 6, 10, 25)}
    for t in range(STEPS):
        m.step()
        if t < BURN:
            continue
        pol.append(polarization1(m.last_displacement, m.L))
        phi.append(float(np.hypot(np.cos(m.theta).mean(), np.sin(m.theta).mean())))
        rules.append(np.array(m.last_rule_counts) / m.N)
        for Ls in dens:
            dens[Ls].append(density_grid(m.position, m.S, Ls))
    r = np.mean(rules, 0)
    return dict(tag=tag, P=P, alpha=alpha, seed=seed, pol=float(np.mean(pol)), pol_sd=float(np.std(pol)),
                phi=float(np.mean(phi)), ma=float(r[0]), fol=float(r[1]), free=float(r[2]),
                **{f"dens{Ls}": float(np.mean(v)) for Ls, v in dens.items()})


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "report_data" / "planL_paper1_alt.json"
    jobs = [(tag, P, a, s) for tag in SETS for P in (10, 15, 20)
            for a in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) for s in range(3)]
    jobs += [(tag, P, a, s) for tag in SETS for P in (1, 2) for a in (0.5, 0.7) for s in range(3)]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        res = list(ex.map(run, jobs, chunksize=2))
    json.dump(res, open(out, "w", encoding="utf-8"))
    print("saved", out, len(res))
