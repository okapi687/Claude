"""計画M：論文1 図8 と同じ P × α の格子のスナップショット（11/24）
α 文面どおりの一番よい版（移動で速度一致・人気度は遷移の本数、round・sequential）。
図8 の条件：N=100、50×50、L=4、R(Nf)=2、R(M)=3。P = 1, 2, 5, 10, 20、α = 0.3, 0.5, 0.7, 0.9
使い方: python planM_fig8.py   （図は report_fig/fig35_planM_fig8.png）
"""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # 研究/
from model.MAM_paper1 import CrabSwarmPaper1, polarization1, torus_diff  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
OUT = HERE.parent / "report_fig" / "fig35_planM_fig8.png"
PS = (1, 2, 5, 10, 20)
ALPHAS = (0.3, 0.5, 0.7, 0.9)
KW = dict(alpha_scale=1, int_mode="round", ma_rule="sequential", vm_source="disp", popularity="transition")
STEPS, TAIL = 600, 5


def run(job):
    P, a = job
    m = CrabSwarmPaper1(P=P, alpha=a, seed=0, **KW)
    hist = m.run(STEPS, record_history=True)
    pol = []
    for t in range(STEPS - 100, STEPS):
        pol.append(polarization1(torus_diff(hist[t + 1] - hist[t], m.S), m.L))
    return P, a, hist[-TAIL - 1:], float(np.mean(pol))


if __name__ == "__main__":
    jobs = [(P, a) for a in ALPHAS for P in PS]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = {(P, a): (h, p) for P, a, h, p in ex.map(run, jobs)}
    fig, axes = plt.subplots(len(ALPHAS), len(PS), figsize=(2.3 * len(PS), 2.45 * len(ALPHAS)))
    S = 50
    for i, a in enumerate(ALPHAS):
        for j, P in enumerate(PS):
            ax = axes[i, j]
            h, p = res[(P, a)]
            for k in range(h.shape[1]):              # 軌跡（トーラスをまたぐ所は描かない）
                tr = h[:, k, :]
                for t in range(len(tr) - 1):
                    if np.abs(tr[t + 1] - tr[t]).max() < S / 2:
                        ax.plot(tr[t:t + 2, 0], tr[t:t + 2, 1], color="0.6", lw=0.6)
            ax.scatter(h[-1, :, 0], h[-1, :, 1], s=7, marker="s", color="k")
            ax.set_xlim(-0.5, S - 0.5); ax.set_ylim(-0.5, S - 0.5); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"整列度 {p:.2f}", fontsize=8, color="0.3")
            if i == 0:
                ax.set_title(f"P={P}")
            if j == 0:
                ax.set_ylabel(f"α={a}", fontsize=11)
    fig.suptitle("論文1 図8 と同じ格子：α 文面どおり・round・移動で速度一致・人気度は遷移の本数（seed 0、t=600、灰色は最後の5ステップ、整列度は最後の100ステップの平均）",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(OUT, dpi=110)
    print("fig:", OUT)
