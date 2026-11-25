"""計画O（11/25）：論文2 で向き θ の揺らぎがどこから来ているかを分ける診断。
付録の式(12)(13)：θ(t+1) = angle(V_u + ω·v_u(k,t;0))。v_u(k,t;0) は式(2) の整数ベクトル（の単位ベクトル）。
1ステップの θ の変化を、
  丸め：θ(t) → 整数ベクトル v(k,t;0) の向き（式(2) の Int）
  移動：整数ベクトルの向き → θ(t+1)（式(13) の V_u の寄与）
に分け、それぞれの大きさ（二乗平均の平方根、度）を出す。扇＝実際の移動 V の向きと θ(t) の差。
比べるのは ベスト（α 2倍）と α 1倍 × v_u を整数ベクトル（式どおり）にするか (cosθ, sinθ) にするか。
青・赤・緑 × 3シード、1500 ステップ（t=500 以降を集計）、ω=5。
使い方: python planO_theta_diag.py [出力のjson]   既定は ../report_data/planO_theta.json
"""
import sys, json, collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # 研究/
from model.MAM_paper2 import CrabSwarmPaper2, PRESETS, CONDITIONS, polarization2  # noqa: E402


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run(job):
    name, preset, spv, seed = job
    m = CrabSwarmPaper2(**CONDITIONS[name], omega=5.0, seed=seed, **PRESETS[preset], spv_mode=spv)
    e_snap, e_v, d_tot, fan, pols, lattice = [], [], [], [], [], []
    for t in range(1500):
        th0 = m.theta.copy()
        m.step()
        if t < 500:
            continue
        snap = np.arctan2(m._pv_int[:, 1], m._pv_int[:, 0]) if spv == "int_vector" else th0
        d = m.last_displacement.astype(float)
        mv = np.linalg.norm(d, axis=1) > 0
        e_snap.append(wrap(snap - th0)); e_v.append(wrap(m.theta - snap)); d_tot.append(wrap(m.theta - th0))
        fan.append(wrap(np.arctan2(d[mv, 1], d[mv, 0]) - th0[mv]))
        lattice.append(np.mod(m.theta, np.pi / 2))
        pols.append(polarization2(m.last_displacement))
    c = np.concatenate
    sd = lambda x: float(np.degrees(np.sqrt(np.mean(c(x) ** 2))))
    hist = np.histogram(c(lattice), bins=18, range=(0, np.pi / 2))[0]   # θ を 90° で折りたたんだ分布の偏り
    return dict(name=name, preset=preset, spv=spv, seed=seed, sd_snap=sd(e_snap), sd_v=sd(e_v), sd_tot=sd(d_tot),
                sd_fan=sd(fan), pol=float(np.mean(pols)), lattice_ratio=float(hist.max() / hist.mean()),
                MA=m.move_counts["MA"] / sum(m.move_counts.values()))


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "report_data" / "planO_theta.json"
    jobs = [(n, p, s, seed) for n in ("青", "赤", "緑") for p in ("best", "murakami") for s in ("int_vector", "angle")
            for seed in range(3)]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run, jobs))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    g = collections.defaultdict(list)
    for r in res:
        g[(r["name"], r["preset"], r["spv"])].append(r)
    for k, rs in g.items():
        f = lambda key: np.mean([r[key] for r in rs])
        print(*k, f"pol {f('pol'):.2f} MA {100*f('MA'):.1f} 丸め {f('sd_snap'):.1f} 移動 {f('sd_v'):.1f} "
              f"合計 {f('sd_tot'):.1f} 扇 {f('sd_fan'):.0f} 偏り {f('lattice_ratio'):.2f}")
