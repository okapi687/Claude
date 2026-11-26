"""計画T（11/27）：郡司研の動画（本『群れは意識をもつ』の図3-8）と同じ条件（S=100、N=200・P=10 / N=400・P=15、L=4）で
論文1 のモデルを回し、動画から測ったのと同じ量を出す。
  整列度：移動の単位ベクトルの平均の長さ（1ステップごと）
  離れた個体（ほかの個体から 8 マス以上）の歩幅と、留まる割合
  塊の数と大きさ（チェビシェフ距離 2 以内でつながるもの）
速度一致は 移動で平均（今の一番よい版）と θ で平均（式(8) の記号どおり）。t=130・680・1220 から 40 ステップずつ、2シード。
使い方: python planT_video_model.py   出力は ../report_data/planT_video_model.json
"""
import sys, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # 研究/
from model.MAM_paper1 import CrabSwarmPaper1  # noqa: E402

S = 100
BASE = dict(alpha_scale=1, int_mode="round", ma_rule="sequential", popularity="transition")
SETS = {"移動で速度一致": dict(vm_source="disp"), "θで速度一致": dict(vm_source="theta")}


def pdist(a, b):
    d = (b[None, :, :] - a[:, None, :]).astype(float)
    d -= S * np.round(d / S)
    return d


def metrics(pos_prev, pos, disp):
    l = np.linalg.norm(disp, axis=1); mv = l > 0
    pol = np.linalg.norm((disp[mv] / l[mv, None]).sum(0)) / len(disp)
    d = pdist(pos_prev, pos_prev); dd = np.hypot(d[..., 0], d[..., 1]); np.fill_diagonal(dd, 1e9)
    iso = dd.min(1) >= 8
    cheb = np.abs(pdist(pos, pos)).max(-1) <= 2
    n = len(pos); seen = np.zeros(n, bool); comps = []
    for i in range(n):
        if seen[i]:
            continue
        st = [i]; seen[i] = True; sz = 0
        while st:
            j = st.pop(); sz += 1
            for k in np.flatnonzero(cheb[j] & ~seen):
                seen[k] = True; st.append(k)
        comps.append(sz)
    comps.sort(reverse=True)
    return pol, l[iso], (l == 0).mean(), len(comps), comps[:5]


def run(job):
    name, N, P, alpha, seed = job
    m = CrabSwarmPaper1(N=N, S=S, L=4.0, P=P, alpha=alpha, seed=seed, **BASE, **SETS[name])
    out = []
    marks = {130: [], 680: [], 1220: []}
    for t in range(1, 1261):
        prev = m.position.copy()
        m.step()
        for t0 in marks:
            if t0 <= t < t0 + 40:
                marks[t0].append(metrics(prev, m.position.copy(), m.last_displacement.astype(float)))
    for t0, ms in marks.items():
        pol = np.mean([x[0] for x in ms]); iso = np.concatenate([x[1] for x in ms])
        out.append(dict(t0=t0, pol=round(float(pol), 2), stayAll=round(float(np.mean([x[2] for x in ms])), 2),
                        stayIso=round(float((iso == 0).mean()) if len(iso) else -1, 2),
                        lenIso=round(float(iso.mean()) if len(iso) else -1, 2),
                        clusters=ms[-1][3], big=ms[-1][4]))
    return dict(set=name, N=N, P=P, alpha=alpha, seed=seed, seg=out)


if __name__ == "__main__":
    jobs = [(s, N, P, a, sd) for s in SETS for (N, P) in ((200, 10), (400, 15)) for a in (0.3, 0.5, 0.7) for sd in (0, 1)]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run, jobs))
    (HERE.parent / "report_data" / "planT_video_model.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in res:
        print(r["set"], f"N={r['N']} P={r['P']} α={r['alpha']} s{r['seed']}", " | ".join(
            f"t{x['t0']}: 整列度 {x['pol']} 留まる {x['stayAll']}/{x['stayIso']} 歩幅 {x['lenIso']} 塊 {x['clusters']} {x['big'][:3]}"
            for x in r["seg"]))
