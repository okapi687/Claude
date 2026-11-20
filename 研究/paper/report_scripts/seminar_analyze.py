import json, collections
import numpy as np

res = json.load(open("res_seminar_p2.json"))
g = collections.defaultdict(list)
for r in res:
    g[(r["kind"], r["tag"], r.get("P"), r.get("alpha"), r["scale"])].append(r)


def m(rs, k):
    return np.mean([r[k] for r in rs])


print("== sweep P=40 (実際の α) ==")
for tag in ["principal", "velocity"]:
    print(tag)
    for a in [round(0.05 * i, 2) for i in range(1, 21)]:
        rs = g[("sweep", tag, 40, a, 1)]
        print(f"  a={a:.2f} pol={m(rs,'pol'):.2f} ({' '.join(f'{r['pol']:.2f}' for r in rs)}) dens={m(rs,'dens'):.2f} "
              f"ma={m(rs,'ma'):.3f} fol={m(rs,'fol'):.3f} free={m(rs,'free'):.3f} stay={m(rs,'stay'):.4f} zero={m(rs,'zero'):.4f}")
    for a in [0.1, 0.2]:
        rs = g[("sweep", tag, 10, a, 1)]
        print(f"  P=10 a={a} pol={m(rs,'pol'):.2f} dens={m(rs,'dens'):.2f} ma={m(rs,'ma'):.2f} fol={m(rs,'fol'):.2f} free={m(rs,'free'):.2f} stay={m(rs,'stay'):.4f}")

print("== grid (principal) ==")
for sc in [1, 2]:
    print("scale", sc)
    for P in [10, 20, 30, 40]:
        row = []
        for a in [0.1, 0.2, 0.3, 0.4, 0.5]:
            if P == 40:
                rs = g[("sweep", "principal", 40, round(a * sc, 2), 1)]
            else:
                rs = g[("grid", "principal", P, a, sc)]
            row.append(f"{m(rs,'pol'):.2f}/{m(rs,'dens'):.1f}")
        print(f"  P={P}: " + "  ".join(row))

print("== pair ==")
for tag in sorted({r["tag"] for r in res if r["kind"] == "pair"}):
    rs = [r for r in res if r["kind"] == "pair" and r["tag"] == tag]
    print(f"{tag:28s} pol={m(rs,'pol'):.2f} ({' '.join(f'{r['pol']:.2f}' for r in rs)}) dist={m(rs,'dist'):.1f} near={m(rs,'near_frac'):.2f} "
          f"ma={m(rs,'ma'):.2f} fol={m(rs,'fol'):.3f} free={m(rs,'free'):.2f} stay={m(rs,'stay'):.3f}")
    tf = np.nanmean(np.array([[np.nan if v is None else v for v in r["tf_lr"]] for r in rs], float), 0)
    sf = np.nanmean(np.array([[np.nan if v is None else v for v in r["sf_fb"]] for r in rs], float), 0)
    print("   turn vs LR(-6..6):", " ".join(f"{v:+.2f}" for v in tf))
    print("   speed vs FB(-6..6):", " ".join(f"{v:+.2f}" for v in sf))
