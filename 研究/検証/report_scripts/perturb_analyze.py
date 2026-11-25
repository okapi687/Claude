import json, sys
from collections import defaultdict
import numpy as np

res = json.load(open(sys.argv[1]))
g = defaultdict(lambda: defaultdict(list))
for r in res:
    g[r["tag"]][(r["alpha"], r["eps"])].append(r)
for tag in ["p1_noise", "p30_noise", "arith", "ma_noise"]:
    print(f"== {tag}")
    for k in sorted(g[tag]):
        rs = g[tag][k]; m = lambda q: np.mean([r[q] for r in rs])
        print(f"  α={k[0]:.1f} ε={k[1]:.1f} pol={m('pol'):.2f}±{np.std([r['pol'] for r in rs]):.2f} phi={m('phi'):.2f} fwd={m('fwd'):.2f} dens={m('dens'):.2f} MA={m('ma')*100:.0f}%")
