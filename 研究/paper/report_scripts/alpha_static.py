"""可能的遷移そのものの形が α でどう変わるか（動かさずに調べる）"""
import numpy as np

rng = np.random.default_rng(0)
n = 400000
print("L  Int    α   | (0,0)%  前向き成分の平均 E[cosφ]: 実数 → 整数 | 0でない遷移の向きの広がり（円周標準偏差, 度）実数 → 整数 | 異なる遷移先の数(P本中)")
for L, im in [(4, "floor"), (4, "round"), (6, "round"), (6, "floor")]:
    f = np.floor if im == "floor" else np.round
    for a in [0.1, 0.3, 0.5, 0.7]:
        th = rng.uniform(0, 2 * np.pi, n)
        eta = rng.uniform(0, 1, n); xi = rng.uniform(-a * np.pi, a * np.pi, n)
        cx, cy = L * eta * np.cos(th + xi), L * eta * np.sin(th + xi)
        vx, vy = f(cx), f(cy)
        r = np.hypot(vx, vy); nz = r > 0
        # θ 方向に対する角度
        rel_c = np.angle(np.exp(1j * (np.arctan2(cy, cx) - th)))
        rel_i = np.angle(np.exp(1j * (np.arctan2(vy[nz], vx[nz]) - th[nz])))
        cos_c = np.mean(np.hypot(cx, cy) * np.cos(rel_c)) / L
        cos_i = np.mean(r * np.cos(np.angle(np.exp(1j * (np.arctan2(vy, vx) - th))))) / L
        R_c = np.abs(np.mean(np.exp(1j * rel_c))); R_i = np.abs(np.mean(np.exp(1j * rel_i)))
        sd = lambda R: np.degrees(np.sqrt(-2 * np.log(R)))
        # P=15 本のうち異なるサイト数
        P = 15 if L == 4 else 40
        m = 4000
        th2 = rng.uniform(0, 2 * np.pi, (m, 1)); e2 = rng.uniform(0, 1, (m, P)); x2 = rng.uniform(-a * np.pi, a * np.pi, (m, P))
        v = f(L * e2 * np.cos(th2 + x2)).astype(int) * 100 + f(L * e2 * np.sin(th2 + x2)).astype(int)
        uniq = np.mean([len(np.unique(row)) for row in v])
        print(f"{L}  {im:5s} {a:.1f} | {100 * (1 - nz.mean()):5.1f}  {cos_c:.3f} → {cos_i:.3f}   | {sd(R_c):5.1f} → {sd(R_i):5.1f}   | {uniq:.1f}/{P}")
