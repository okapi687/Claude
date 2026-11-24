"""
MAM_paper2.py — 論文2（Murakami et al. 2017, Sci. Rep. 7:46447 + 付録 S1）のモデル。11/24 に MAM_basic_v1.py から分けた。

MAM_basic_v1.py の model="paper2" の処理を、そのまま移したもの（付録の式(1)〜(13)。速度一致はない）。
乱数を引く順番も同じなので、同じ引数・同じ seed なら MAM_basic_v1 と同じ軌跡になる（確かめ方はファイルの最後）。

1ステップの流れ
  可能的遷移を作る（式(1)〜(3)） → 人気度（式(4)(5)） → 相互予期（式(6)〜(9)） → 追従（式(10)） → 自由移動（式(11)）
  → 基本ベクトルを自分の移動で更新（式(12)(13)）

使い方
    from model.MAM_paper2 import CrabSwarmPaper2, PRESETS, CONDITIONS, simulate
    m = CrabSwarmPaper2(seed=0, **CONDITIONS["赤"], **PRESETS["best"])
    pol, dens, rules = simulate(1500, seed=0, **CONDITIONS["赤"], **PRESETS["best"])

■ 設定の組（PRESETS）
  "text"     : 文面どおり（α そのまま・floor・ω は式(13)どおり）。引数を何も指定しないときと同じ。
               floor のせいで、どの条件でも整列してしまう
  "murakami" : α そのまま・round・ω は式どおり（旧版 MAM_basic_Murakami と同じ）。図S2 の青の 63% が合う
  "best"     : 今のベスト（α 2倍・round・ω は式どおり）。図3・図3a・図S3 の多くが合う（検証レポート 0.3 節）
  文面と違うのは「α 2倍」と「round」（検証レポート 0.4 節）。図に使うときは、どの設定かを必ず書く。

■ 論文2 の図3 の3つの条件（CONDITIONS）
  青 P=10, α=0.1（ばらばら）／ 赤 P=40, α=0.2（一方向に進む群れ）／ 緑 P=40, α=0.5（その場にとどまる塊）
"""
import numpy as np


PRESETS = {
    "text": dict(),
    "murakami": dict(alpha_scale=1, int_mode="round"),
    "best": dict(alpha_scale=2, int_mode="round"),
}
PRESET_NAMES = {"text": "文面どおり（floor）", "murakami": "α そのまま・round", "best": "ベスト（α 2倍・round）"}
CONDITIONS = {
    "青": dict(P=10, alpha=0.1),
    "赤": dict(P=40, alpha=0.2),
    "緑": dict(P=40, alpha=0.5),
}
COLORS = {"青": "tab:blue", "赤": "tab:red", "緑": "tab:green"}


class CrabSwarmPaper2:
    """
    論文2のモデル。

    Parameters
    ----------
    N : int, default 100            個体数
    S : int, default 50             格子の一辺（S×S のトーラス）
    L : float, default 6            基本ベクトルの長さ
    P : int, default 40             可能的遷移の数
    alpha : float, default 0.2      変異角 α（ξ は [-απ, απ] の一様乱数）
    c : int, default 2              人気度の閾値（ζ ≧ c で相互予期が成立）
    omega : float, default 5.0      ω
    seed : int or None              乱数シード

    読み方を切り替えるオプション（既定値が文面どおり）
    alpha_scale : float, default 1.0
        ξ の範囲を [-alpha_scale·απ, alpha_scale·απ] にする。1.0 が付録の式(3)の文面どおり。
        2.0 は「α=0.5 でほぼ全方向」（論文2 p.4）に合わせた読み方
    int_mode : {"floor", "ceil", "round", "trunc", "stochastic"}, default "floor"
        付録の式(2)の Int。文面（X ≤ x < X+1）は floor。floor だと遷移が平均 (−0.5, −0.5) ずれ、どの条件でも整列する
        （MAM_basic_v1 の paper2 の既定は "round" だったので注意）
    omega_on : {"principal", "velocity"}, default "principal"
        "principal" : q = V_u + ω·v_u（付録の式(13)どおり）
        "velocity"  : q = v_u + ω·V_u（図S3 の説明文の読み方。MAM_basic_v1 の既定だった）
    dedupe_votes : bool, default True
        True なら 1個体につき1サイト1票、False なら (個体, 遷移) の組を全部1票と数える
    spv_mode : {"int_vector", "angle"}, default "int_vector"
        式(13) の v_u(k,t;0) を、整数化した基本ベクトルにするか (cosθ, sinθ) にするか
    static_vacancy : bool, default True
        True なら、相互予期の行き先はステップ開始時に空いていたサイトだけ（付録の式(5)どおり）
    tiebreak : {"angle", "random"}, default "angle"
        人気度が同率のとき、基本ベクトルに向きが一番近いものを選ぶか、ランダムに選ぶか

    Attributes
    ----------
    position : (N, 2) int           現在の座標
    theta : (N,) float              基本ベクトルの向き
    last_displacement : (N, 2) int  直前のステップの変位（トーラスをまたいだ分は補正済み）
    last_rule_counts : tuple        直前のステップで 相互予期 / 追従 / 自由移動 / 留まる になった個体数
    move_counts : dict              これまでの累計（キーは "MA", "follow", "free", "stay"）
    """

    INT_MODES = ("floor", "ceil", "round", "trunc", "stochastic")

    def __init__(self, N=100, S=50, L=6.0, P=40, alpha=0.2, c=2, omega=5.0, seed=None,
                 alpha_scale=1.0, int_mode="floor", omega_on="principal", dedupe_votes=True,
                 spv_mode="int_vector", static_vacancy=True, tiebreak="angle"):
        if int_mode not in self.INT_MODES:
            raise ValueError(f"int_mode は {self.INT_MODES} のどれか（指定値: {int_mode!r}）")
        if omega_on not in ("velocity", "principal"):
            raise ValueError(f'omega_on は "velocity" か "principal"（指定値: {omega_on!r}）')
        if spv_mode not in ("int_vector", "angle"):
            raise ValueError(f'spv_mode は "int_vector" か "angle"（指定値: {spv_mode!r}）')
        if tiebreak not in ("angle", "random"):
            raise ValueError(f'tiebreak は "angle" か "random"（指定値: {tiebreak!r}）')
        if N > S * S:
            raise ValueError(f"N({N}) が格子点の総数 S*S({S*S}) を超えています。")

        self.rng = np.random.default_rng(seed)
        self.N, self.S, self.L, self.P, self.alpha, self.c = N, S, L, P, alpha, c
        self.omega, self.seed = omega, seed
        self.alpha_scale, self.int_mode, self.omega_on = alpha_scale, int_mode, omega_on
        self.dedupe_votes, self.spv_mode = dedupe_votes, spv_mode
        self.static_vacancy, self.tiebreak = static_vacancy, tiebreak
        self.t = 0

        # ---- 初期配置：重なりのないランダムな格子点、向きは一様乱数 ----
        all_sites = self.rng.permutation(S * S)[:N]
        self.position = np.stack([all_sites % S, all_sites // S], axis=1).astype(int)
        self.theta = self.rng.uniform(0.0, 2 * np.pi, size=N)

        # occupancy[x, y] = その格子にいる個体の番号、空きなら -1
        self.occupancy = -np.ones((S, S), dtype=int)
        self.occupancy[self.position[:, 0], self.position[:, 1]] = np.arange(N)

        # 付録の χ：wanderer（相互予期で動いた個体）が空けたサイト
        self.w_ma = np.zeros((S, S), dtype=int)

        self.last_displacement = np.zeros((N, 2), dtype=int)
        self.last_rule_counts = (0, 0, 0, 0)
        self.move_counts = {"MA": 0, "follow": 0, "free": 0, "stay": 0}

    # ------------------------------------------------------------------
    # 準備
    # ------------------------------------------------------------------

    def _to_int(self, x):
        """付録の式(2)の Int。int_mode に従って実数を整数にする"""
        if self.int_mode == "floor":
            return np.floor(x).astype(int)
        if self.int_mode == "ceil":
            return np.ceil(x).astype(int)
        if self.int_mode == "round":
            return np.round(x).astype(int)      # 偶数丸め。ちょうど .5 はほぼ出ない
        if self.int_mode == "trunc":
            return np.trunc(x).astype(int)
        lo = np.floor(x)                        # stochastic：小数部分の確率で切り上げ
        return (lo + (self.rng.random(np.shape(x)) < (x - lo))).astype(int)

    def _generate_potential_transitions(self):
        """
        可能的遷移 v(k,t;i) = Int(L·η·(cos(θ+ξ), sin(θ+ξ)))、η∈[0,1]、ξ∈[-απ, απ]（付録の式(1)〜(3)）
        i=0 は基本ベクトル（η=1, ξ=0）。返り値は 遷移先 (N,P,2) と 整数ベクトル (N,P,2)
        """
        N, P, S, L = self.N, self.P, self.S, self.L
        alpha = self.alpha * self.alpha_scale
        eta = self.rng.uniform(0.0, 1.0, size=(N, P))
        xi = self.rng.uniform(-alpha * np.pi, alpha * np.pi, size=(N, P))
        eta[:, 0] = 1.0
        xi[:, 0] = 0.0

        theta = self.theta[:, None]
        vx = self._to_int(L * eta * np.cos(theta + xi))
        vy = self._to_int(L * eta * np.sin(theta + xi))

        tx = np.mod(self.position[:, 0:1] + vx, S)
        ty = np.mod(self.position[:, 1:2] + vy, S)
        return np.stack([tx, ty], axis=-1), np.stack([vx, vy], axis=-1)

    def _compute_popularity(self, targets):
        """人気度 ζ(x,y;t)（付録の式(4)(5)）。dedupe_votes=True なら同じ個体が同じサイトを何本選んでいても1票"""
        N, P, _ = targets.shape
        flat_x = targets[:, :, 0].ravel()
        flat_y = targets[:, :, 1].ravel()
        if self.dedupe_votes:
            triples = np.unique(np.stack([np.repeat(np.arange(N), P), flat_x, flat_y], axis=1), axis=0)
            flat_x, flat_y = triples[:, 1], triples[:, 2]
        z = np.zeros((self.S, self.S), dtype=int)
        np.add.at(z, (flat_x, flat_y), 1)
        return z

    # ------------------------------------------------------------------
    # 1ステップ
    # ------------------------------------------------------------------

    def step(self):
        """1ステップ進める"""
        old_position = self.position.copy()
        counts = self._move_all()

        d = self.position - old_position
        self.last_displacement = (d + self.S // 2) % self.S - self.S // 2
        self.last_rule_counts = counts
        for key, n in zip(("MA", "follow", "free", "stay"), counts):
            self.move_counts[key] += n

        self._update_principal_vector()
        self.t += 1

    def run(self, n_steps, record_history=False):
        """n_steps 回 step() を実行する。record_history=True なら位置の履歴 (n_steps+1, N, 2) を返す"""
        history = [self.position.copy()] if record_history else None
        for _ in range(n_steps):
            self.step()
            if record_history:
                history.append(self.position.copy())
        return np.stack(history, axis=0) if record_history else None

    def _move_all(self):
        N, S, c = self.N, self.S, self.c
        self.w_ma[:, :] = 0

        targets, vecs = self._generate_potential_transitions()
        z = self._compute_popularity(targets)
        self._pv_int = vecs[:, 0, :].copy()          # 式(13) の v(k,t;0)

        occupancy = self.occupancy.copy()
        vacant0 = self.occupancy == -1               # ステップ開始時に空いているサイト
        taken = np.zeros((S, S), dtype=bool)
        new_position = self.position.copy()
        cnt = [0, 0, 0, 0]

        ang = None
        if self.tiebreak == "angle":
            a = np.arctan2(vecs[:, :, 1], vecs[:, :, 0])
            ang = np.abs((a - a[:, :1] + np.pi) % (2 * np.pi) - np.pi)

        def move(k, nx, ny):
            x0, y0 = self.position[k]
            occupancy[x0, y0] = -1
            occupancy[nx, ny] = k
            new_position[k] = (nx, ny)

        order = [int(k) for k in self.rng.permutation(N)]
        wanderer = np.zeros(N, dtype=bool)

        # ---- 式(6)〜(9)：相互予期 ----
        for k in order:
            tx, ty = targets[k, :, 0], targets[k, :, 1]
            ok = (vacant0[tx, ty] & ~taken[tx, ty]) if self.static_vacancy else (occupancy[tx, ty] == -1)
            if not ok.any():
                continue
            zz = np.where(ok, z[tx, ty], -1)
            m = zz.max()
            if m < c:
                continue
            cand = np.flatnonzero(zz == m)
            if ang is not None:                      # 同率なら基本ベクトルに向きが一番近いもの
                a_c = ang[k, cand]
                cand = cand[a_c <= a_c.min() + 1e-9]
            i = int(self.rng.choice(cand))
            self.w_ma[self.position[k][0], self.position[k][1]] = 1
            move(k, tx[i], ty[i])
            taken[tx[i], ty[i]] = True
            wanderer[k] = True
            cnt[0] += 1

        # ---- 式(10)：追従（自分の遷移先のうち、wanderer が空けたサイト）----
        rest = []
        for k in order:
            if wanderer[k]:
                continue
            tx, ty = targets[k, :, 0], targets[k, :, 1]
            ok = (self.w_ma[tx, ty] == 1) & (occupancy[tx, ty] == -1)
            if ok.any():
                s = int(self.rng.choice(np.unique(tx[ok] * S + ty[ok])))
                move(k, s // S, s % S)
                taken[s // S, s % S] = True
                cnt[1] += 1
            else:
                rest.append(k)

        # ---- 式(11)：自由移動 ----
        for k in rest:
            tx, ty = targets[k, :, 0], targets[k, :, 1]
            ok = occupancy[tx, ty] == -1
            if ok.any():
                s = int(self.rng.choice(np.unique(tx[ok] * S + ty[ok])))
                move(k, s // S, s % S)
                taken[s // S, s % S] = True
                cnt[2] += 1
            else:
                cnt[3] += 1                          # その場に留まる

        self.position = new_position
        self.occupancy = occupancy
        return tuple(cnt)

    def _update_principal_vector(self):
        """
        式(12)(13)：V = P(t+1) - P(t)、θ(t+1) = angle(q)
          omega_on="principal" : q = V_u + ω·v_u(k,t;0)
          omega_on="velocity"  : q = v_u(k,t;0) + ω·V_u
        留まった個体は V_u = 0。
        """
        d = self.last_displacement.astype(float)
        norm = np.linalg.norm(d, axis=1)
        u = np.zeros((self.N, 2))
        mv = norm > 0
        u[mv] = d[mv] / norm[mv, None]

        spv = np.stack([np.cos(self.theta), np.sin(self.theta)], axis=1)
        if self.spv_mode == "int_vector":
            pn = np.linalg.norm(self._pv_int, axis=1)
            ok = pn > 0
            spv[ok] = self._pv_int[ok] / pn[ok, None]

        if self.omega_on == "principal":
            q = u + self.omega * spv
        else:
            q = spv + self.omega * u
        self.theta = np.arctan2(q[:, 1], q[:, 0])


# ======================================================================
# 評価用の関数
# ======================================================================

def polarization2(displacement):
    """論文2の整列度：動いた個体の単位変位ベクトルの平均の長さ（0〜1）"""
    d = displacement.astype(float)
    n = np.linalg.norm(d, axis=1)
    m = n > 0
    if not m.any():
        return 0.0
    return float(np.linalg.norm((d[m] / n[m, None]).mean(axis=0)))


def density_paper2(position, S):
    """論文2の密度：周り3×3にいる個体数の平均（自分は含まない）"""
    g = np.zeros((S, S), dtype=int)
    g[position[:, 0], position[:, 1]] = 1
    nb = sum(np.roll(np.roll(g, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1)) - g
    return float(nb[position[:, 0], position[:, 1]].mean())


def simulate(n_steps, **kwargs):
    """
    n_steps 回まわして、整列度・密度・ルールの割合の時系列を返す。
    kwargs は CrabSwarmPaper2 の引数（CONDITIONS・PRESETS の中身もここに渡す）。
    返り値: pol (n_steps,), dens (n_steps,), rules (n_steps, 4)
            rules の列は 相互予期 / 追従 / 自由移動 / 留まる の割合
    """
    m = CrabSwarmPaper2(**kwargs)
    pol, dens, rules = [], [], []
    for _ in range(n_steps):
        m.step()
        pol.append(polarization2(m.last_displacement))
        dens.append(density_paper2(m.position, m.S))
        rules.append(m.last_rule_counts)
    return np.array(pol), np.array(dens), np.array(rules) / m.N


if __name__ == "__main__":
    # MAM_basic_v1 と同じ軌跡になるかを確かめる（研究/ で  python -m model.MAM_paper2）
    from model.MAM_basic_v1 import SoldierCrabSwarm_v1
    for name, kw in PRESETS.items():
        full = dict(alpha_scale=1, int_mode="floor", omega_on="principal")
        full.update(kw)
        a = CrabSwarmPaper2(seed=3, **CONDITIONS["赤"], **kw)
        b = SoldierCrabSwarm_v1(model="paper2", seed=3, **CONDITIONS["赤"], **full)
        a.run(200); b.run(200)
        same = np.array_equal(a.position, b.position) and np.allclose(a.theta, b.theta)
        print(f"{name}: MAM_basic_v1 と同じ = {same}")
