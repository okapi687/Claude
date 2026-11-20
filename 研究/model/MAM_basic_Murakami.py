import numpy as np


class SoldierCrabSwarm_Murakami:
    """
    相互予期モデル（Murakami et al. 2017, Sci. Rep. 7:46447 + Supplementary Appendix S1）。
    速度一致は使わず、各個体の基本ベクトルを自分の直前の移動で更新する。

    1ステップの流れ（付録 式(1)〜(13)）
      0. 各個体が P 本の可能的遷移を作り、各サイトの人気度 ζ を数える
      1. 相互予期（wanderer）: ランダムな順に、自分の遷移先のうち人気度が最大（ζ>=c）の
         空きサイトへ移動。同率なら基本ベクトルに向きが最も近いもの。
      2. 追従（follower）: 1で動けなかった個体が、wanderer の空けたサイトのうち
         自分の遷移先になっているものへ移動。
      3. 自由移動（free）: 残りが、自分の遷移先の空きサイトへランダムに移動。
         なければその場に留まる。
      4. 向きの更新: V = P(t+1) - P(t),  q = V_u + ω·v_u(k,t;0),  θ(t+1) = angle(q)

    Parameters
    ----------
    N : int, default 100
        個体数
    S : int, default 50
        格子の一辺（S×Sのトーラス）
    L : float, default 6.0
        基本ベクトルの長さ（Murakami の Fig.3 は L=6）
    P : int, default 40
        可能的遷移の数
    alpha : float, default 0.2
        変異角 α（ξ は [-απ, απ] の一様乱数）
    omega : float, default 5.0
        直前の基本ベクトルの重み（向きの粘り強さ）
    c : int, default 2
        人気度の閾値（ζ >= c で相互予期が成立。付録は ζ > 1）
    seed : int or None
        乱数シード

    --- 付録の文面があいまいな点の切り替え（既定値は Fig.S2 の割合に合うもの） ---
    int_mode : {"round", "floor", "trunc"}, default "round"
        Int(x)。付録は floor と書くが、floor だと (-,-) 方向の偏りが向きの更新に積もって
        偏極が常に 0.97 以上になる。Fig.S2（青: 相互予期/追従/自由 ≈ 63/5/32）に合うのは round。
    dedupe_votes : bool, default True
        True : 1個体につき1サイト最大1票（自分の票は含む）。Fig.S2 の割合に合う。
        False: (個体, 遷移) の組を全部1票として数える。青でも相互予期が97%になる。
    spv_mode : {"int_vector", "angle"}, default "int_vector"
        式(13)の v_u(k,t;0)。"int_vector" は整数化した基本ベクトルの単位ベクトル（付録の式どおり）、
        "angle" は (cosθ, sinθ)。
    static_vacancy : bool, default True
        True : ステップ開始時に占有されているサイトは人気度の対象外（付録式(5)どおり）。
        False: 動いた個体の空き跡も、相互予期の対象にする。
    tiebreak : {"angle", "random"}, default "angle"
        最大人気度が複数あるときの選び方。"angle" は基本ベクトルに向きが最も近いもの（付録どおり）。
    """

    def __init__(self, N=100, S=50, L=6.0, P=40, alpha=0.2, omega=5.0, c=2, seed=None,
                 int_mode="round", dedupe_votes=True, spv_mode="int_vector",
                 static_vacancy=True, tiebreak="angle"):

        if int_mode not in ("round", "floor", "trunc"):
            raise ValueError("int_mode は 'round' / 'floor' / 'trunc' のどれかです。")
        if spv_mode not in ("int_vector", "angle"):
            raise ValueError("spv_mode は 'int_vector' か 'angle' です。")
        if tiebreak not in ("angle", "random"):
            raise ValueError("tiebreak は 'angle' か 'random' です。")
        if N > S * S:
            raise ValueError(
                f"N({N}) が格子点の総数 S*S({S*S}) を超えています。"
                " 1つの格子点に最大1体しか配置できません。"
            )

        self.rng = np.random.default_rng(seed)
        self.S, self.N, self.L, self.P = S, N, L, P
        self.alpha, self.omega, self.c, self.seed = alpha, omega, c, seed
        self.int_mode, self.dedupe_votes, self.spv_mode = int_mode, dedupe_votes, spv_mode
        self.static_vacancy, self.tiebreak = static_vacancy, tiebreak

        self.t = 0
        # 直近の step で各ルールで動いた個体数 (相互予期, 追従, 自由移動, 留まる)
        self.last_rule_counts = (0, 0, 0, 0)

        # 初期配置: 重複なしのランダムな格子点、基本ベクトルの向きは一様乱数
        all_sites = self.rng.permutation(S * S)[:N]
        xs = all_sites % S
        ys = all_sites // S
        self.position = np.stack([xs, ys], axis=1).astype(int)    # P(k, t) = (x, y)
        self.theta = self.rng.uniform(0.0, 2 * np.pi, size=N)

        # w[x, y] = 1: wanderer が空けたサイト（付録の χ）
        self.w = np.zeros((S, S), dtype=int)
        # occupancy[x, y] = k（その格子にいる個体の番号）、空きなら -1
        self.occupancy = -np.ones((S, S), dtype=int)
        for k in range(N):
            x, y = self.position[k]
            self.occupancy[x, y] = k

    # ------------------------------------------------------------------
    # 可能的遷移と人気度
    # ------------------------------------------------------------------

    def _generate_potential_transitions(self):
        """式(2)(3)(4): 可能的遷移ベクトル v と目標サイト τ = P + v（トーラス）。"""
        N, P, S, L, alpha = self.N, self.P, self.S, self.L, self.alpha

        eta = self.rng.uniform(0.0, 1.0, size=(N, P))
        xi = self.rng.uniform(-alpha * np.pi, alpha * np.pi, size=(N, P))
        eta[:, 0] = 1.0     # i=0 は基本ベクトル: η=1, ξ=0
        xi[:, 0] = 0.0

        theta = self.theta[:, None]
        to_int = {"floor": np.floor, "trunc": np.trunc, "round": np.rint}[self.int_mode]
        vx = to_int(L * eta * np.cos(theta + xi)).astype(int)
        vy = to_int(L * eta * np.sin(theta + xi)).astype(int)

        tx = np.mod(self.position[:, 0:1] + vx, S)
        ty = np.mod(self.position[:, 1:2] + vy, S)
        targets = np.stack([tx, ty], axis=-1)       # (N, P, 2)
        vecs = np.stack([vx, vy], axis=-1)          # (N, P, 2) 整数ベクトル
        return targets, vecs

    def _compute_popularity(self, targets):
        """式(5): 人気度 ζ(x, y; t)。占有サイトの除外は、移動先を選ぶときに行う。"""
        N, P, _ = targets.shape
        S = self.S
        flat_x = targets[:, :, 0].ravel()
        flat_y = targets[:, :, 1].ravel()

        if self.dedupe_votes:
            agent_idx = np.repeat(np.arange(N), P)
            triples = np.stack([agent_idx, flat_x, flat_y], axis=1)
            triples = np.unique(triples, axis=0)
            flat_x, flat_y = triples[:, 1], triples[:, 2]

        # ===== ここを追加 =====
        # 現在エージェントが存在する格子は popularity の計算対象から除外
        vacant = self.occupancy[
            triples[:, 1],
            triples[:, 2]
        ] == -1

        triples = triples[vacant]
        # =====================

        z = np.zeros((S, S), dtype=int)
        np.add.at(z, (flat_x, flat_y), 1)
        return z

    # ------------------------------------------------------------------
    # 1ステップ
    # ------------------------------------------------------------------

    def step(self):
        N, S, c = self.N, self.S, self.c

        self.w[:, :] = 0
        old_position = self.position.copy()

        targets, vecs = self._generate_potential_transitions()
        z = self._compute_popularity(targets)
        self._pv_int = vecs[:, 0, :].copy()          # 式(13) の v(k,t;0)

        occupancy = self.occupancy.copy()
        vacant0 = self.occupancy == -1               # ステップ開始時に空いているサイト
        taken = np.zeros((S, S), dtype=bool)         # 更新済み個体が入ったサイト U_N
        new_position = self.position.copy()
        counts = {"ma": 0, "follow": 0, "free": 0, "stay": 0}

        # 同率のときの「基本ベクトルとの向きの近さ」
        ang = None
        if self.tiebreak == "angle":
            a = np.arctan2(vecs[:, :, 1], vecs[:, :, 0])
            ang = np.abs((a - a[:, :1] + np.pi) % (2 * np.pi) - np.pi)

        def move(k, nx, ny, kind):
            x0, y0 = self.position[k]
            occupancy[x0, y0] = -1
            occupancy[nx, ny] = k
            new_position[k] = (nx, ny)
            counts[kind] += 1

        order = [int(k) for k in self.rng.permutation(N)]   # 非同期更新の順序
        wanderer = np.zeros(N, dtype=bool)

        # ---- 1. 式(6)〜(9): 相互予期（wanderer）----
        for k in order:
            tx, ty = targets[k, :, 0], targets[k, :, 1]
            if self.static_vacancy:
                ok = vacant0[tx, ty] & ~taken[tx, ty]
            else:
                ok = occupancy[tx, ty] == -1
            if not ok.any():
                continue
            zz = np.where(ok, z[tx, ty], -1)
            m = zz.max()
            if m < c:
                continue
            cand = np.flatnonzero(zz == m)
            if ang is not None:
                a_c = ang[k, cand]
                cand = cand[a_c <= a_c.min() + 1e-9]
            i = int(self.rng.choice(cand))
            nx, ny = tx[i], ty[i]
            self.w[self.position[k][0], self.position[k][1]] = 1    # 式(9): 空けたサイトを記録
            move(k, nx, ny, "ma")
            taken[nx, ny] = True
            wanderer[k] = True

        # ---- 2. 式(10): 追従（follower）----
        rest = []
        for k in order:
            if wanderer[k]:
                continue
            tx, ty = targets[k, :, 0], targets[k, :, 1]
            ok = (self.w[tx, ty] == 1) & (occupancy[tx, ty] == -1)
            if ok.any():
                sites = np.unique(tx[ok] * S + ty[ok])
                s = int(self.rng.choice(sites))
                move(k, s // S, s % S, "follow")
                taken[s // S, s % S] = True
            else:
                rest.append(k)

        # ---- 3. 式(11): 自由移動 ----
        for k in rest:
            tx, ty = targets[k, :, 0], targets[k, :, 1]
            ok = occupancy[tx, ty] == -1
            if ok.any():
                sites = np.unique(tx[ok] * S + ty[ok])
                s = int(self.rng.choice(sites))
                move(k, s // S, s % S, "free")
                taken[s // S, s % S] = True
            else:
                move(k, *self.position[k], "stay")

        self.position = new_position
        self.occupancy = occupancy
        self.last_rule_counts = (counts["ma"], counts["follow"], counts["free"], counts["stay"])

        # ---- 4. 式(12)(13): 基本ベクトルの更新 ----
        self._update_principal_vector(old_position, new_position)
        self.t += 1

    def _update_principal_vector(self, old_position, new_position):
        """
        V = P(t+1) - P(t),  q = V_u + ω·v_u(k,t;0),  θ(t+1) = angle(q)
        その場に留まった個体は V_u = 0 とするので、向きは変わらない。
        """
        S = self.S
        d = new_position - old_position
        d = (d + S // 2) % S - S // 2                # トーラス補正
        norm = np.linalg.norm(d, axis=1)

        u = np.zeros((self.N, 2))
        moved = norm > 0
        u[moved] = d[moved] / norm[moved, None]

        spv = np.stack([np.cos(self.theta), np.sin(self.theta)], axis=1)
        if self.spv_mode == "int_vector":
            pn = np.linalg.norm(self._pv_int, axis=1)
            ok = pn > 0
            spv[ok] = self._pv_int[ok] / pn[ok, None]

        q = u + self.omega * spv
        self.theta = np.arctan2(q[:, 1], q[:, 0])

    def run(self, n_steps, record_history=False):
        """n_steps 回 step() を実行する。record_history=True なら位置の履歴 (n_steps+1, N, 2) を返す。"""
        history = None
        if record_history:
            history = [self.position.copy()]
        for _ in range(n_steps):
            self.step()
            if record_history:
                history.append(self.position.copy())
        if record_history:
            return np.stack(history, axis=0)
        return None


# ----------------------------------------------------------------------
# 評価用の関数
# ----------------------------------------------------------------------

def calc_density(position, S):
    """Murakami の密度: 3×3近傍（自分を除く）にいる個体数の平均。"""
    g = np.zeros((S, S), dtype=int)
    g[position[:, 0], position[:, 1]] = 1
    nb = sum(np.roll(np.roll(g, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1)) - g
    return nb[position[:, 0], position[:, 1]].mean()


def calc_polarity(prev_position, position, S):
    """Murakami の偏極: 実際に動いた向き（単位ベクトル）の平均ベクトルの長さ。0〜1。"""
    d = position - prev_position
    d = (d + S // 2) % S - S // 2
    n = np.linalg.norm(d, axis=1)
    m = n > 0
    if not m.any():
        return 0.0
    u = d[m] / n[m, None]
    return float(np.linalg.norm(u.mean(axis=0)))


def simulate(n_steps, **kwargs):
    """
    n_steps 回回して、密度・偏極・各ルールの使用割合の時系列を返す。
    返り値: density (n_steps+1,), polarity (n_steps+1, 先頭は nan), rules (n_steps, 4)
            rules の列は 相互予期, 追従, 自由移動, 留まる の割合。
    例: Murakami の Fig.3 の赤 → simulate(1500, P=40, alpha=0.2, seed=0)
    """
    model = SoldierCrabSwarm_Murakami(**kwargs)
    S = model.S
    prev = model.position.copy()
    dens = [calc_density(prev, S)]
    pol = [np.nan]
    rules = []
    for _ in range(n_steps):
        model.step()
        dens.append(calc_density(model.position, S))
        pol.append(calc_polarity(prev, model.position, S))
        rules.append(model.last_rule_counts)
        prev = model.position.copy()
    return np.array(dens), np.array(pol), np.array(rules) / model.N
