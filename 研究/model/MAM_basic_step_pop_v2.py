import numpy as np

class SoldierCrabSwarm_step_pop_v2:
    """
    Parameters
    ----------
    N : int
        エージェント（カニ）の数
    S : int
        格子空間の一辺の大きさ（S×Sのトーラス空間）
    L : float, default 4
        基本ベクトルの長さ
    P : int, default 15
        可能的遷移の数
    alpha : float, default 0.5
        変異角 α
    R_Nf : int, default 2
        追従者近傍の半径 R(Nf)
    R_M : int, default 3
        速度一致近傍の半径 R(M)
    c : int, default 2
        人気度の閾値 ζ ≧ c で相互予期による移動が発生する。
    seed : int or None
        乱数シード（再現性のため）
    follow_own : bool, default True
        True : 追従で動ける先を「自分の可能的遷移の先」に限定する（Murakami et al. 2017 の記述）
        False: 近傍の空いたマスならどこでもOK（旧コードの挙動）
    pop_dynamic : bool, default True
        True : すでに動いた個体の「票」を人気度から取り除く。
               人気度は「まだ動いていない個体が狙っている数」になる。
        False: 動いた個体の票も最後まで残す（旧コードの挙動）

    ※ follow_own=False, pop_dynamic=False にすると、旧コードと同じ乱数列・同じ挙動になる。
    """

    def __init__(self, N=100, S=50, L=4.0, P=15, alpha=0.5, R_Nf=2, R_M=3, c=2, seed=None,
                 follow_own=True, pop_dynamic=True):

        # ---- 乱数生成 ----
        self.rng = np.random.default_rng(seed)

        # ---- 格子空間の設定 ----
        self.S = S

        # ---- パラメータの設定 ----
        self.N = N
        self.L = L
        self.P = P
        self.alpha = alpha
        self.R_Nf = R_Nf
        self.R_M = R_M
        self.c = c
        self.seed = seed
        self.follow_own = follow_own
        self.pop_dynamic = pop_dynamic

        # 現在のステップ数
        self.t = 0

        # 直近の step で、各ルールで動いた個体数 (相互予期, 追従, 自由移動, その場に留まった)
        self.last_rule_counts = (0, 0, 0, 0)

        # ---- エージェントの配置 ----
        if N > S * S:
            raise ValueError(
                f"N({N}) が格子点の総数 S*S({S*S}) を超えています。"
                " 1つの格子点に最大1体しか配置できません。"
            )

        all_sites = self.rng.permutation(S * S)[:N]
        xs = all_sites % S
        ys = all_sites // S

        # position[k] = [x, y]
        self.position = np.stack([xs, ys], axis=1).astype(int)

        # 各エージェントの基本ベクトルの向きをランダムに初期化
        self.theta = self.rng.uniform(0.0, 2 * np.pi, size=N)

        # 非同期更新のためのフラグ w(x, y; t)（格子点ごと）
        self.w = np.zeros((S, S), dtype=int)

        # occupancy[x, y] = k のインデックス, 空きなら -1
        self.occupancy = -np.ones((S, S), dtype=int)
        for k in range(N):
            x, y = self.position[k]
            self.occupancy[x, y] = k

    # ------------------------------------------------------------------
    # 各ステップでの動作
    # ------------------------------------------------------------------

    def _square_neighbor_offsets(self, R):
        """半径Rの正方形近傍のオフセット (dx, dy) 一覧を返す（(2R+1)^2個、中心含む）"""
        rng = np.arange(-R, R + 1)
        dx, dy = np.meshgrid(rng, rng, indexing="ij")
        return dx.ravel(), dy.ravel()

    def _generate_potential_transitions(self):
        """
        可能的遷移の生成
        """
        N, P, S, L, alpha = self.N, self.P, self.S, self.L, self.alpha

        eta = self.rng.uniform(0.0, 1.0, size=(N, P))
        xi = self.rng.uniform(-alpha * np.pi, alpha * np.pi, size=(N, P))

        # i=0 は基本ベクトル（principal vector）: eta_0=1.0, xi_0=0.0
        eta[:, 0] = 1.0
        xi[:, 0] = 0.0

        theta = self.theta[:, None]

        # 遷移ベクトル（小数点以下切り捨て = 論文の Int(x)）
        vx = np.floor(L * eta * np.cos(theta + xi)).astype(int)
        vy = np.floor(L * eta * np.sin(theta + xi)).astype(int)

        # 遷移先の座標（トーラス）
        tx = np.mod(self.position[:, 0:1] + vx, S)
        ty = np.mod(self.position[:, 1:2] + vy, S)

        targets = np.stack([tx, ty], axis=-1)
        return targets

    def _compute_target_counts(self, targets):
        """
        人気度の計算（1個体は1つのサイトに最大1票）
        """
        N, P, _ = targets.shape
        S = self.S

        agent_idx = np.repeat(np.arange(N), P)
        flat_x = targets[:, :, 0].ravel()
        flat_y = targets[:, :, 1].ravel()
        triples = np.stack([agent_idx, flat_x, flat_y], axis=1)

        # 同一エージェントが同じサイトを複数の遷移で狙っている分を除去
        unique_triples = np.unique(triples, axis=0)

        counts = np.zeros((S, S), dtype=int)
        np.add.at(counts, (unique_triples[:, 1], unique_triples[:, 2]), 1)
        return counts

    def _retire_votes(self, k_targets, target_counts):
        """
        動き終わった個体の票を人気度から取り除く（pop_dynamic=True のとき）
        """
        S = self.S
        site_ids = np.unique(k_targets[:, 0] * S + k_targets[:, 1])   # 重複なしの狙いサイト
        target_counts[site_ids // S, site_ids % S] -= 1

    def step(self):
        """
        1タイムステップ分の更新を行う
        """

        N, S, c, R_Nf = self.N, self.S, self.c, self.R_Nf

        # ---- 非同期更新の準備 ----
        self.w[:, :] = 0

        # ---- 可能的遷移の生成 と 人気度の事前計算 ----
        targets = self._generate_potential_transitions()
        target_counts = self._compute_target_counts(targets)

        occupancy = self.occupancy.copy()
        new_position = self.position.copy()

        dx_Nf, dy_Nf = self._square_neighbor_offsets(R_Nf)

        n_ma = n_follow = n_free = n_stay = 0

        # === 非同期更新（相互予期 → 追従 → 自由移動 を1個体ずつ判定） ===
        order = self.rng.permutation(N)

        for k in order:
            x0, y0 = self.position[k]
            k_targets = targets[k]
            tx, ty = k_targets[:, 0], k_targets[:, 1]
            vacant_mask = occupancy[tx, ty] == -1  # 自分の遷移先が現在空いているか

            nx, ny = None, None
            kind = None

            # --- 式(4): 相互予期 ---
            # 空いていて、かつ人気度が最大の遷移先へ（人気度 >= c のとき）
            if vacant_mask.any():
                cand_counts = np.where(vacant_mask, target_counts[tx, ty], -1)
                m = cand_counts.max()
                if m >= c:
                    tie_idx = np.flatnonzero(cand_counts == m)
                    chosen_i = self.rng.choice(tie_idx)  # 同率ならランダム
                    nx, ny = k_targets[chosen_i]
                    kind = "ma"

            # --- 式(5): 追従 ---
            # 近傍で「いま空いた（w=1 かつ空き）」マスへ。
            # follow_own=True なら、自分の可能的遷移の先になっているマスだけが候補。
            if nx is None:
                nbx = np.mod(x0 + dx_Nf, S)
                nby = np.mod(y0 + dy_Nf, S)
                is_freed = (self.w[nbx, nby] == 1) & (occupancy[nbx, nby] == -1)
                if self.follow_own:
                    own_ids = tx * S + ty
                    is_freed &= np.isin(nbx * S + nby, own_ids)
                if is_freed.any():
                    free_idx = np.flatnonzero(is_freed)
                    chosen_site = self.rng.choice(free_idx)
                    nx, ny = nbx[chosen_site], nby[chosen_site]
                    kind = "follow"

            # --- 式(6): 自由移動 ---
            if nx is None:
                if vacant_mask.any():
                    vacant_idx = np.flatnonzero(vacant_mask)
                    chosen_i = self.rng.choice(vacant_idx)
                    nx, ny = k_targets[chosen_i]
                    kind = "free"
                else:
                    # 空いている遷移先が1つもなければその場に留まる（例外処理）
                    nx, ny = x0, y0
                    kind = "stay"

            # --- 実際に遷移させる ---
            if self.pop_dynamic:
                self._retire_votes(k_targets, target_counts)   # 動いたので、この個体の票は取り下げ
            occupancy[x0, y0] = -1
            occupancy[nx, ny] = k
            self.w[x0, y0] = 1  # 元いた場所は「更新済み（空いた）」とマーク
            new_position[k] = (nx, ny)

            if kind == "ma":
                n_ma += 1
            elif kind == "follow":
                n_follow += 1
            elif kind == "free":
                n_free += 1
            else:
                n_stay += 1

        self.position = new_position
        self.occupancy = occupancy
        self.last_rule_counts = (n_ma, n_follow, n_free, n_stay)

        # ---- 次のステップへの更新 ----
        self._velocity_matching()

        self.t += 1

    def _velocity_matching(self):
        """
        速度一致
        """
        N, S, R_M = self.N, self.S, self.R_M
        dx, dy = self._square_neighbor_offsets(R_M)

        new_theta = np.empty(N, dtype=float)

        for k in range(N):
            x0, y0 = self.position[k]
            nbx = np.mod(x0 + dx, S)
            nby = np.mod(y0 + dy, S)
            occupants = self.occupancy[nbx, nby]
            occupants = occupants[occupants != -1]

            if occupants.size == 0:
                new_theta[k] = self.theta[k]
                continue

            angles = self.theta[occupants]
            mean_vec = np.array([np.cos(angles).mean(), np.sin(angles).mean()])
            if np.allclose(mean_vec, 0.0):
                new_theta[k] = self.theta[k]
            else:
                new_theta[k] = np.arctan2(mean_vec[1], mean_vec[0])

        self.theta = new_theta

    def run(self, n_steps, record_history=False):
        """
        n_steps 回 step() を実行する
        """
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
# 評価用の関数（ノートブックから使う）
# ----------------------------------------------------------------------

def calc_density(position, S):
    """3×3近傍（自分を除く）にいる個体数の平均。ランダム配置 (N=100, S=50) だと約0.32。"""
    g = np.zeros((S, S), dtype=int)
    g[position[:, 0], position[:, 1]] = 1
    nb = sum(np.roll(np.roll(g, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1)) - g
    return nb[position[:, 0], position[:, 1]].mean()


def calc_polarity(prev_position, position, S):
    """実際に動いた方向（単位ベクトル）の平均ベクトルの長さ。0〜1。"""
    d = position - prev_position
    d = (d + S // 2) % S - S // 2          # トーラス補正
    n = np.linalg.norm(d, axis=1)
    m = n > 0
    if not m.any():
        return 0.0
    u = d[m] / n[m, None]
    return float(np.linalg.norm(u.mean(axis=0)))


def simulate(n_steps, **kwargs):
    """
    シミュレーションを回して、密度・偏極・各ルールの使用割合の時系列を返す。
    kwargs は SoldierCrabSwarm_step_pop の引数。
    """
    model = SoldierCrabSwarm_step_pop_v2(**kwargs)
    S = model.S
    prev = model.position.copy()
    density = [calc_density(prev, S)]
    polarity = [np.nan]
    rules = []
    for _ in range(n_steps):
        model.step()
        density.append(calc_density(model.position, S))
        polarity.append(calc_polarity(prev, model.position, S))
        rules.append(model.last_rule_counts)
        prev = model.position.copy()
    rules = np.array(rules) / model.N     # 列: 相互予期, 追従, 自由移動, 留まる
    return np.array(density), np.array(polarity), rules
