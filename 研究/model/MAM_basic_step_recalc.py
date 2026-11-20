import numpy as np

class SoldierCrabSwarm_step_recalc:
    """
    人気度を「個体が1体遷移するたびに」再計算するバージョン。

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
    recompute_popularity : bool, default True
        True  : 誰かが遷移するたびに、最新の占有状況で人気度を再計算する。
                あわせて、遷移済みの個体の可能的遷移は人気度の計算から除外する
        False : ステップ開始時点の占有状況で人気度を固定する（元のモデルの挙動）
    seed : int or None
        乱数シード（再現性のため）
    """

    def __init__(self, N=100, S=50, L=4.0, P=15, alpha=0.5, R_Nf=2, R_M=3, c=2,
                 recompute_popularity=True, seed=None):

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
        self.recompute_popularity = recompute_popularity
        self.seed = seed

        self.t = 0

        # ---- エージェントの配置 ----
        if N > S * S:
            raise ValueError(
                f"N({N}) が格子点の総数 S*S({S*S}) を超えています。"
                " 1つの格子点に最大1体しか配置できません。"
            )

        all_sites = self.rng.permutation(S * S)[:N]
        xs = all_sites % S
        ys = all_sites // S

        self.position = np.stack([xs, ys], axis=1).astype(int)
        self.theta = self.rng.uniform(0.0, 2 * np.pi, size=N)

        # 非同期更新のためのフラグ w(x, y; t)
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
        """可能的遷移の生成"""
        N, P, S, L, alpha = self.N, self.P, self.S, self.L, self.alpha

        eta = self.rng.uniform(0.0, 1.0, size=(N, P))
        xi = self.rng.uniform(-alpha * np.pi, alpha * np.pi, size=(N, P))

        # i=0 は基本ベクトル（principal vector）
        eta[:, 0] = 1.0
        xi[:, 0] = 0.0

        theta = self.theta[:, None]

        vx = np.floor(L * eta * np.cos(theta + xi)).astype(int)
        vy = np.floor(L * eta * np.sin(theta + xi)).astype(int)

        tx = np.mod(self.position[:, 0:1] + vx, S)
        ty = np.mod(self.position[:, 1:2] + vy, S)

        return np.stack([tx, ty], axis=-1)

    def _compute_raw_counts(self, targets):
        """
        「空き/占有」を考慮しない生の狙われ数 raw_counts[x, y] を計算する。
        （同一エージェントが同じサイトを複数の遷移で狙っている分は1回と数える）
        空きかどうかは、人気度を使う時点の occupancy で判定する。
        """
        N, P, _ = targets.shape
        S = self.S

        agent_idx = np.repeat(np.arange(N), P)
        flat_x = targets[:, :, 0].ravel()
        flat_y = targets[:, :, 1].ravel()
        triples = np.stack([agent_idx, flat_x, flat_y], axis=1)

        unique_triples = np.unique(triples, axis=0)

        raw_counts = np.zeros((S, S), dtype=int)
        np.add.at(raw_counts, (unique_triples[:, 1], unique_triples[:, 2]), 1)
        return raw_counts

    def step(self):
        """1タイムステップ分の更新を行う"""

        N, S, c, R_Nf = self.N, self.S, self.c, self.R_Nf

        self.w[:, :] = 0

        # ---- 可能的遷移の生成 と 生カウントの事前計算 ----
        targets = self._generate_potential_transitions()
        raw_counts = self._compute_raw_counts(targets)

        occupancy = self.occupancy.copy()

        # recompute_popularity=False のときは、ステップ開始時点の空き状況で人気度を固定
        if not self.recompute_popularity:
            fixed_counts = np.where(occupancy == -1, raw_counts, 0)

        new_position = self.position.copy()
        moved = np.zeros(N, dtype=bool)

        dx_Nf, dy_Nf = self._square_neighbor_offsets(R_Nf)

        order = self.rng.permutation(N)

        for k in order:
            x0, y0 = self.position[k]
            k_targets = targets[k]
            tx, ty = k_targets[:, 0], k_targets[:, 1]
            vacant_mask = occupancy[tx, ty] == -1  # 現在の占有状況で空いているか

            # ---- 人気度の取得 ----
            # 再計算モード：raw_counts を「いまの occupancy」でマスクしたものが
            #   「直前までの遷移を反映して再計算した人気度」と一致する。
            #   （空いたマスは人気度に加算され、埋まったマスは0になる）
            if self.recompute_popularity:
                pop = np.where(vacant_mask, raw_counts[tx, ty], 0)
            else:
                pop = fixed_counts[tx, ty]

            nx, ny = None, None

            # --- 式(4): 相互予期 ---
            if vacant_mask.any():
                cand_counts = np.where(vacant_mask, pop, -1)
                m = cand_counts.max()
                if m >= c:
                    tie_idx = np.flatnonzero(cand_counts == m)
                    chosen_i = self.rng.choice(tie_idx)
                    nx, ny = k_targets[chosen_i]

            # --- 式(5): 追従 ---
            if nx is None:
                nbx = np.mod(x0 + dx_Nf, S)
                nby = np.mod(y0 + dy_Nf, S)
                is_freed = (self.w[nbx, nby] == 1) & (occupancy[nbx, nby] == -1)
                own_targets = set(map(tuple, k_targets))
                candidate_idx = [
                    i for i in np.flatnonzero(is_freed)
                    if (nbx[i], nby[i]) in own_targets
                ]
                if candidate_idx:
                    chosen_site = self.rng.choice(candidate_idx)
                    nx, ny = nbx[chosen_site], nby[chosen_site]

            # --- 式(6): 自由移動 ---
            if nx is None:
                if vacant_mask.any():
                    vacant_idx = np.flatnonzero(vacant_mask)
                    chosen_i = self.rng.choice(vacant_idx)
                    nx, ny = k_targets[chosen_i]
                else:
                    nx, ny = x0, y0

            # --- 実際に遷移させる ---
            # occupancy を更新した時点で、次の個体が見る人気度は自動的に再計算後のものになる
            occupancy[x0, y0] = -1
            occupancy[nx, ny] = k
            self.w[x0, y0] = 1
            new_position[k] = (nx, ny)
            moved[k] = True

            # --- 遷移済み個体の可能的遷移を人気度の計算から除外 ---
            # 生成時と同じく、同一サイトの重複は1回として数えているので、
            # 引くときも重複を除いた遷移先に対して1ずつ引く
            if self.recompute_popularity:
                uniq_targets = np.unique(k_targets, axis=0)
                np.subtract.at(raw_counts, (uniq_targets[:, 0], uniq_targets[:, 1]), 1)

        self.position = new_position
        self.occupancy = occupancy

        # 速度一致
        self._velocity_matching()

        self.t += 1

    def _velocity_matching(self):
        """速度一致"""
        N, S, R_M = self.N, self.S, self.R_M
        dx, dy = self._square_neighbor_offsets(R_M)

        new_theta = np.empty(N, dtype=float)

        for k in range(N):
            x0, y0 = self.position[k]
            nbx = np.mod(x0 + dx, S)
            nby = np.mod(y0 + dy, S)
            occupants = self.occupancy[nbx, nby]
            occupants = occupants[(occupants != -1) & (occupants != k)]  # 自分自身は除外

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
        """n_steps 回 step() を実行する"""
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
