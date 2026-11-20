import numpy as np

class SoldierCrabSwarm_step_popnew_noVM:
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
    omega : float, default 5.0
        基本ベクトル更新時の「前の基本ベクトル」の重み ω（方向の持続性）
        ω=0 のとき、基本ベクトルの向きは直前の実際の移動方向と一致する。
    c : int, default 2
        人気度の閾値 ζ ≧ c で相互予期による移動が発生する。
    seed : int or None
        乱数シード（再現性のため）
    """

    def __init__(self, N=100, S=50, L=4.0, P=15, alpha=0.5, R_Nf=2, omega=5.0, c=2, seed=None):

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
        self.omega = omega
        self.c = c
        self.seed = seed

        # 現在のステップ数
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

        # position[k] = [x, y]
        self.position = np.stack([xs, ys], axis=1).astype(int)

        # 各エージェントの基本ベクトルの向きをランダムに初期化
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

        # i=0 は基本ベクトル（principal vector）: eta_0=1.0, xi_0=0.0
        eta[:, 0] = 1.0
        xi[:, 0] = 0.0

        theta = self.theta[:, None]

        vx = np.floor(L * eta * np.cos(theta + xi)).astype(int)
        vy = np.floor(L * eta * np.sin(theta + xi)).astype(int)

        tx = np.mod(self.position[:, 0:1] + vx, S)
        ty = np.mod(self.position[:, 1:2] + vy, S)

        targets = np.stack([tx, ty], axis=-1)
        return targets

    def _compute_target_counts(self, targets):
        """人気度の計算"""
        N, P, _ = targets.shape
        S = self.S

        agent_idx = np.repeat(np.arange(N), P)
        flat_x = targets[:, :, 0].ravel()
        flat_y = targets[:, :, 1].ravel()
        triples = np.stack([agent_idx, flat_x, flat_y], axis=1)

        unique_triples = triples

        # 現在エージェントが存在する格子は popularity の計算対象から除外
        vacant = self.occupancy[unique_triples[:, 1], unique_triples[:, 2]] == -1
        unique_triples = unique_triples[vacant]

        counts = np.zeros((S, S), dtype=int)
        np.add.at(counts, (unique_triples[:, 1], unique_triples[:, 2]), 1)
        return counts

    def step(self):
        """1タイムステップ分の更新を行う"""

        N, S, c, R_Nf = self.N, self.S, self.c, self.R_Nf

        # ---- 非同期更新の準備 ----
        self.w[:, :] = 0

        # ---- 可能的遷移の生成 と 人気度の事前計算 ----
        targets = self._generate_potential_transitions()
        target_counts = self._compute_target_counts(targets)

        occupancy = self.occupancy.copy()

        # ---- 遷移 ----
        new_position = self.position.copy()
        moved = np.zeros(N, dtype=bool)

        dx_Nf, dy_Nf = self._square_neighbor_offsets(R_Nf)

        order = self.rng.permutation(N)

        for k in order:
            x0, y0 = self.position[k]
            k_targets = targets[k]
            tx, ty = k_targets[:, 0], k_targets[:, 1]
            vacant_mask = occupancy[tx, ty] == -1

            nx, ny = None, None

            # --- 式(4): 相互予期 ---
            if vacant_mask.any():
                cand_counts = np.where(vacant_mask, target_counts[tx, ty], -1)
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
                if is_freed.any():
                    free_idx = np.flatnonzero(is_freed)
                    chosen_site = self.rng.choice(free_idx)
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
            occupancy[x0, y0] = -1
            occupancy[nx, ny] = k
            self.w[x0, y0] = 1
            new_position[k] = (nx, ny)
            moved[k] = True

        # ---- 実際の移動量（速度 v_{i,t}）を計算 ----
        # トーラス上なので、端をまたいだ移動は最短の変位に直す
        displacement = new_position - self.position
        displacement = displacement - S * np.round(displacement / S).astype(int)

        self.position = new_position
        self.occupancy = occupancy

        # ---- 次のステップへの更新（論文2: 自分の前回の速度による基本ベクトルの更新） ----
        self._update_principal_vector(displacement)

        self.t += 1

    def _update_principal_vector(self, displacement):
        """
        基本ベクトルの更新（論文2）
            q = u + ω * spv
            pv_{t+1} = L * q / |q|
        u   : 実際に動いた変位 v の単位ベクトル
        spv : 前の基本ベクトルの単位ベクトル (cosθ, sinθ)
        向きだけ持っていれば長さは常に L なので、θ を arctan2 で更新する。
        """
        omega = self.omega

        # 変位の単位ベクトル u（動かなかった個体は u = 0 のままにする）
        norm = np.linalg.norm(displacement, axis=1)
        u = np.zeros((self.N, 2), dtype=float)
        nonzero = norm > 0
        u[nonzero] = displacement[nonzero] / norm[nonzero, None]

        # 前の基本ベクトルの単位ベクトル
        spv = np.stack([np.cos(self.theta), np.sin(self.theta)], axis=1)

        q = u + omega * spv
        self.theta = np.arctan2(q[:, 1], q[:, 0])

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