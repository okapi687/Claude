import numpy as np

class SoldierCrabSwarm_step_pop_dynamic:
    """
    人気度を「遷移先が1体決まるたびに」再計算するバージョン。

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

    人気度の定義（動的版）
    ----------------------
    ある格子点 (x, y) の人気度 = 「まだ動いていない個体」のうち、
    その格子点を可能的遷移に含む個体の数（同一個体の重複は1と数える）。
    ・個体が移動を確定するたびに、その個体の寄与を人気度から取り除く
    ・空き/占有の判定は、そのときどきの occupancy で行う
    """

    def __init__(self, N=100, S=50, L=4.0, P=15, alpha=0.5, R_Nf=2, R_M=3, c=2, seed=None):

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

    def _build_unique_sites(self, targets):
        """
        各個体の可能的遷移先を「重複なし」の格子点インデックス(x*S+y)にまとめる。
        戻り値: 長さNのリスト。unique_sites[k] は個体kが狙う格子点の1次元インデックス配列。
        """
        S = self.S
        flat = targets[:, :, 0] * S + targets[:, :, 1]   # (N, P)
        return [np.unique(flat[k]) for k in range(self.N)]

    def _initial_counts(self, unique_sites):
        """
        全個体が未移動の状態での人気度（生カウント）を作る。
        占有されているかどうかはここでは見ない。
        （参照時に vacant_mask で空きだけを候補にするので、結果は元の定義と同じになる）
        """
        S = self.S
        counts = np.zeros(S * S, dtype=int)
        for sites in unique_sites:
            counts[sites] += 1      # 個体内で一意なので fancy index の += で安全
        return counts.reshape(S, S)

    def step(self):
        """
        1タイムステップ分の更新を行う
        """
        N, S, c, R_Nf = self.N, self.S, self.c, self.R_Nf

        # ---- 非同期更新の準備 ----
        self.w[:, :] = 0

        # ---- 可能的遷移の生成 ----
        targets = self._generate_potential_transitions()

        # ---- 人気度の初期値（全員未移動の状態）----
        unique_sites = self._build_unique_sites(targets)
        pop_counts = self._initial_counts(unique_sites)   # 動的に更新していく

        occupancy = self.occupancy.copy()
        new_position = self.position.copy()

        dx_Nf, dy_Nf = self._square_neighbor_offsets(R_Nf)

        order = self.rng.permutation(N)

        for k in order:
            x0, y0 = self.position[k]
            k_targets = targets[k]
            tx, ty = k_targets[:, 0], k_targets[:, 1]
            vacant_mask = occupancy[tx, ty] == -1

            nx, ny = None, None

            # --- 式(4): 相互予期（その時点の人気度 pop_counts を使う）---
            if vacant_mask.any():
                cand_counts = np.where(vacant_mask, pop_counts[tx, ty], -1)
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

            # --- 人気度の再計算（差分更新）---
            # 個体kは移動が確定したので、kが狙っていた格子点の人気度から寄与を取り除く。
            # 空き/占有の変化（元の場所が空く、移動先が埋まる）は occupancy 側で
            # 反映されているので、次の個体は vacant_mask を通して正しく扱う。
            sites = unique_sites[k]
            pop_counts[sites // S, sites % S] -= 1

        self.position = new_position
        self.occupancy = occupancy

        # ---- 速度一致 ----
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
