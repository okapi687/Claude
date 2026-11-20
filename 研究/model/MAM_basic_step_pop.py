import numpy as np

class SoldierCrabSwarm_step_pop:
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
    N_W : int, default 1
        壁監視近傍の半径
    c : int, default 2
        人気度の閾値 ζ ≧ c で相互予期による移動が発生する。
    seed : int or None
        乱数シード（再現性のため）
        この値を決めることで、同じ初期配置・同じ乱数系列でシミュレーションを再現できる。
    """

    def __init__(self, N=100, S=50, L=4.0, P=15, alpha=0.5, R_Nf=2, R_M=3, c=2, seed=None):

        # ---- 乱数生成 ----
        self.rng = np.random.default_rng(seed)

        # ---- 格子空間の設定 ----
        # S×Sの2次元格子（座標平面）、境界条件はトーラス構造
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
        # エラー処理
        if N > S * S:
            raise ValueError(
                f"N({N}) が格子点の総数 S*S({S*S}) を超えています。"
                " 1つの格子点に最大1体しか配置できません。"
            )

        # S*S個の格子点インデックスから重複なくN個を選ぶ
        all_sites = self.rng.permutation(S * S)[:N]
        xs = all_sites % S
        ys = all_sites // S

        # P(k, t) = (x, y)  現在地の記録
        # position[k] = [x, y]
        self.position = np.stack([xs, ys], axis=1).astype(int)  

        # 各エージェントの基本ベクトルの向きをランダムに初期化
        self.theta = self.rng.uniform(0.0, 2 * np.pi, size=N)   # 一様乱数の生成

        # 非同期更新のためのフラグ w(x, y; t)（格子点ごと）
        self.w = np.zeros((S, S), dtype=int)    # 要素がすべて0の配列

        # 格子点の占有状況を管理する配列（衝突回避の確認や人気度計算に使用）
        # occupancy[x, y] = k のインデックス, 空きなら -1
        # 何番目の個体がどこにいるか保存される
        self.occupancy = -np.ones((S, S), dtype=int)    # 要素がすべて-1の配列
        for k in range(N):
            x, y = self.position[k]
            self.occupancy[x, y] = k

    # ------------------------------------------------------------------
    # 各ステップでの動作
    # ------------------------------------------------------------------

    def _square_neighbor_offsets(self, R):
        """半径Rの正方形近傍のオフセット (dx, dy) 一覧を返す（(2R+1)^2個、中心含む）"""
        rng = np.arange(-R, R + 1)      # 一定間隔で数字を並べた配列[-R, -R+1, ..., 0, ..., R-1, R]
        dx, dy = np.meshgrid(rng, rng, indexing="ij")   # 与えられた配列から2次元の座標を作る
        return dx.ravel(), dy.ravel()   # 1次元配列に変換して返す

    def _generate_potential_transitions(self):
        """
        可能的遷移の生成
        """
        N, P, S, L, alpha = self.N, self.P, self.S, self.L, self.alpha

        # eta, xi をエージェント×遷移ごとに生成
        eta = self.rng.uniform(0.0, 1.0, size=(N, P))
        xi = self.rng.uniform(-alpha * np.pi, alpha * np.pi, size=(N, P))

        # i=0 は基本ベクトル（principal vector）: eta_0=1.0, xi_0=0.0
        eta[:, 0] = 1.0
        xi[:, 0] = 0.0

        # theta　を縦ベクトルに変換する
        theta = self.theta[:, None]

        # 遷移ベクトルを生成
        # ここでは小数点以下切り捨て（floor）で整数にする
        vx = np.floor(L * eta * np.cos(theta + xi)).astype(int)
        vy = np.floor(L * eta * np.sin(theta + xi)).astype(int)

        # 遷移先の座標を計算
        tx = self.position[:, 0:1] + vx
        ty = self.position[:, 1:2] + vy

        # トーラス境界条件でラップ
        tx = np.mod(tx, S)
        ty = np.mod(ty, S)

        # # ----- デバッグ用 -----
        # dx = vx.ravel()
        # dy = vy.ravel()

        # # 長さ0のベクトルは除外
        # mask = (dx != 0) | (dy != 0)

        # angles = np.degrees(np.arctan2(dy[mask], dx[mask]))

        # # 保存
        # self.debug_generated_angles = angles

        # zero_ratio = np.mean((vx == 0) & (vy == 0))
        # print(f"(0,0) ratio = {zero_ratio:.3f}")

        # from collections import Counter

        # counter = Counter(zip(vx.ravel(), vy.ravel()))

        # print(counter.most_common(30))

        targets = np.stack([tx, ty], axis=-1)
        return targets

    def _compute_target_counts(self, targets):
        """
        人気度の計算
        """
        N, P, _ = targets.shape
        S = self.S

        # (エージェント番号, x, y) の組を作る
        agent_idx = np.repeat(np.arange(N), P)  # 各座標にどのエージェントのものかというラベルをつける
        flat_x = targets[:, :, 0].ravel()       # x座標を1次元配列に変換
        flat_y = targets[:, :, 1].ravel()       # y座標を1次元配列に変換
        triples = np.stack([agent_idx, flat_x, flat_y], axis=1)     # 3列の配列にまとめる

        # 同一エージェントが同じサイトを複数の遷移で重複して狙っている分を除去
        unique_triples = np.unique(triples, axis=0)

        # # ===== ここを追加 =====
        # # 現在エージェントが存在する格子は popularity の計算対象から除外
        # vacant = self.occupancy[
        #     unique_triples[:, 1],
        #     unique_triples[:, 2]
        # ] == -1

        # unique_triples = unique_triples[vacant]
        # =====================

        # 重複除去後の (x, y) をサイトごとに数える
        counts = np.zeros((S, S), dtype=int)
        np.add.at(counts, (unique_triples[:, 1], unique_triples[:, 2]), 1)     # 2列目と3列目の座標を使ってカウントする
        return counts

    def step(self):
        """
        1タイムステップ分の更新を行う
        """

        N, S, c, R_Nf = self.N, self.S, self.c, self.R_Nf

        # ---- 非同期更新の準備 ----
        self.w[:, :] = 0

        # ---- 可能的遷移の生成 と 人気度(生カウント)の事前計算 ----
        # z(x, y; t) は t 時点の状態で1回だけ定義され、ステップ中は変化しない（論文 式3）
        targets = self._generate_potential_transitions()
        target_counts = self._compute_target_counts(targets)

        # occupancy は __init__ 時点のものを一旦コピーして、このステップ中の移動に応じて逐次更新していく
        occupancy = self.occupancy.copy()

        # ---- 遷移 ----
        new_position = self.position.copy()  # 今のポジションをコピー
        moved = np.zeros(N, dtype=bool)

        dx_Nf, dy_Nf = self._square_neighbor_offsets(R_Nf)  # 追従用の近傍オフセットは使い回す

        # === 非同期更新（相互予期 → 追従 → 自由移動 を1個体ずつ判定） ===
        # 「動いていない個体からランダムに1体選ぶ」を全個体が動くまで繰り返す
        order = self.rng.permutation(N)  # N体分のランダムな更新順（毎ステップ変わる）

        for k in order:
            x0, y0 = self.position[k]
            k_targets = targets[k]
            tx, ty = k_targets[:, 0], k_targets[:, 1]
            vacant_mask = occupancy[tx, ty] == -1  # 自分の遷移先が現在空いているか

            nx, ny = None, None

            # --- 式(4): 相互予期 ---
            # 自分の遷移先の中で、空いていてかつ人気度が最大のものを求める
            if vacant_mask.any():
                cand_counts = np.where(vacant_mask, target_counts[tx, ty], -1)
                m = cand_counts.max()
                if m >= c:
                    tie_idx = np.flatnonzero(cand_counts == m)
                    chosen_i = self.rng.choice(tie_idx)  # 同率ならランダム
                    nx, ny = k_targets[chosen_i]

            # --- 式(5): 追従 ---
            # 相互予期が成立しなかった個体のみ、追従の資格を判定する
            # if nx is None:
            #     nbx = np.mod(x0 + dx_Nf, S)
            #     nby = np.mod(y0 + dy_Nf, S)
            #     is_freed = (self.w[nbx, nby] == 1) & (occupancy[nbx, nby] == -1)
            #     # 自分のpotential transitionsに含まれるマスだけに限定
            #     own_targets = set(map(tuple, k_targets))
            #     candidate_idx = [
            #         i for i in np.flatnonzero(is_freed)
            #         if (nbx[i], nby[i]) in own_targets
            #     ]
            #     if candidate_idx:
            #         chosen_site = self.rng.choice(candidate_idx)
            #         nx, ny = nbx[chosen_site], nby[chosen_site]
            if nx is None:
                nbx = np.mod(x0 + dx_Nf, S)
                nby = np.mod(y0 + dy_Nf, S)
                is_freed = (self.w[nbx, nby] == 1) & (occupancy[nbx, nby] == -1)
                if is_freed.any():
                    free_idx = np.flatnonzero(is_freed)
                    chosen_site = self.rng.choice(free_idx)
                    nx, ny = nbx[chosen_site], nby[chosen_site]

            # --- 式(6): 自由移動 ---
            # 相互予期・追従のどちらも成立しなかった個体は、空いている遷移先からランダムに1つ選ぶ
            if nx is None:
                if vacant_mask.any():
                    vacant_idx = np.flatnonzero(vacant_mask)
                    chosen_i = self.rng.choice(vacant_idx)
                    nx, ny = k_targets[chosen_i]
                else:
                    # 空いている遷移先が1つもなければその場に留まる（例外処理）
                    nx, ny = x0, y0

            # --- 実際に遷移させる ---
            occupancy[x0, y0] = -1
            occupancy[nx, ny] = k
            self.w[x0, y0] = 1  # 元いた場所は「更新済み（空いた）」とマーク
            new_position[k] = (nx, ny)
            moved[k] = True

            # # 実際に移動したベクトル
            # dx_move = nx - x0
            # dy_move = ny - y0

            # # トーラス補正
            # if dx_move > S//2:
            #     dx_move -= S
            # elif dx_move < -S//2:
            #     dx_move += S

            # if dy_move > S//2:
            #     dy_move -= S
            # elif dy_move < -S//2:
            #     dy_move += S

            # angle = np.degrees(np.arctan2(dy_move, dx_move))

            # # 保存
            # if not hasattr(self, "debug_selected_angles"):
            #     self.debug_selected_angles = []

            # self.debug_selected_angles.append(angle)

        self.position = new_position
        self.occupancy = occupancy

        # ---- 次のステップへの更新 ----
        # 速度一致
        self._velocity_matching()

        self.t += 1


    def _velocity_matching(self):
        """
        速度一致
        """
        N, S, R_M = self.N, self.S, self.R_M
        dx, dy = self._square_neighbor_offsets(R_M)

        new_theta = np.empty(N, dtype=float)

        # 各格子点にどのエージェントがいるか
        for k in range(N):
            x0, y0 = self.position[k]
            nbx = np.mod(x0 + dx, S)
            nby = np.mod(y0 + dy, S)
            occupants = self.occupancy[nbx, nby]
            occupants = occupants[occupants != -1]

            # occupants = occupants[occupants != k]

            # 近傍に仲間がいない場合はそのまま
            if occupants.size == 0:
                new_theta[k] = self.theta[k]
                continue

            angles = self.theta[occupants]
            # 循環平均（ベクトル平均をとった後角度に戻す）
            mean_vec = np.array([np.cos(angles).mean(), np.sin(angles).mean()])
            if np.allclose(mean_vec, 0.0):
                new_theta[k] = self.theta[k]
            else:
                new_theta[k] = np.arctan2(mean_vec[1], mean_vec[0]) # arctanで角度を求める

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