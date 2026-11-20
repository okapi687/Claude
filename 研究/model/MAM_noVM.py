import numpy as np


class SoldierCrabSwarm_noVM:
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
    c : int, default 2
        人気度の閾値 ζ ≧ c で相互予期による移動が発生する。
    seed : int or None
        乱数シード（再現性のため）
        この値を決めることで、同じ初期配置・同じ乱数系列でシミュレーションを再現できる。
    omega : int
        速度を合わせるために使用
    """

    def __init__(self, N=100, S=50, L=4.0, P=15, alpha=0.5,
                 R_Nf=2, R_M=3, c=2, omega=5.0, seed=None):

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
        self.omega = omega
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
        targets = self._generate_potential_transitions()          
        target_counts = self._compute_target_counts(targets)      

        # occupancy は __init__ 時点のものを一旦コピーして、このステップ中の移動に応じて逐次更新していく
        occupancy = self.occupancy.copy()

        # ---- 遷移 ----
        new_position = self.position.copy() # 今のポジションをコピー
        moved = np.zeros(N, dtype=bool) 
        # 各エージェントが今ステップ内で移動済みか（値はTrueかFalse）
        # False → まだ動いていない、True → もう動いた

        # === 相互予期 ===
        while True:
            # ループの終了条件：動いていないエージェントがいなくなったら終了
            unmoved = np.flatnonzero(~moved)
            if unmoved.size == 0:
                break

            # 各未移動エージェントについて、自分の遷移先の中で「現在空いている」かつ「最も人気度が高い」候補を1つ求める
            best_count = np.full(unmoved.size, -1, dtype=int)       # 人気度の最大地点を保存
            best_target = np.zeros((unmoved.size, 2), dtype=int)    # その人気地点の座標を保存

            for idx, k in enumerate(unmoved):
                k_targets = targets[k]
                tx, ty = k_targets[:, 0], k_targets[:, 1]
                vacant_mask = occupancy[tx, ty] == -1   # 空いているかチェック
                if not vacant_mask.any():
                    continue
                cand_counts = np.where(vacant_mask, target_counts[tx, ty], -1)  # 人気度を取ってくる（ただし埋まっていたら-1）
                m = cand_counts.max()   # 最大値を入れる
                if m >= c:  # 人気度が閾値以上の候補があれば、最大値の中からランダムに1つ選ぶ
                    tie_idx = np.flatnonzero(cand_counts == m)  
                    chosen_i = self.rng.choice(tie_idx)     # 同率ならランダム
                    best_count[idx] = m
                    best_target[idx] = k_targets[chosen_i]
            
            # 遷移可能な（人気度≧cの候補を少なくとも1つ持つ）エージェントの一覧
            eligible_idx = np.flatnonzero(best_count >= c)

            # 遷移できるエージェントがもういなければ、相互予期フェーズを終了
            if eligible_idx.size == 0:
                break

            # 遷移可能なエージェントの中から1体だけランダムに選ぶ（エージェント主導）
            chosen_idx = self.rng.choice(eligible_idx)
            chosen_k = unmoved[chosen_idx]
            nx, ny = best_target[chosen_idx]   # そのエージェント自身にとって最も人気の高い候補地
            x0, y0 = self.position[chosen_k]

            occupancy[x0, y0] = -1
            occupancy[nx, ny] = chosen_k
            self.w[x0, y0] = 1      # 元いた場所は「更新済み」とマーク
            new_position[chosen_k] = (nx, ny)
            moved[chosen_k] = True

        # === 追従（following） ===
        # 「まだ移動していないエージェントがいるか？」→「近傍内に、仲間の移動で今空いたばかりの地点があるか？」を繰り返し確認
        while True:
            # ループの終了条件：動いていないエージェントがいなくなったら終了
            unmoved = np.flatnonzero(~moved)
            if unmoved.size == 0:
                break

            candidates = [] # 追従できるエージェント一覧
            # 移動していないエージェントについて追従できるか探す
            for k in unmoved:
                x0, y0 = self.position[k]
                dx, dy = self._square_neighbor_offsets(R_Nf)
                nbx = np.mod(x0 + dx, S)
                nby = np.mod(y0 + dy, S)
                #更新済みかつ今は空いているサイトを探す
                is_freed = (self.w[nbx, nby] == 1) & (occupancy[nbx, nby] == -1) 
                if is_freed.any():
                    candidates.append(k) # 追従一覧に追加

            # 追従できるエージェントがもう無ければ、追従フェーズを終了
            if not candidates:
                break

            # 移動するエージェントを一匹選ぶ
            chosen_k = self.rng.choice(candidates)

            # 追従するエージェントの近傍内で、仲間が移動して空いた地点の中からランダムに1つ選ぶ
            x0, y0 = self.position[chosen_k]
            dx, dy = self._square_neighbor_offsets(R_Nf)
            nbx = np.mod(x0 + dx, S)
            nby = np.mod(y0 + dy, S)
            is_freed = (self.w[nbx, nby] == 1) & (occupancy[nbx, nby] == -1)
            free_idx = np.flatnonzero(is_freed)
            chosen_site = self.rng.choice(free_idx)
            nx, ny = nbx[chosen_site], nby[chosen_site]

            # 選ばれた地点に遷移する
            occupancy[x0, y0] = -1
            occupancy[nx, ny] = chosen_k
            self.w[x0, y0] = 1
            new_position[chosen_k] = (nx, ny)
            moved[chosen_k] = True

        # === 自由移動（free moving） ===
        # 残った未移動エージェントを1体ずつランダムな順で、自分の可能的遷移の中から空いている地点へ移動させる
        unmoved = self.rng.permutation(np.flatnonzero(~moved)) # 移動していない個体をランダムな順に並べる

        # 一匹ずつ処理をする
        for k in unmoved:
            x0, y0 = self.position[k]
            k_targets = targets[k]
            tx, ty = k_targets[:, 0], k_targets[:, 1] # x座標とy座標を分ける
            vacant_mask = occupancy[tx, ty] == -1 # 空いているかチェック

            # 空いている候補にランダムで移動、空いてなかったらその場に留まる
            if vacant_mask.any():
                vacant_idx = np.flatnonzero(vacant_mask)
                chosen_i = self.rng.choice(vacant_idx)
                nx, ny = k_targets[chosen_i]
            else:
                # 例外処理
                nx, ny = x0, y0

            occupancy[x0, y0] = -1
            occupancy[nx, ny] = k
            self.w[x0, y0] = 1
            new_position[k] = (nx, ny)
            moved[k] = True

        self._update_principal_vector(new_position)
        self.position = new_position
        self.occupancy = occupancy

        # ---- 次のステップへの更新 ----
        self.t += 1

    def _update_principal_vector(self, new_position):
        """
        論文2 (Murakami, Niizato & Gunji, 2017) の主ベクトル更新
        q_i,t = u_i,t + omega * spv_i,t
        theta_i,t+1 = angle(q_i,t)
        """
        S, omega = self.S, self.omega

        # 実際の変位ベクトル v_i,t（トーラス境界を考慮した最短距離で計算）
        raw_diff = (new_position - self.position).astype(float)
        raw_diff = (raw_diff + S / 2) % S - S / 2   # 端をまたいだ場合の補正

        norm = np.linalg.norm(raw_diff, axis=1)
        u = np.zeros_like(raw_diff)
        moved = norm > 1e-9
        u[moved] = raw_diff[moved] / norm[moved, None]
        # まったく動けなかった個体(自由移動ですら空き地がなかった場合)は
        # 直前の主ベクトルの向きをそのまま「動いた向き」とみなす
        u[~moved] = np.stack([np.cos(self.theta[~moved]), np.sin(self.theta[~moved])], axis=1)

        spv = np.stack([np.cos(self.theta), np.sin(self.theta)], axis=1)
        q = u + omega * spv

        qnorm = np.linalg.norm(q, axis=1)
        # 万が一 u と spv が真逆でq=0になった場合は角度を維持
        new_theta = np.where(qnorm > 1e-9, np.arctan2(q[:, 1], q[:, 0]), self.theta)

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