import numpy as np

class SoldierCrabSwarm_noVM:

    def __init__(self, N=100, S=50, L=6.0, P=40, alpha=0.2, omega=5.0, c=2, seed=None):

        # 入力チェック
        if N > S * S:
            raise ValueError(
                f"N({N}) が格子点の総数 S*S({S*S}) を超えています。"
                " 1つの格子点に最大1体しか配置できません。"
            )
        
        # ---- パラメータの設定 ----
        self.N = N
        self.S = S
        self.L = L
        self.P = P
        self.alpha = alpha
        self.omega = omega
        self.c = c
        self.seed = seed

        # ---- 乱数生成 ----
        self.rng = np.random.default_rng(seed)

        # ---- ステップ数の初期化 ----
        self.t = 0

        # --- 初期配置 ----
        all_sites = self.rng.permutation(S * S)[:N]
        xs = all_sites % S
        ys = all_sites // S

        # P(k, t) = (x, y)  現在地の記録
        # position[k] = [x, y]
        self.position = np.stack([xs, ys], axis=1).astype(int)  

        # 各エージェントの基本ベクトルの向きをランダムに初期化
        self.theta = self.rng.uniform(0.0, 2 * np.pi, size=N)

        # wanderer が空けたサイトを記録
        self.w = np.zeros((S, S), dtype=int)    # 要素がすべて0の配列

        # 格子点の占有状況を管理する配列（衝突回避の確認や人気度計算に使用）
        # occupancy[x, y] = k のインデックス, 空きなら -1
        # 何番目の個体がどこにいるか保存される
        self.occupancy = -np.ones((S, S), dtype=int)    # 要素がすべて-1の配列
        for k in range(N):
            x, y = self.position[k]
            self.occupancy[x, y] = k

    # ------------------------------------------------------------------
    # 補助関数
    # ------------------------------------------------------------------

    def _generate_potential_transitions(self):
        """可能的遷移の生成"""

        N, P, S, L, alpha = self.N, self.P, self.S, self.L, self.alpha

        # eta, xi をエージェント×遷移ごとに生成
        # i=0 は基本ベクトル: η=1, ξ=0
        eta = self.rng.uniform(0.0, 1.0, size=(N, P))
        xi = self.rng.uniform(-alpha * np.pi, alpha * np.pi, size=(N, P))
        eta[:, 0] = 1.0     # i=0 は基本ベクトル: η=1, ξ=0
        xi[:, 0] = 0.0

        # theta　を縦ベクトルに変換する
        theta = self.theta[:, None]

        # 遷移ベクトルを生成
        # ここでは小数点以下切り捨て（floor）で整数にする
        vx = np.round(L * eta * np.cos(theta + xi)).astype(int)
        vy = np.round(L * eta * np.sin(theta + xi)).astype(int)

        # 遷移先の座標を計算
        tx = self.position[:, 0:1] + vx
        ty = self.position[:, 1:2] + vy

        # トーラス境界条件でラップ
        tx = np.mod(tx, S)
        ty = np.mod(ty, S)

        # 標的座標と遷移ベクトルを返す
        targets = np.stack([tx, ty], axis=-1) 
        vecs = np.stack([vx, vy], axis=-1)
        return targets, vecs

    def _compute_popularity(self, targets):
        """人気度の計算"""

        N, P, _ = targets.shape
        S = self.S

        # (エージェント番号, x, y) の組を作る
        agent_idx = np.repeat(np.arange(N), P)  # 各座標にどのエージェントのものかというラベルをつける
        flat_x = targets[:, :, 0].ravel()       # x座標を1次元配列に変換
        flat_y = targets[:, :, 1].ravel()       # y座標を1次元配列に変換

        # 同一エージェントが同じサイトを複数の遷移で重複して狙っている分を除去
        triples = np.stack([agent_idx, flat_x, flat_y], axis=1) 
        unique_triples = np.unique(triples, axis=0)

        # 人気度の計算
        counts = np.zeros((S, S), dtype=int)
        np.add.at(counts, (unique_triples[:, 1], unique_triples[:, 2]), 1)     # 2列目と3列目の座標を使ってカウントする
        return counts

    def _update_principal_vector(self, old_position, new_position):
        """
        V = P(t+1) - P(t),  q = V_u + ω·v_u(k,t;0),  θ(t+1) = angle(q)
        v_u(k,t;0) は整数化した基本ベクトルの単位ベクトル。
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
        pn = np.linalg.norm(self._pv_int, axis=1)
        ok = pn > 0
        spv[ok] = self._pv_int[ok] / pn[ok, None]
 
        q = u + self.omega * spv
        self.theta = np.arctan2(q[:, 1], q[:, 0])

    # ------------------------------------------------------------------
    # 1ステップ
    # ------------------------------------------------------------------

    def step(self):
        """1タイムステップ分の更新を行う"""

        N, S, c = self.N, self.S, self.c
        
        # 非同期更新の準備
        self.w[:, :] = 0

        # 現在の座標をコピーしておく
        old_position = self.position.copy()

        # 可能的遷移の生成と人気度の計算
        targets, vecs = self._generate_potential_transitions()
        self._pv_int = vecs[:, 0, :].copy()      # ← これが足りない（式(13)の整数化した基本ベクトル）
        popularity = self._compute_popularity(targets)

        # 現状態を保存
        occupancy = self.occupancy.copy()
        vacant0 = self.occupancy == -1               # ステップ開始時に空いているサイト
        taken = np.zeros((S, S), dtype=bool)         # 更新済み個体が入ったサイト U_N
        new_position = self.position.copy()

        # 人気度が同率のときの「基本ベクトルとの向きの近さ」
        ang = None
        a = np.arctan2(vecs[:, :, 1], vecs[:, :, 0])
        ang = np.abs((a - a[:, :1] + np.pi) % (2 * np.pi) - np.pi)

        # 遷移を表す関数
        def move(k, nx, ny):
            x0, y0 = self.position[k]
            occupancy[x0, y0] = -1
            occupancy[nx, ny] = k
            new_position[k] = (nx, ny)

        order = [int(k) for k in self.rng.permutation(N)]   # 非同期更新の順序
        wanderer = np.zeros(N, dtype=bool)                  # どの個体がwandererになったかを記録

        # ---- 相互予期 ----
        for k in order:
            tx = targets[k, :, 0]
            ty = targets[k, :, 1]
            ok = vacant0[tx, ty] & ~taken[tx, ty]
            z = np.where(ok, popularity[tx, ty], -1)
            m = z.max()
            if m < c:
                continue
            cand = np.flatnonzero(z == m) # 候補サイトのインデックス
            a_c = ang[k, cand]
            cand = cand[a_c <= a_c.min() + 1e-9]
            i = int(self.rng.choice(cand))
            nx, ny = tx[i], ty[i]
            self.w[self.position[k][0], self.position[k][1]] = 1    # 式(9): 空けたサイトを記録
            move(k, nx, ny)
            taken[nx, ny] = True
            wanderer[k] = True

        # ---- 追従 ----
        rest = []
        for k in order:
            if wanderer[k]:
                continue
            tx, ty = targets[k, :, 0], targets[k, :, 1]
            ok = (self.w[tx, ty] == 1) & (occupancy[tx, ty] == -1)
            if ok.any():
                sites = np.unique(tx[ok] * S + ty[ok])
                s = int(self.rng.choice(sites))
                move(k, s // S, s % S)
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
                move(k, s // S, s % S)
                taken[s // S, s % S] = True
            else:
                move(k, *self.position[k])

        self.position = new_position
        self.occupancy = occupancy

        # 速度更新
        self._update_principal_vector(old_position, new_position)

        # 時刻更新
        self.t += 1

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