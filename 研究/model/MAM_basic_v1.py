import numpy as np


class SoldierCrabSwarm_v1:
    """
    MAM_basic_v1.py — 11/18 時点で、論文の図に一番近かった設定をまとめた版。

    論文1と論文2は別のモデルとして扱う（10/31 のメモ）。そのため model 引数で切り替える。
    どちらも、これまでの最新版と同じ処理を移したもの。旧版の挙動に戻すオプションも残している。

      model="paper1" : 論文1（Gunji et al., Robust swarm of soldier crabs）
                       MAM_basic_ex_v2 と同じ処理。
                       相互予期（全員） → 追従（全員） → 自由移動（全員） → 速度一致
      model="paper2" : 論文2（Murakami et al. 2017, Sci. Rep. 7:46447 + 付録 S1）
                       MAM_basic_Murakami と同じ処理（付録の式(1)〜(13)）。速度一致はない。
                       基本ベクトルの向きは、自分の直前の移動で更新する。

    ■ 既定値に選んだ設定と、その理由（数値は 研究/paper/検証レポート.md を参照）
      paper1
        int_mode="floor"     : 論文の Int の定義（X ≤ x < X+1）どおり。
                               round にすると、図10・11・13・14 の整列度が論文からさらに離れた。
        follow_chain=False   : 追従で動けるのは「相互予期で空いたサイト」だけ（論文1 p.7）。
                               これで P=2 で群れが崩れる現象（図8）が出るようになった。
        ma_rule="sequential" : これまでのモデルと同じ。
      paper2
        int_mode="round"     : 図S2 のルールの割合（青: 63/5/32%）と図3 の青の整列度が合う。
                               floor だと (-,-) 向きの偏りがたまって、どの条件でも整列度が 0.96 以上になる。
        omega_on="velocity"  : q = v_u(k,t;0) + ω·V_u（ω が直前の移動の重み）。
                               図S3 の説明文（ω が小さいと前の基本ベクトルのまま、大きいと
                               移動の向きに近づく）に合わせた読み方。本文の式どおりの
                               q = V_u + ω·v_u は omega_on="principal"（Murakami の既定）。
                               velocity の方が、図S3 の ω=0（0.1 / 2.0）が一致して、
                               赤（P=40, α=0.2）の整列度も論文に近づいた。

    ■ 旧版からの修正（結果は変わらない）
      - Murakami の _compute_popularity にあった「今いる個体のセルを除外」の処理は、
        数えるのに使っていない配列を絞り込んでいたので効いていなかった。
        占有サイトは移動先を選ぶときに除いているので、その処理は消した。
        dedupe_votes=False のときに NameError になる問題も直した。
      - round は np.round（ちょうど .5 は偶数丸め）。ちょうど .5 はほぼ出ないので影響はない。

    ■ 旧版と同じ結果に戻す設定（同じ seed なら同じ軌跡になる）
      MAM_basic_ex_v2   → SoldierCrabSwarm_v1(model="paper1", ...)（既定のまま）
      MAM_basic_Murakami → SoldierCrabSwarm_v1(model="paper2", omega_on="principal", ...)

    Parameters
    ----------
    model : {"paper1", "paper2"}
    N : int, default 100            個体数
    S : int, default 50             格子の一辺（S×S のトーラス）
    L : float or None               基本ベクトルの長さ。None なら paper1 は 4、paper2 は 6
    P : int or None                 可能的遷移の数。None なら paper1 は 15、paper2 は 40
    alpha : float or None           変異角 α（ξ は [-απ, απ] の一様乱数）。None なら 0.5 / 0.2
    alpha_scale : float, default 1.0
        ξ の範囲を [-alpha_scale·απ, alpha_scale·απ] にする。1.0 が式(1)・付録の式(3)の文面どおり。
        2.0 は「α=0.5 でほぼ全方向」（論文2 p.4）に合わせた読み方（検証レポート 8章）
    c : int, default 2              人気度の閾値（ζ ≧ c で相互予期が成立）
    int_mode : str or None          Int の方法。None なら paper1 は "floor"、paper2 は "round"
                                    "floor" / "ceil" / "round" / "trunc" / "stochastic"
    seed : int or None              乱数シード

    paper1 だけで使うもの
    R_Nf : int, default 2           追従者近傍の半径
    R_M : int, default 3            速度一致近傍の半径
    follow_chain : bool, default False
        True にすると、追従した個体が空けたサイトにも追従できる（MAM_basic_ex の挙動）
    free_wait : bool, default False
        True にすると、自由移動で行き先がふさがっている個体が、前の個体が動いて空くまで待つ
    ma_rule : {"sequential", "literal"}, default "sequential"
        "literal" は論文1 p.7 の「同じサイトを狙ったら1体だけ、残りは2番目に人気のサイトへ」を
        文字どおりに実装したもの

    paper2 だけで使うもの
    omega : float, default 5.0      ω
    omega_on : {"velocity", "principal"}, default "velocity"
        "velocity"  : q = v_u + ω·V_u
        "principal" : q = V_u + ω·v_u（付録の式(13)どおり）
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

    MODELS = ("paper1", "paper2")
    INT_MODES = ("floor", "ceil", "round", "trunc", "stochastic")
    DEFAULTS = {
        "paper1": dict(L=4.0, P=15, alpha=0.5, int_mode="floor"),
        "paper2": dict(L=6.0, P=40, alpha=0.2, int_mode="round"),
    }

    def __init__(self, model="paper1", N=100, S=50, L=None, P=None, alpha=None, c=2,
                 int_mode=None, seed=None,
                 # paper1
                 R_Nf=2, R_M=3, follow_chain=False, free_wait=False, ma_rule="sequential",
                 # paper2
                 omega=5.0, omega_on="velocity", dedupe_votes=True, spv_mode="int_vector",
                 static_vacancy=True, tiebreak="angle",
                 # 共通
                 alpha_scale=1.0):

        if model not in self.MODELS:
            raise ValueError(f"model は {self.MODELS} のどれか（指定値: {model!r}）")
        d = self.DEFAULTS[model]
        L = d["L"] if L is None else L
        P = d["P"] if P is None else P
        alpha = d["alpha"] if alpha is None else alpha
        int_mode = d["int_mode"] if int_mode is None else int_mode

        if int_mode not in self.INT_MODES:
            raise ValueError(f"int_mode は {self.INT_MODES} のどれか（指定値: {int_mode!r}）")
        if ma_rule not in ("sequential", "literal"):
            raise ValueError(f'ma_rule は "sequential" か "literal"（指定値: {ma_rule!r}）')
        if omega_on not in ("velocity", "principal"):
            raise ValueError(f'omega_on は "velocity" か "principal"（指定値: {omega_on!r}）')
        if spv_mode not in ("int_vector", "angle"):
            raise ValueError(f'spv_mode は "int_vector" か "angle"（指定値: {spv_mode!r}）')
        if tiebreak not in ("angle", "random"):
            raise ValueError(f'tiebreak は "angle" か "random"（指定値: {tiebreak!r}）')
        if N > S * S:
            raise ValueError(f"N({N}) が格子点の総数 S*S({S*S}) を超えています。")

        self.model = model
        self.rng = np.random.default_rng(seed)
        self.N, self.S, self.L, self.P, self.alpha, self.c = N, S, L, P, alpha, c
        self.int_mode, self.seed = int_mode, seed
        self.alpha_scale = alpha_scale
        self.R_Nf, self.R_M = R_Nf, R_M
        self.follow_chain, self.free_wait, self.ma_rule = follow_chain, free_wait, ma_rule
        self.omega, self.omega_on = omega, omega_on
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

        # 空いたサイトの印。w：誰かが動いて空いた、w_ma：相互予期で空いた（論文2 の χ もこちら）
        self.w = np.zeros((S, S), dtype=int)
        self.w_ma = np.zeros((S, S), dtype=int)

        self.last_displacement = np.zeros((N, 2), dtype=int)
        self.last_rule_counts = (0, 0, 0, 0)
        self.move_counts = {"MA": 0, "follow": 0, "free": 0, "stay": 0}

        self._dx_Nf, self._dy_Nf = self._square_neighbor_offsets(R_Nf)
        self._dx_M, self._dy_M = self._square_neighbor_offsets(R_M)

    # ==================================================================
    # 共通の処理
    # ==================================================================

    @staticmethod
    def _square_neighbor_offsets(R):
        """半径 R の正方形近傍のオフセット (dx, dy)（(2R+1)^2 個、中心を含む）"""
        r = np.arange(-R, R + 1)
        dx, dy = np.meshgrid(r, r, indexing="ij")
        return dx.ravel(), dy.ravel()

    def _to_int(self, x):
        """論文の Int。int_mode に従って実数を整数にする"""
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
        可能的遷移 v(k,t;i) = Int(L·η·(cos(θ+ξ), sin(θ+ξ)))、η∈[0,1]、ξ∈[-απ, απ]
        （alpha_scale を指定すると ξ∈[-alpha_scale·απ, alpha_scale·απ]）
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

    def _compute_popularity(self, targets, dedupe=True):
        """
        人気度 ζ(x,y;t)：そのサイトを遷移先に含む個体の数。
        dedupe=True なら同じ個体が同じサイトを何本選んでいても1票。
        占有されているサイトの除外は、移動先を選ぶときに行う。
        """
        N, P, _ = targets.shape
        flat_x = targets[:, :, 0].ravel()
        flat_y = targets[:, :, 1].ravel()
        if dedupe:
            triples = np.unique(np.stack([np.repeat(np.arange(N), P), flat_x, flat_y], axis=1), axis=0)
            flat_x, flat_y = triples[:, 1], triples[:, 2]
        z = np.zeros((self.S, self.S), dtype=int)
        np.add.at(z, (flat_x, flat_y), 1)
        return z

    def step(self):
        """1ステップ進める"""
        old_position = self.position.copy()
        if self.model == "paper1":
            counts = self._step_paper1()
        else:
            counts = self._step_paper2()

        d = self.position - old_position
        self.last_displacement = (d + self.S // 2) % self.S - self.S // 2
        self.last_rule_counts = counts
        for key, n in zip(("MA", "follow", "free", "stay"), counts):
            self.move_counts[key] += n

        if self.model == "paper1":
            self._velocity_matching()
        else:
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

    # ==================================================================
    # 論文1（MAM_basic_ex_v2 と同じ）
    # ==================================================================

    def _step_paper1(self):
        N = self.N
        self.w[:, :] = 0
        self.w_ma[:, :] = 0

        targets, _ = self._generate_potential_transitions()
        z = self._compute_popularity(targets, dedupe=True)

        occupancy = self.occupancy.copy()
        new_position = self.position.copy()
        moved = np.zeros(N, dtype=bool)
        cnt = [0, 0, 0, 0]

        def apply_move(k, nx, ny, by_MA=False):
            x0, y0 = self.position[k]
            occupancy[x0, y0] = -1
            occupancy[nx, ny] = k
            self.w[x0, y0] = 1
            if by_MA:
                self.w_ma[x0, y0] = 1
            new_position[k] = (nx, ny)
            moved[k] = True

        # === 相互予期 ===
        if self.ma_rule == "literal":
            cnt[0] += self._literal_MA_phase(targets, z, occupancy, apply_move)
        else:
            # ランダム順に1体ずつ、空いている遷移先の中で人気度が最大（≧c）のサイトへ動く。
            # 誰かが動くと元の場所が空くので、誰も動かなくなるまで繰り返す
            while True:
                any_moved = False
                for k in self.rng.permutation(np.flatnonzero(~moved)):
                    dest = self._best_MA_target(targets[k], z, occupancy)
                    if dest is not None:
                        apply_move(k, dest[0], dest[1], by_MA=True)
                        cnt[0] += 1
                        any_moved = True
                if not any_moved:
                    break

        # === 追従：追従者近傍の中で、相互予期で空いたサイトへ動く ===
        flag = self.w if self.follow_chain else self.w_ma
        S = self.S
        while True:
            any_moved = False
            for k in self.rng.permutation(np.flatnonzero(~moved)):
                x0, y0 = self.position[k]
                nbx = np.mod(x0 + self._dx_Nf, S)
                nby = np.mod(y0 + self._dy_Nf, S)
                ok = (flag[nbx, nby] == 1) & (occupancy[nbx, nby] == -1)
                fx, fy = nbx[ok], nby[ok]
                if fx.size > 0:
                    i = self.rng.integers(fx.size)
                    apply_move(k, fx[i], fy[i])
                    cnt[1] += 1
                    any_moved = True
            if not any_moved or not self.follow_chain:
                break

        # === 自由移動：空いている遷移先へランダムに動く ===
        if self.free_wait:
            while True:
                any_moved = False
                for k in self.rng.permutation(np.flatnonzero(~moved)):
                    kt = targets[k]
                    vac = occupancy[kt[:, 0], kt[:, 1]] == -1
                    if vac.any():
                        nx, ny = kt[self.rng.choice(np.flatnonzero(vac))]
                        apply_move(k, nx, ny)
                        cnt[2] += 1
                        any_moved = True
                if not any_moved:
                    break

        for k in self.rng.permutation(np.flatnonzero(~moved)):
            kt = targets[k]
            vac = occupancy[kt[:, 0], kt[:, 1]] == -1
            if vac.any():
                nx, ny = kt[self.rng.choice(np.flatnonzero(vac))]
                cnt[2] += 1
            else:
                nx, ny = self.position[k]           # 空きがなければその場に留まる
                cnt[3] += 1
            apply_move(k, nx, ny)

        self.position = new_position
        self.occupancy = occupancy
        return tuple(cnt)

    def _best_MA_target(self, kt, z, occupancy):
        """空いている遷移先の中で人気度が最大かつ c 以上のサイト。なければ None"""
        vac = occupancy[kt[:, 0], kt[:, 1]] == -1
        if not vac.any():
            return None
        cc = np.where(vac, z[kt[:, 0], kt[:, 1]], -1)
        m = cc.max()
        if m < self.c:
            return None
        return kt[self.rng.choice(np.flatnonzero(cc == m))]   # 同率ならランダム

    def _literal_MA_phase(self, targets, z, occupancy, apply_move):
        """ma_rule="literal"：全員が1番人気を狙い、負けたら2番目へ。それでもだめなら相互予期では動かない"""
        vacant_at_start = occupancy == -1
        ranked = []
        for k in range(self.N):
            u = np.unique(targets[k], axis=0)
            zz = z[u[:, 0], u[:, 1]]
            ok = vacant_at_start[u[:, 0], u[:, 1]] & (zz >= self.c)
            u, zz = u[ok], zz[ok]
            order = np.lexsort((self.rng.random(len(zz)), -zz))
            ranked.append([tuple(s) for s in u[order]])

        n_moved = 0
        taken = set()
        bidders = list(range(self.N))
        for rank in (0, 1):
            bids = {}
            for k in bidders:
                if len(ranked[k]) > rank and ranked[k][rank] not in taken:
                    bids.setdefault(ranked[k][rank], []).append(k)
            bidders = []
            for site, ks in bids.items():
                winner = ks[self.rng.integers(len(ks))]
                apply_move(winner, site[0], site[1], by_MA=True)
                taken.add(site)
                n_moved += 1
                bidders += [k for k in ks if k != winner]
        return n_moved

    def _velocity_matching(self):
        """速度一致：速度一致近傍（自分を含む）の向きのベクトル平均にそろえる"""
        S = self.S
        new_theta = np.empty(self.N, dtype=float)
        for k in range(self.N):
            x0, y0 = self.position[k]
            occ = self.occupancy[np.mod(x0 + self._dx_M, S), np.mod(y0 + self._dy_M, S)]
            occ = occ[occ != -1]
            if occ.size == 0:
                new_theta[k] = self.theta[k]
                continue
            a = self.theta[occ]
            mv = np.array([np.cos(a).mean(), np.sin(a).mean()])
            new_theta[k] = self.theta[k] if np.allclose(mv, 0.0) else np.arctan2(mv[1], mv[0])
        self.theta = new_theta

    # ==================================================================
    # 論文2（MAM_basic_Murakami と同じ。付録の式(1)〜(13)）
    # ==================================================================

    def _step_paper2(self):
        N, S, c = self.N, self.S, self.c
        self.w_ma[:, :] = 0                          # 付録の χ：wanderer が空けたサイト

        targets, vecs = self._generate_potential_transitions()
        z = self._compute_popularity(targets, dedupe=self.dedupe_votes)
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
          omega_on="velocity"  : q = v_u(k,t;0) + ω·V_u
          omega_on="principal" : q = V_u + ω·v_u(k,t;0)
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
# 評価用の関数（MAM_valid.ipynb の定義と同じもの）
# ======================================================================

def polarization1(displacement, L):
    """論文1の整列度：変位ベクトルの平均の長さ / L"""
    return float(np.linalg.norm(displacement.mean(axis=0)) / L)


def polarization2(displacement):
    """論文2の整列度：動いた個体の単位変位ベクトルの平均の長さ（0〜1）"""
    d = displacement.astype(float)
    n = np.linalg.norm(d, axis=1)
    m = n > 0
    if not m.any():
        return 0.0
    return float(np.linalg.norm((d[m] / n[m, None]).mean(axis=0)))


def density_sliding(position, S, Ls):
    """論文1の密度：Ls×Ls の窓をトーラス上でずらしたときの最大個体数 / N"""
    g = np.zeros((S, S), dtype=int)
    g[position[:, 0], position[:, 1]] = 1
    c = np.pad(np.tile(g, (2, 2)), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    win = c[Ls:Ls + S, Ls:Ls + S] - c[:S, Ls:Ls + S] - c[Ls:Ls + S, :S] + c[:S, :S]
    return float(win.max() / len(position))


def density_paper2(position, S):
    """論文2の密度：周り3×3にいる個体数の平均（自分は含まない）"""
    g = np.zeros((S, S), dtype=int)
    g[position[:, 0], position[:, 1]] = 1
    nb = sum(np.roll(np.roll(g, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1)) - g
    return float(nb[position[:, 0], position[:, 1]].mean())


def simulate(n_steps, Ls=10, **kwargs):
    """
    n_steps 回まわして、整列度・密度・ルールの割合の時系列を返す。
    paper1 は (polarization1, density_sliding(Ls))、paper2 は (polarization2, density_paper2)。
    返り値: pol (n_steps,), dens (n_steps,), rules (n_steps, 4)
            rules の列は 相互予期 / 追従 / 自由移動 / 留まる の割合
    例: 論文2 図3 の赤 → simulate(1500, model="paper2", P=40, alpha=0.2, seed=0)
    """
    m = SoldierCrabSwarm_v1(**kwargs)
    pol, dens, rules = [], [], []
    for _ in range(n_steps):
        m.step()
        if m.model == "paper1":
            pol.append(polarization1(m.last_displacement, m.L))
            dens.append(density_sliding(m.position, m.S, Ls))
        else:
            pol.append(polarization2(m.last_displacement))
            dens.append(density_paper2(m.position, m.S))
        rules.append(m.last_rule_counts)
    return np.array(pol), np.array(dens), np.array(rules) / m.N
