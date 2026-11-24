"""
MAM_paper1.py — 論文1（Gunji et al., Robust swarm of soldier crabs）のモデル。11/24 に MAM_basic_v1.py から分けた。

MAM_basic_v1.py の model="paper1" の処理を、そのまま移したもの。乱数を引く順番も同じなので、
同じ引数・同じ seed なら MAM_basic_v1 と同じ軌跡になる（確かめ方はファイルの最後）。

1ステップの流れ（論文1 p.7〜8、図2）
  可能的遷移を作る → 人気度を数える → 相互予期（全員） → 追従（全員） → 自由移動（全員） → 速度一致（式(8)）

使い方
    from model.MAM_paper1 import CrabSwarmPaper1, PRESETS, simulate
    m = CrabSwarmPaper1(P=15, alpha=0.5, seed=0, **PRESETS["A"])
    m.step()                                   # 1ステップ進める
    hist = m.run(600, record_history=True)     # 位置の履歴 (T+1, N, 2)
    pol, dens, rules = simulate(900, P=15, alpha=0.3, seed=0, **PRESETS["B"])

■ 設定の組（PRESETS）。先生の答えが出るまで、論文1 は1つに決まっていない（検証レポート 0.3 節）
  "text" : 文面どおり（α そのまま・floor・基本ベクトルの向きで速度一致）。引数を何も指定しないときと同じ
  "A"    : 候補A（α 2倍・round）。整列度は α≥0.4 で論文に合う。θ が完全にそろい、動きが固い
  "B"    : 候補B（α そのまま・round・移動ベクトルで速度一致）。α=0.3 の整列度と図13・14 の密度（Ls=10）が合う。
           群れがいくつかに分かれる
  文面と違うのは「α 2倍」と「round」（検証レポート 0.4 節）。図に使うときは、どの設定かを必ず書く。
"""
import numpy as np


PRESETS = {
    "text": dict(),
    "A": dict(alpha_scale=2, int_mode="round"),
    "B": dict(alpha_scale=1, int_mode="round", vm_source="disp"),
}
PRESET_NAMES = {"text": "文面どおり", "A": "候補A（α 2倍・round）", "B": "候補B（round・移動で速度一致）"}


class CrabSwarmPaper1:
    """
    論文1のモデル。

    Parameters
    ----------
    N : int, default 100            個体数
    S : int, default 50             格子の一辺（S×S のトーラス）
    L : float, default 4            基本ベクトルの長さ
    P : int, default 15             可能的遷移の数
    alpha : float, default 0.5      変異角 α（ξ は [-απ, απ] の一様乱数）
    c : int, default 2              人気度の閾値（ζ ≧ c で相互予期が成立）
    R_Nf : int, default 2           追従者近傍の半径
    R_M : int, default 3            速度一致近傍の半径（図15 は 2）
    seed : int or None              乱数シード

    読み方を切り替えるオプション（既定値が文面どおり）
    alpha_scale : float, default 1.0
        ξ の範囲を [-alpha_scale·απ, alpha_scale·απ] にする。1.0 が式(1)の文面どおり。2.0 が候補A の読み方
    int_mode : {"floor", "ceil", "round", "trunc", "stochastic"}, default "floor"
        論文の Int。文面（X ≤ x < X+1）は floor。floor だと遷移が平均 (−0.5, −0.5) ずれる
    vm_source : {"theta", "unit_disp", "disp"}, default "theta"
        速度一致（式(8)）で近傍の何を平均するか。
        "theta" は基本ベクトルの向き（式の記号どおり）、"disp" は直前の移動ベクトル、
        "unit_disp" は直前の移動の単位ベクトル（"disp" と "unit_disp" は言葉の説明「平均の速度の向き」の読み方）
    ma_rule : {"sequential", "literal"}, default "sequential"
        相互予期の競合。"sequential" は空いている次に人気のサイトへ何度でも動ける。
        "literal" は p.8「残りは2番目に人気のサイトへ」を文字どおりに（2番目まで）
    choose_by : {"vector", "site"}, default "vector"
        同率のサイト（式(4)）と自由移動（式(6)）の行き先を、遷移の本数単位で選ぶかサイト単位で選ぶか
    follow_chain : bool, default False
        True にすると、追従した個体が空けたサイトにも追従できる（旧版 MAM_basic_ex の挙動）
    free_wait : bool, default False
        True にすると、自由移動で行き先がふさがっている個体が、前の個体が動いて空くまで待つ
    popularity : {"individual", "transition"}, default "individual"
        人気度（式(3)）の数え方。"individual" はそのサイトを遷移先に含む個体の数（同じ個体は1票、v1 と同じ）。
        "transition" は遷移の本数（p.7「the number of transitions reaching that site」の文面どおり）。
        "transition" だと、1体の遷移が2本重なるだけで人気度 2 になる（11/24 追加）

    Attributes
    ----------
    position : (N, 2) int           現在の座標
    theta : (N,) float              基本ベクトルの向き
    last_displacement : (N, 2) int  直前のステップの変位（トーラスをまたいだ分は補正済み）
    last_rule_counts : tuple        直前のステップで 相互予期 / 追従 / 自由移動 / 留まる になった個体数
    move_counts : dict              これまでの累計（キーは "MA", "follow", "free", "stay"）
    """

    INT_MODES = ("floor", "ceil", "round", "trunc", "stochastic")

    def __init__(self, N=100, S=50, L=4.0, P=15, alpha=0.5, c=2, R_Nf=2, R_M=3, seed=None,
                 alpha_scale=1.0, int_mode="floor", vm_source="theta", ma_rule="sequential",
                 choose_by="vector", follow_chain=False, free_wait=False, popularity="individual"):
        if int_mode not in self.INT_MODES:
            raise ValueError(f"int_mode は {self.INT_MODES} のどれか（指定値: {int_mode!r}）")
        if ma_rule not in ("sequential", "literal"):
            raise ValueError(f'ma_rule は "sequential" か "literal"（指定値: {ma_rule!r}）')
        if choose_by not in ("vector", "site"):
            raise ValueError(f'choose_by は "vector" か "site"（指定値: {choose_by!r}）')
        if vm_source not in ("theta", "unit_disp", "disp"):
            raise ValueError(f'vm_source は "theta" / "unit_disp" / "disp" のどれか（指定値: {vm_source!r}）')
        if popularity not in ("individual", "transition"):
            raise ValueError(f'popularity は "individual" か "transition"（指定値: {popularity!r}）')
        if N > S * S:
            raise ValueError(f"N({N}) が格子点の総数 S*S({S*S}) を超えています。")

        self.rng = np.random.default_rng(seed)
        self.N, self.S, self.L, self.P, self.alpha, self.c = N, S, L, P, alpha, c
        self.R_Nf, self.R_M, self.seed = R_Nf, R_M, seed
        self.alpha_scale, self.int_mode, self.vm_source = alpha_scale, int_mode, vm_source
        self.ma_rule, self.choose_by = ma_rule, choose_by
        self.follow_chain, self.free_wait = follow_chain, free_wait
        self.popularity = popularity
        self.t = 0

        # ---- 初期配置：重なりのないランダムな格子点、向きは一様乱数 ----
        all_sites = self.rng.permutation(S * S)[:N]
        self.position = np.stack([all_sites % S, all_sites // S], axis=1).astype(int)
        self.theta = self.rng.uniform(0.0, 2 * np.pi, size=N)

        # occupancy[x, y] = その格子にいる個体の番号、空きなら -1
        self.occupancy = -np.ones((S, S), dtype=int)
        self.occupancy[self.position[:, 0], self.position[:, 1]] = np.arange(N)

        # 空いたサイトの印。w：誰かが動いて空いた、w_ma：相互予期で空いた
        self.w = np.zeros((S, S), dtype=int)
        self.w_ma = np.zeros((S, S), dtype=int)

        self.last_displacement = np.zeros((N, 2), dtype=int)
        self.last_rule_counts = (0, 0, 0, 0)
        self.move_counts = {"MA": 0, "follow": 0, "free": 0, "stay": 0}

        self._dx_Nf, self._dy_Nf = self._square_neighbor_offsets(R_Nf)
        self._dx_M, self._dy_M = self._square_neighbor_offsets(R_M)

    # ------------------------------------------------------------------
    # 準備
    # ------------------------------------------------------------------

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
        可能的遷移 v(k,t;i) = Int(L·η·(cos(θ+ξ), sin(θ+ξ)))、η∈[0,1]、ξ∈[-απ, απ]（式(1)）
        i=0 は基本ベクトル（η=1, ξ=0）。返り値は 遷移先 (N,P,2)
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
        return np.stack([tx, ty], axis=-1)

    def _compute_popularity(self, targets):
        """人気度 ζ(x,y;t)：そのサイトを遷移先に含む個体の数（同じ個体が何本選んでいても1票）"""
        N, P, _ = targets.shape
        flat_x = targets[:, :, 0].ravel()
        flat_y = targets[:, :, 1].ravel()
        if self.popularity == "transition":         # 遷移の本数で数える（p.7 の文面どおり）
            z = np.zeros((self.S, self.S), dtype=int)
            np.add.at(z, (flat_x, flat_y), 1)
            return z
        triples =np.unique(np.stack([np.repeat(np.arange(N), P), flat_x, flat_y], axis=1), axis=0)
        z = np.zeros((self.S, self.S), dtype=int)
        np.add.at(z, (triples[:, 1], triples[:, 2]), 1)
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

        self._velocity_matching()
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
        N = self.N
        self.w[:, :] = 0
        self.w_ma[:, :] = 0

        targets = self._generate_potential_transitions()
        z = self._compute_popularity(targets)

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

        # === 相互予期（式(2)〜(4)） ===
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

        # === 追従（式(5)）：追従者近傍の中で、相互予期で空いたサイトへ動く ===
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

        # === 自由移動（式(6)）：空いている遷移先へランダムに動く ===
        if self.free_wait:
            while True:
                any_moved = False
                for k in self.rng.permutation(np.flatnonzero(~moved)):
                    kt = targets[k]
                    vac = occupancy[kt[:, 0], kt[:, 1]] == -1
                    if vac.any():
                        nx, ny = self._pick(kt, vac)
                        apply_move(k, nx, ny)
                        cnt[2] += 1
                        any_moved = True
                if not any_moved:
                    break

        for k in self.rng.permutation(np.flatnonzero(~moved)):
            kt = targets[k]
            vac = occupancy[kt[:, 0], kt[:, 1]] == -1
            if vac.any():
                nx, ny = self._pick(kt, vac)
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
        return self._pick(kt, cc == m)                         # 同率ならランダム

    def _pick(self, kt, mask):
        """遷移先 kt のうち mask が True のものから1つ選ぶ。
        choose_by="vector" は遷移（本数）単位、"site" はサイト単位で一様に選ぶ"""
        idx = np.flatnonzero(mask)
        if self.choose_by == "vector":
            return kt[self.rng.choice(idx)]
        sites = np.unique(kt[idx], axis=0)
        return sites[self.rng.integers(len(sites))]

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
        """速度一致（式(8)）：速度一致近傍（自分を含む）の向きのベクトル平均にそろえる"""
        S = self.S
        new_theta = np.empty(self.N, dtype=float)
        if self.vm_source == "theta":
            vec = np.stack([np.cos(self.theta), np.sin(self.theta)], axis=1)
        else:
            d = self.last_displacement.astype(float)
            if self.vm_source == "unit_disp":
                n = np.linalg.norm(d, axis=1)
                vec = np.zeros_like(d)
                vec[n > 0] = d[n > 0] / n[n > 0, None]
            else:
                vec = d
        for k in range(self.N):
            x0, y0 = self.position[k]
            occ = self.occupancy[np.mod(x0 + self._dx_M, S), np.mod(y0 + self._dy_M, S)]
            occ = occ[occ != -1]
            if occ.size == 0:
                new_theta[k] = self.theta[k]
                continue
            if self.vm_source == "theta":
                a = self.theta[occ]
                mv = np.array([np.cos(a).mean(), np.sin(a).mean()])
            else:
                mv = vec[occ].mean(axis=0)
            new_theta[k] = self.theta[k] if np.allclose(mv, 0.0) else np.arctan2(mv[1], mv[0])
        self.theta = new_theta


# ======================================================================
# 評価用の関数
# ======================================================================

def torus_diff(d, S):
    """トーラスをまたいだ差を (-S/2, S/2] に直す"""
    return (d + S / 2) % S - S / 2


def polarization1(displacement, L):
    """論文1の整列度（p.17）：変位ベクトルの平均の長さ / L"""
    return float(np.linalg.norm(displacement.mean(axis=0)) / L)


def density_grid(position, S, Ls):
    """論文1の密度（p.17、文面どおり）：空間を Ls×Ls の区画に分け、区画の中の最大個体数 / N。
    図11 は Ls=25、図13・14 は Ls≈6 か 10（検証レポート 2.9 節）"""
    n = S // Ls
    h = np.zeros((n, n), int)
    np.add.at(h, (np.minimum(position[:, 0] // Ls, n - 1), np.minimum(position[:, 1] // Ls, n - 1)), 1)
    return float(h.max() / len(position))


def density_sliding(position, S, Ls):
    """前に使っていた密度：Ls×Ls の窓をトーラス上でずらしたときの最大個体数 / N"""
    g = np.zeros((S, S), dtype=int)
    g[position[:, 0], position[:, 1]] = 1
    c = np.pad(np.tile(g, (2, 2)), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    win = c[Ls:Ls + S, Ls:Ls + S] - c[:S, Ls:Ls + S] - c[Ls:Ls + S, :S] + c[:S, :S]
    return float(win.max() / len(position))


def simulate(n_steps, Ls=10, **kwargs):
    """
    n_steps 回まわして、整列度・密度（区画 Ls）・ルールの割合の時系列を返す。
    kwargs は CrabSwarmPaper1 の引数（PRESETS の中身もここに渡す）。
    返り値: pol (n_steps,), dens (n_steps,), rules (n_steps, 4)
            rules の列は 相互予期 / 追従 / 自由移動 / 留まる の割合
    """
    m = CrabSwarmPaper1(**kwargs)
    pol, dens, rules = [], [], []
    for _ in range(n_steps):
        m.step()
        pol.append(polarization1(m.last_displacement, m.L))
        dens.append(density_grid(m.position, m.S, Ls))
        rules.append(m.last_rule_counts)
    return np.array(pol), np.array(dens), np.array(rules) / m.N


def correlation_function(position, velocity, S, bin_width=1.0, min_pairs=10):
    """
    スケールフリー相関（論文1 p.22）。u_i = v_i − (1/N)Σv_k、C(r) = Σ u_i·u_j δ(r − r_ij) / Σ δ(r − r_ij)。
    C は自分との積の平均 <u_i·u_i> で割る（Cavagna et al. 2010 と同じ。論文1 には規格化が書かれていない）。
    返り値: (r, C, xi)。xi は C が最初に正から負に変わる距離（線形補間）。求まらなければ nan。揺らぎが 0 なら None
    velocity に何を渡すかで結果が大きく変わる（検証レポート 2.13 節）
      移動：torus_diff(pos[t+1] - pos[t], S)、基本ベクトル：L·(cosθ, sinθ)
    """
    N = len(position)
    u = velocity - velocity.mean(0)
    c0 = np.mean((u ** 2).sum(1))
    if c0 == 0:
        return None
    dist = np.sqrt((torus_diff(position[:, None, :] - position[None, :, :], S) ** 2).sum(-1))
    iu = np.triu_indices(N, 1)
    r, c = dist[iu], (u @ u.T)[iu]
    edges = np.arange(0, S / 2 + bin_width, bin_width)
    idx = np.digitize(r, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    sums = np.bincount(idx[ok], weights=c[ok], minlength=len(edges) - 1)
    cnts = np.bincount(idx[ok], minlength=len(edges) - 1)
    good = cnts >= min_pairs
    rr = (0.5 * (edges[:-1] + edges[1:]))[good]
    C = sums[good] / cnts[good] / c0
    xi = np.nan
    if len(C) >= 2 and C[0] > 0:
        for i in range(len(C) - 1):
            if C[i] > 0 and C[i + 1] <= 0:
                xi = rr[i] + (rr[i + 1] - rr[i]) * C[i] / (C[i] - C[i + 1])
                break
    return rr, C, xi


if __name__ == "__main__":
    # MAM_basic_v1 と同じ軌跡になるかを確かめる（研究/ で  python -m model.MAM_paper1）
    from model.MAM_basic_v1 import SoldierCrabSwarm_v1
    for name, kw in PRESETS.items():
        a = CrabSwarmPaper1(P=15, alpha=0.4, seed=3, **kw)
        b = SoldierCrabSwarm_v1(model="paper1", P=15, alpha=0.4, seed=3, **kw)
        a.run(200); b.run(200)
        same = np.array_equal(a.position, b.position) and np.allclose(a.theta, b.theta)
        print(f"{name}: MAM_basic_v1 と同じ = {same}")
