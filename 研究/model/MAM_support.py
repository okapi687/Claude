import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from IPython.display import Image, display, HTML

from model.MAM_basic_ex import SoldierCrabSwarm_ex
from model.MAM_noVM import SoldierCrabSwarm_noVM

class SwarmVisualizer:
    """
    SoldierCrabSwarm のシミュレーションを実行したり、スナップショットをとったりする補助クラス

    使い方
    >>> swarm = SoldierCrabSwarm()      パラメータを変更する場合は括弧内に代入する
    >>> viz = SwarmVisualizer(swarm)    
    >>> viz.summary()                   パラメータ設定を表示する
    >>> viz.animate()                   n_stepで実行step数、pathでファイル名、show=Falseで表示しないなど設定できる

    >>> viz.save_snapshot(frame=)       指定したframeのスナップショットを保存
    """

    def __init__(self, swarm, arrow_len=0.8, figsize=(6, 6)):
        self.swarm = swarm
        self.arrow_len = arrow_len
        self.figsize = figsize

        # run() で記録される履歴
        self.positions_history = None
        self.theta_history = None

        # build_animation() で作られる描画関連オブジェクト
        self.fig = None
        self.ax = None
        self.quiv = None
        self.step_text = None
        self.ani = None

    # ------------------------------------------------------------------
    # パラメータ表示
    # ------------------------------------------------------------------
    def summary(self):
        """
        現在のモデル状態の要約を表示する
        """
        s = self.swarm
        print("=== パラメータ設定 ===")
        print(f"格子空間サイズ S       : {s.S} x {s.S} (トーラス境界)")
        print(f"エージェント数 N       : {s.N}")
        print(f"基本ベクトル長 L       : {s.L}")
        print(f"可能的遷移数 P         : {s.P}")
        print(f"変異角 alpha           : {s.alpha}")
        print(f"追従者近傍半径 R(Nf)   : {s.R_Nf}")
        print(f"速度一致近傍半径 R(M)  : {s.R_M}")
        print(f"人気度閾値 c           : {s.c}")
        print(f"シード値 seed          : {s.seed}")
        # print(f"壁監視近傍半径 N_W     : {s.N_W}")
        
        # print(f"現在ステップ t         : {s.t}")

    # ------------------------------------------------------------------
    # ワンストップ実行：シミュレーション実行 → 描画 → 保存（→ 表示）
    # ------------------------------------------------------------------
    def animate(self, n_steps=300, path="soldier_crab_swarm.gif", fps=10,
                interval=100, show=True):
        """
        アニメーションを実行して保存する
        fps GIFのフレームレート
        interval アニメーション再生時のフレーム間隔（ミリ秒）。GIF自体の速度は fps で決まる。
        """
        self.run(n_steps)
        self.build_animation(interval=interval)
        self.ani.save(path, writer=PillowWriter(fps=fps))

        if show:
            html = HTML(self.ani.to_jshtml())
            display(html)

        plt.close(self.fig)
        return path

        # self.run(n_steps)
        # if show:
        #     return self.show(path=path, fps=fps, interval=interval)
        # return self.save(path=path, fps=fps, interval=interval)
    
    def save_snapshot(self, frame, path=None, dpi=150, show=True):
        """
        frameステップ分シミュレーションを実行し、その時点の状態を保存・表示する。
        """
        self.run(frame)

        if path is None:
            path = f"snapshot_step{frame:04d}.png"

        fig, ax = plt.subplots(figsize=self.figsize)
        self._draw_frame(len(self.positions_history) - 1, ax)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(fig)

        return path
    
    # ------------------------------------------------------------------
    # 内部処理に使用した関数たち
    # ------------------------------------------------------------------
    def run(self, n_steps):
        """
        n_steps 分シミュレーションを実行しながら、各ステップの position と theta を記録する。
        swarm.run() は position しか記録できないため、矢印描画に使うthetaもここで一緒に保存している。
        """
        positions_history = [self.swarm.position.copy()]
        theta_history = [self.swarm.theta.copy()]

        for _ in range(n_steps):
            self.swarm.step()
            positions_history.append(self.swarm.position.copy())
            theta_history.append(self.swarm.theta.copy())

        self.positions_history = np.array(positions_history)  # (n_steps+1, N, 2)
        self.theta_history = np.array(theta_history)           # (n_steps+1, N)
        return self
    
    # ------------------------------------------------------------------
    # 描画関連
    # ------------------------------------------------------------------
    def _build_figure(self):
        S = self.swarm.S
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(-0.5, S - 0.5)
        ax.set_ylim(-0.5, S - 0.5)
        ax.set_aspect("equal")
        ax.set_title("Soldier Crab Swarm")

        x0 = self.positions_history[0][:, 0]
        y0 = self.positions_history[0][:, 1]
        th0 = self.theta_history[0]
        u0 = self.arrow_len * np.cos(th0)
        v0 = self.arrow_len * np.sin(th0)

        quiv = ax.quiver(
            x0, y0, u0, v0,
            angles="xy", scale_units="xy", scale=1,
            width=0.004, color="tab:blue",
        )
        step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

        self.fig, self.ax, self.quiv, self.step_text = fig, ax, quiv, step_text

    def _update(self, frame):
        x = self.positions_history[frame][:, 0]
        y = self.positions_history[frame][:, 1]
        th = self.theta_history[frame]
        u = self.arrow_len * np.cos(th)
        v = self.arrow_len * np.sin(th)

        self.quiv.set_offsets(np.column_stack([x, y]))
        self.quiv.set_UVC(u, v)
        self.step_text.set_text(f"step = {frame}")
        return self.quiv, self.step_text

    # ------------------------------------------------------------------
    # 保存・表示関連
    # ------------------------------------------------------------------
    def build_animation(self, interval=100):
        """FuncAnimation オブジェクトを作成して返す（run() 実行後に呼ぶ）"""
        if self.positions_history is None:
            raise RuntimeError(
                "履歴がありません。先に animator.run(n_steps) を呼んでください。"
            )

        self._build_figure()
        self.ani = animation.FuncAnimation(
            self.fig,
            self._update,
            frames=len(self.positions_history),
            interval=interval,
            blit=False,
        )
        return self.ani

    # def save(self, path="soldier_crab_swarm.gif", fps=10, interval=100):
    #     """アニメーションをGIFとして保存する。パスを返す。"""
    #     if self.ani is None:
    #         self.build_animation(interval=interval)

    #     self.ani.save(path, writer=PillowWriter(fps=fps))
    #     plt.close(self.fig)
    #     return path

    # def show(self, path="soldier_crab_swarm.gif", fps=10, interval=100):
    #     """GIFとして保存しつつ、Notebook内にも同じものを表示する。"""
    #     gif_path = self.save(path=path, fps=fps, interval=interval)
    #     display(Image(filename=gif_path))
    #     return gif_path

    def _draw_frame(self, frame, ax):
        """指定したステップの状態を ax に描画する（矢印のみ、内部利用）"""
        S = self.swarm.S
        ax.set_xlim(-0.5, S - 0.5)
        ax.set_ylim(-0.5, S - 0.5)
        ax.set_aspect("equal")
        ax.set_title("Soldier Crab Swarm")
 
        x = self.positions_history[frame][:, 0]
        y = self.positions_history[frame][:, 1]
        th = self.theta_history[frame]
        u = self.arrow_len * np.cos(th)
        v = self.arrow_len * np.sin(th)
 
        ax.quiver(
            x, y, u, v,
            angles="xy", scale_units="xy", scale=1,
            width=0.004, color="tab:blue",
        )
        ax.text(0.02, 0.98, f"step = {frame}", transform=ax.transAxes, va="top")

    