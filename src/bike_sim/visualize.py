"""Side-view visualizer: rider on the bike, wind, and the metrics that drive the policy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Ellipse, FancyArrow, FancyBboxPatch, Polygon, Rectangle

from . import rider
from .aero import aero
from .baselines import BASELINES
from .config import DEFAULT, Params
from .dynamics import forces
from .env import BikeWindEnv
from .evaluate import RESULTS, rollout

BG = "#f4f2ec"
PANEL = "#ffffff"
ROAD = "#d8d2c8"
INK = "#1f2430"
MUTED = "#6a6f78"
STEEL = "#3d5a73"
SAGE = "#4a6b58"
RUST = "#8c4a38"
SAND = "#b0893a"
BIKE = "#2c3340"
RULE = "#d9d4cc"

VIEW_LEFT = -0.45
VIEW_RIGHT = 6.05
VIEW_BOTTOM = -0.32
VIEW_TOP = 2.18
WHEEL_R = 0.335


@dataclass
class FrameState:
    t: float
    x: float
    v: float
    posture: float
    power: float
    w_bal: float
    headwind: float
    policy: str
    course_length: float = 4000.0
    w_prime_max: float = DEFAULT.w_prime_max
    v_ref: float = 8.0
    dt_control: float = 2.0
    finished: bool = False

    @property
    def episode_return(self) -> float:
        return (self.x - self.v_ref * self.t) / (self.v_ref * self.dt_control)

    @property
    def step_reward(self) -> float:
        return (self.v - self.v_ref) / self.v_ref


def _mix(a: str, b: str, t: float) -> str:
    t = float(np.clip(t, 0.0, 1.0))
    ca = np.array([int(a[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    cb = np.array([int(b[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    c = (ca * (1.0 - t) + cb * t).astype(int)
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


def _lerp(p, q, u):
    return (p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u)


def rider_points(posture: float, crank: float) -> dict[str, tuple[float, float]]:
    """Side-view keypoints in metres. Origin is the rear tyre contact patch."""
    u = float(np.clip(posture, 0.0, 1.0))
    rear = (0.0, WHEEL_R)
    front = (1.10, WHEEL_R)
    bb = (0.42, 0.27)
    saddle = _lerp((0.30, 0.98), (0.50, 0.88), u)
    head_tube = (0.90, 0.78)
    hands = _lerp((1.02, 0.86), (0.97, 0.64), u)

    hip = _lerp(saddle, (saddle[0] + 0.04, saddle[1] - 0.04), 0.4)
    torso_ang = np.deg2rad(78.0 - 64.0 * u)
    shoulder = (hip[0] + 0.48 * np.cos(torso_ang), hip[1] + 0.48 * np.sin(torso_ang))
    head = (
        shoulder[0] + 0.20 * np.cos(torso_ang - 0.15),
        shoulder[1] + 0.20 * np.sin(torso_ang - 0.15),
    )
    elbow = (
        0.55 * shoulder[0] + 0.45 * hands[0],
        0.55 * shoulder[1] + 0.45 * hands[1] - 0.10,
    )

    crank_len = 0.165
    pedal_n = (bb[0] + crank_len * np.cos(crank), bb[1] + crank_len * np.sin(crank))
    pedal_f = (bb[0] + crank_len * np.cos(crank + np.pi), bb[1] + crank_len * np.sin(crank + np.pi))
    knee_n = ((hip[0] + pedal_n[0]) / 2.0 + 0.08, (hip[1] + pedal_n[1]) / 2.0 + 0.06)
    knee_f = ((hip[0] + pedal_f[0]) / 2.0 + 0.08, (hip[1] + pedal_f[1]) / 2.0 + 0.06)
    return {
        "rear": rear,
        "front": front,
        "bb": bb,
        "saddle": saddle,
        "head_tube": head_tube,
        "hands": hands,
        "hip": hip,
        "shoulder": shoulder,
        "head": head,
        "elbow": elbow,
        "pedal_n": pedal_n,
        "pedal_f": pedal_f,
        "knee_n": knee_n,
        "knee_f": knee_f,
    }


def _metrics(s: FrameState, p: Params = DEFAULT):
    cda = rider.cda(s.posture, p)
    a = aero(s.v, -s.headwind, 0.0, cda, p)
    f_prop, f_aero, f_roll, f_grade = forces(s.v, s.power, s.posture, (-s.headwind, 0.0), 0.0, p)
    return {
        "cda": cda,
        "aero": a,
        "f_prop": f_prop,
        "f_aero": f_aero,
        "f_roll": f_roll,
        "f_grade": f_grade,
        "cp": rider.critical_power(s.posture, p),
        "p_cap": rider.p_cap(s.posture, p),
    }


def _line(ax, pts, color, lw, z=3, alpha=1.0):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=z, alpha=alpha)


def _draw_bike(ax, s: FrameState) -> None:
    crank = (s.x / WHEEL_R) % (2.0 * np.pi)
    g = rider_points(s.posture, crank)
    ox = 1.25
    p = {k: (v[0] + ox, v[1]) for k, v in g.items()}
    power_t = float(np.clip((s.power - 180.0) / 170.0, 0.0, 1.0))
    jersey = _mix("#4a6b58", "#8c4a38", power_t)
    shorts = _mix("#3d5348", "#6b4538", power_t)

    axle_n = (p["rear"][0], WHEEL_R)
    axle_f = (p["front"][0], WHEEL_R)
    ax.add_patch(Ellipse(((axle_n[0] + axle_f[0]) / 2.0, -0.02), 1.55, 0.10, color="#8a847c", alpha=0.22, zorder=1))
    for axle in (axle_n, axle_f):
        ax.add_patch(Circle(axle, WHEEL_R, fc="#f7f6f2", ec="#2c3340", lw=3.6, zorder=4))
        ax.add_patch(Circle(axle, WHEEL_R - 0.045, fill=False, ec="#9aa0a8", lw=1.0, zorder=4))
        ax.add_patch(Circle(axle, 0.055, color="#2c3340", zorder=5))
        for k in range(8):
            ang = crank + k * np.pi / 4.0
            ax.plot(
                [axle[0], axle[0] + (WHEEL_R - 0.04) * np.cos(ang)],
                [axle[1], axle[1] + (WHEEL_R - 0.04) * np.sin(ang)],
                color="#8b919a",
                lw=0.8,
                zorder=4,
            )

    _line(ax, [p["rear"], p["bb"], p["head_tube"], p["saddle"], p["rear"]], BIKE, 3.4, z=5)
    _line(ax, [p["bb"], p["saddle"]], BIKE, 3.2, z=5)
    _line(ax, [p["head_tube"], p["front"]], BIKE, 3.2, z=5)
    _line(ax, [p["head_tube"], p["hands"]], BIKE, 2.6, z=5)
    hx, hy = p["hands"]
    ax.plot([hx - 0.06, hx + 0.07], [hy, hy], color=BIKE, lw=4.0, zorder=6, solid_capstyle="round")
    sx, sy = p["saddle"]
    ax.plot([sx - 0.09, sx + 0.10], [sy + 0.01, sy + 0.01], color="#1f2430", lw=6.0, zorder=6, solid_capstyle="round")

    _line(ax, [p["hip"], p["knee_f"], p["pedal_f"]], "#7a828c", 4.2, z=3, alpha=0.8)
    _line(ax, [p["bb"], p["pedal_f"]], "#6a717a", 2.4, z=3)
    _line(ax, [p["hip"], p["knee_n"], p["pedal_n"]], shorts, 5.2, z=6)
    _line(ax, [p["bb"], p["pedal_n"]], SAND, 2.6, z=6)
    ax.add_patch(Circle(p["pedal_n"], 0.035, color=SAND, zorder=7))
    ax.add_patch(Circle(p["bb"], 0.045, color="#2c3340", zorder=6))

    hip, sh = p["hip"], p["shoulder"]
    tx, ty = sh[0] - hip[0], sh[1] - hip[1]
    nrm = max(np.hypot(tx, ty), 1e-6)
    nx, ny = -ty / nrm * 0.11, tx / nrm * 0.11
    ax.add_patch(
        Polygon(
            [
                (hip[0] - nx, hip[1] - ny),
                (hip[0] + nx, hip[1] + ny),
                (sh[0] + nx, sh[1] + ny),
                (sh[0] - nx, sh[1] - ny),
            ],
            closed=True,
            color=jersey,
            zorder=7,
        )
    )
    _line(ax, [p["shoulder"], p["elbow"], p["hands"]], jersey, 4.2, z=8)
    ax.add_patch(Circle(p["hip"], 0.06, color=shorts, zorder=8))
    ax.add_patch(Circle(p["head"], 0.115, color="#f3d2b3", zorder=8))
    helm_ang = np.degrees(np.arctan2(sh[1] - hip[1], sh[0] - hip[0]))
    ax.add_patch(
        Ellipse(
            (p["head"][0] + 0.02, p["head"][1] + 0.03),
            0.28,
            0.16,
            angle=helm_ang,
            color=jersey,
            zorder=9,
        )
    )

    m = _metrics(s)
    scale = 0.018
    base_x, base_y = axle_f[0] + 0.55, 0.55
    fp = max(m["f_prop"] * scale, 0.12)
    fd = min(m["f_aero"] * scale, -0.12)
    ax.add_patch(FancyArrow(base_x, base_y, fp, 0.0, width=0.05, color=SAGE, length_includes_head=True, zorder=10))
    ax.add_patch(FancyArrow(base_x, base_y + 0.28, fd, 0.0, width=0.05, color=RUST, length_includes_head=True, zorder=10))
    ax.text(base_x + 0.06, base_y + 0.10, f"F_prop  {m['f_prop']:.0f} N", color=SAGE, fontsize=8.5)
    ax.text(base_x + 0.06, base_y + 0.38, f"F_drag  {-m['f_aero']:.0f} N", color=RUST, fontsize=8.5)


def _draw_scene(ax, s: FrameState) -> None:
    ax.clear()
    gust = float(np.clip(s.headwind / 7.0, 0.0, 1.0))
    ax.set_facecolor(_mix("#ffffff", "#f3ebe3", gust))
    ax.set_xlim(VIEW_LEFT, VIEW_RIGHT)
    ax.set_ylim(VIEW_BOTTOM, VIEW_TOP)
    ax.axis("off")

    ax.add_patch(Rectangle((VIEW_LEFT, VIEW_BOTTOM), VIEW_RIGHT - VIEW_LEFT, -VIEW_BOTTOM, color=ROAD, zorder=0))
    ax.plot([VIEW_LEFT, VIEW_RIGHT], [0.0, 0.0], color="#b7b1a8", lw=1.6, zorder=1)
    phase = s.x % 2.6
    for mark in np.arange(VIEW_LEFT - 2.6, VIEW_RIGHT + 2.6, 2.6):
        x0 = mark - phase
        ax.plot([x0, x0 + 1.05], [-0.09, -0.09], color="#c4b18a", lw=5.0, solid_capstyle="butt", zorder=1)

    rng = np.random.default_rng(int(s.x * 4.0) * 17 + 3)
    n_streaks = 7 + int(8 * gust)
    v_rel = s.v + s.headwind
    for _ in range(n_streaks):
        y = 1.35 + 0.75 * rng.random()
        x = VIEW_LEFT + (VIEW_RIGHT - VIEW_LEFT) * rng.random()
        length = 0.45 + 0.10 * v_rel
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(x + length, y),
            arrowprops={"arrowstyle": "->", "color": _mix("#7b8794", "#a56b4a", gust), "lw": 1.15 + 0.6 * gust},
            zorder=2,
        )

    _draw_bike(ax, s)
    label = "headwind patch" if s.headwind > 4.0 else "open road"
    ax.text(
        0.015,
        0.93,
        label,
        transform=ax.transAxes,
        color=RUST if s.headwind > 4.0 else INK,
        fontsize=12,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )


def _draw_course(ax, s: FrameState, xs: np.ndarray, headwinds: np.ndarray) -> None:
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.fill_between(xs, 0.0, headwinds, color="#c9b29a", alpha=0.95)
    ax.plot(xs, headwinds, color="#8c6a4a", lw=1.3)
    ax.axvline(s.x, color=STEEL, lw=1.8)
    ax.set_xlim(0.0, s.course_length)
    ax.set_ylim(0.0, max(8.0, float(headwinds.max()) + 0.5))
    ax.set_yticks([1, 7])
    ax.set_yticklabels(["1", "7 m/s"], color=MUTED, fontsize=8)
    ax.set_xticks([0, 2000, 3000, 4000])
    ax.set_xticklabels(["0", "2 km", "patch", "4 km"], color=MUTED, fontsize=8)
    for spine in ax.spines.values():
        spine.set_color(RULE)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_ylabel("wind", color=MUTED, fontsize=8)
    ax.set_title("course", color=MUTED, loc="left", fontsize=9, pad=2)


def _bar(ax, y, frac, color, label, value) -> None:
    ax.add_patch(FancyBboxPatch((0.08, y), 0.84, 0.05, boxstyle="round,pad=0.004,rounding_size=0.012", facecolor="#f1efe9", ec=RULE))
    ax.add_patch(Rectangle((0.08, y), 0.84 * float(np.clip(frac, 0.0, 1.0)), 0.05, color=color, zorder=2))
    ax.text(0.08, y + 0.056, label, color=MUTED, fontsize=8)
    ax.text(0.92, y + 0.056, value, color=INK, fontsize=8, ha="right")


def _draw_hud(ax, s: FrameState) -> None:
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    m = _metrics(s)
    a = m["aero"]
    font = {"fontfamily": "DejaVu Sans"}

    ret = s.episode_return
    step = s.step_reward
    ret_color = SAGE if ret >= 0.0 else RUST
    ax.text(0.08, 0.94, s.policy.replace("_", " "), color=STEEL, fontsize=14, fontweight="bold", **font)
    ax.text(0.08, 0.905, "live ride metrics", color=MUTED, fontsize=8, **font)
    ax.text(0.08, 0.835, f"{s.v * 3.6:.1f}", color=INK, fontsize=24, fontweight="bold", **font)
    ax.text(0.08, 0.795, "km/h speed", color=MUTED, fontsize=8, **font)
    ax.text(0.55, 0.835, f"{s.power:.0f}", color=INK, fontsize=22, fontweight="bold", **font)
    ax.text(0.55, 0.795, "W power", color=MUTED, fontsize=8, **font)
    ax.text(0.08, 0.725, f"{ret:+.2f}", color=ret_color, fontsize=20, fontweight="bold", **font)
    ax.text(0.08, 0.688, "episode return", color=MUTED, fontsize=8, **font)
    ax.text(0.62, 0.725, f"{step:+.2f}", color=SAGE if step >= 0.0 else RUST, fontsize=16, fontweight="bold", **font)
    ax.text(0.62, 0.688, "step reward", color=MUTED, fontsize=8, **font)

    rows = [
        ("drag force", f"{-a.drag_x:.0f} N"),
        ("CdA", f"{m['cda']:.3f} m^2"),
        ("Reynolds", f"{a.reynolds:.2e}"),
        ("dynamic q", f"{a.q:.0f} Pa"),
        ("headwind", f"{s.headwind:.1f} m/s"),
        ("time", f"{int(s.t) // 60}:{int(s.t) % 60:02d}"),
        ("distance", f"{s.x:.0f} m"),
    ]
    y = 0.625
    for name, val in rows:
        ax.text(0.08, y, name, color=MUTED, fontsize=8, **font)
        ax.text(0.92, y, val, color=INK, fontsize=10, ha="right", **font)
        y -= 0.048

    _bar(ax, 0.175, s.w_bal / s.w_prime_max, SAGE, "W' anaerobic reserve", f"{s.w_bal / 1000:.1f} / {s.w_prime_max / 1000:.0f} kJ")
    _bar(ax, 0.08, s.posture, SAND, "posture   upright  ->  tuck", f"u_a  {s.posture:.2f}")
    ax.text(0.08, 0.022, f"CP {m['cp']:.0f} W     peak {m['p_cap']:.0f} W", color=MUTED, fontsize=8, **font)


class Visualizer:
    def __init__(self, xs: np.ndarray, headwinds: np.ndarray):
        self.xs = xs
        self.headwinds = headwinds
        self.fig = plt.figure(figsize=(13.6, 6.5), facecolor=BG)
        plt.rcParams["font.family"] = "DejaVu Sans"
        gs = GridSpec(2, 2, figure=self.fig, width_ratios=[2.7, 1.25], height_ratios=[4.0, 1.05], hspace=0.18, wspace=0.12, left=0.03, right=0.985, top=0.97, bottom=0.07)
        self.ax_scene = self.fig.add_subplot(gs[0, 0])
        self.ax_course = self.fig.add_subplot(gs[1, 0])
        self.ax_hud = self.fig.add_subplot(gs[:, 1])
        for ax in (self.ax_scene, self.ax_course, self.ax_hud):
            ax.set_facecolor(PANEL)
            for spine in ax.spines.values():
                spine.set_color(RULE)

    def update(self, s: FrameState) -> None:
        _draw_scene(self.ax_scene, s)
        _draw_course(self.ax_course, s, self.xs, self.headwinds)
        _draw_hud(self.ax_hud, s)
        self.fig.canvas.draw()

    def rgb(self) -> np.ndarray:
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()


def course_profile(env: BikeWindEnv, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, env.course_length, n)
    hw = np.array([-env.wind_field(x)[0] for x in xs])
    return xs, hw


def load_actor(name: str, model_path: Path):
    if name == "sac":
        if not model_path.exists():
            raise FileNotFoundError(f"no SAC model at {model_path}")
        from stable_baselines3 import SAC

        model = SAC.load(model_path, device="cpu")
        return lambda e, o: model.predict(o, deterministic=True)[0]
    if name not in BASELINES:
        raise ValueError(f"unknown policy {name}; choose from {['sac', *BASELINES]}")
    fn = BASELINES[name]
    return lambda e, o: fn(e)


def resample(trace: dict, dt: float) -> list[FrameState]:
    t = trace["t"]
    grid = np.arange(t[0], t[-1] + 1e-9, dt)
    if grid[-1] < t[-1]:
        grid = np.append(grid, t[-1])

    def at(key):
        return np.interp(grid, t, trace[key])

    return [
        FrameState(
            t=float(ti),
            x=float(x),
            v=float(v),
            posture=float(p),
            power=float(pw),
            w_bal=float(w),
            headwind=float(h),
            policy="",
        )
        for ti, x, v, p, pw, w, h in zip(at("t"), at("x"), at("v"), at("posture"), at("mean_power"), at("w_bal"), at("headwind"))
    ]


def collect(policy: str, model_path: Path) -> tuple[list[FrameState], np.ndarray, np.ndarray, float]:
    env = BikeWindEnv()
    actor = load_actor(policy, model_path)
    result = rollout(env, actor)
    frames = resample(result["trace"], dt=0.25)
    for fr in frames:
        fr.policy = policy
        fr.course_length = env.course_length
        fr.w_prime_max = env.p.w_prime_max
        fr.v_ref = env.v_ref
        fr.dt_control = env.p.dt_control
    frames[-1].finished = result["finished"]
    xs, hw = course_profile(env)
    return frames, xs, hw, result["time"]


def save_stills(out_dir: Path, policy: str, model_path: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = BikeWindEnv()
    xs, hw = course_profile(env)
    viz = Visualizer(xs, hw)
    paths = []

    synthetic = [
        ("upright", FrameState(200.0 / 9.6, 200.0, 9.6, 0.0, 250.0, env.p.w_prime_max, 1.0, "demo")),
        ("tuck", FrameState(200.0 / 10.2, 200.0, 10.2, 1.0, 200.0, env.p.w_prime_max, 1.0, "demo")),
    ]
    for name, state in synthetic:
        viz.update(state)
        path = out_dir / f"viz_{name}.png"
        viz.fig.savefig(path, dpi=120)
        paths.append(path)
    plt.close(viz.fig)

    frames, xs, hw, _ = collect(policy, model_path)
    viz = Visualizer(xs, hw)
    patch = min(frames, key=lambda fr: abs(fr.x - 2500.0))
    patch.policy = policy
    viz.update(patch)
    path = out_dir / "viz_headwind_patch.png"
    viz.fig.savefig(path, dpi=120)
    paths.append(path)
    plt.close("all")
    return paths


def animate(frames: list[FrameState], xs, hw, save: Path | None, fps: int, speed: float) -> None:
    viz = Visualizer(xs, hw)
    # Playback skips so wall-time ≈ sim_time / speed.
    step = max(1, int(round((speed / fps) / 0.25)))
    play = frames[::step]
    if play[-1] is not frames[-1]:
        play.append(frames[-1])

    def update(i):
        viz.update(play[i])
        return []

    anim = FuncAnimation(viz.fig, update, frames=len(play), interval=1000.0 / fps, blit=False)
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        anim.save(save, writer=PillowWriter(fps=fps), dpi=90)
        print(f"saved {save}")
        plt.close(viz.fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate the cyclist with live ride metrics.")
    parser.add_argument("--policy", default="sac", choices=["sac", *BASELINES])
    parser.add_argument("--model", default=str(RESULTS / "sac_bike.zip"))
    parser.add_argument("--save", default=None, help="Write an animated GIF to this path.")
    parser.add_argument("--stills", default=None, help="Directory for upright / tuck / patch PNGs.")
    parser.add_argument("--speed", type=float, default=24.0, help="Playback multiplier vs real time.")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    model_path = Path(args.model)
    if args.policy == "sac" and not model_path.exists():
        print(f"no model at {model_path}; using myopic")
        args.policy = "myopic"

    if args.stills:
        paths = save_stills(Path(args.stills), args.policy, model_path)
        for p in paths:
            print(f"saved {p}")
        if args.save is None:
            return

    print(f"rolling out {args.policy}…")
    frames, xs, hw, t_final = collect(args.policy, model_path)
    print(f"{len(frames)} samples, finish {t_final:.1f} s")
    save = Path(args.save) if args.save else None
    animate(frames, xs, hw, save, fps=args.fps, speed=args.speed)


if __name__ == "__main__":
    main()
