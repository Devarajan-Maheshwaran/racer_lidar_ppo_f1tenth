from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from racing_env import RacingEnv, TRACKS

ARTIFACTS = Path("artifacts")
TRAIN_SEEDS = [11, 22, 33]
EVAL_SEEDS = [101, 202, 303]
SOLO_STEPS = 30_000
TRAFFIC_STEPS = 10_000


def make_model(env: RacingEnv, seed: int) -> PPO:
    # Keep the actor and critic small enough for reproducible CPU smoke runs.
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device="cpu",
        verbose=0,
    )


def pure_pursuit_action(env: RacingEnv) -> np.ndarray:
    # Add curvature feed-forward so the baseline can enter tight corners early.
    _, _, curvature = env.track.geometry(env.s_total + 1.2)
    feed_forward = math.atan(env.WHEELBASE * curvature)
    steering = np.clip(
        (feed_forward - 1.4 * env.heading_error - 0.8 * env.lateral_offset)
        / env.MAX_STEER,
        -1.0,
        1.0,
    )
    target_speed = float(np.clip(2.8 / (1.0 + 5.0 * abs(curvature)), 1.0, 2.6))
    acceleration = np.clip((target_speed - env.speed) / 1.5, -1.0, 1.0)
    return np.array([steering, acceleration], dtype=np.float32)


def rollout(
    label: str,
    *,
    model: PPO | None,
    seed: int,
    traffic: bool,
    opponent_style: str,
    track: str,
):
    # Run one episode and keep behavioural evidence plus the driven trace.
    env = RacingEnv(
        traffic=traffic,
        opponent_style=opponent_style,
        reward_profile="v2",
        track=track,
    )
    observation, info = env.reset(seed=seed)
    trace = [env.ego_position().copy()]
    terminated = truncated = False
    while not (terminated or truncated):
        if label == "pure_pursuit":
            action = pure_pursuit_action(env)
        else:
            action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        trace.append(env.ego_position().copy())
    row = {
        "policy": label,
        "seed": seed,
        "completion": int(info["completion"]),
        "lap_time": "" if info["lap_time"] is None else info["lap_time"],
        "wall_contacts": info["wall_contacts"],
        "track_limit_violations": info["track_limit_violations"],
        "caused_contacts": info["caused_contacts"],
        "overtakes": info["overtakes"],
        "reverse_steps": info["reverse_steps"],
        "distance": info["distance"],
    }
    env.close()
    return row, trace


def evaluate(
    label: str,
    model: PPO | None,
    *,
    traffic: bool,
    opponent_style: str,
    track: str,
):
    # Repeat evaluation across fixed seeds without changing the training budget.
    rows = []
    first_trace = None
    for seed in EVAL_SEEDS:
        row, trace = rollout(
            label,
            model=model,
            seed=seed,
            traffic=traffic,
            opponent_style=opponent_style,
            track=track,
        )
        rows.append(row)
        if first_trace is None:
            first_trace = trace
    return rows, first_trace


def write_csv(path: Path, rows: list[dict]):
    # Store the same behavioural columns for every policy and circuit.
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "policy",
        "seed",
        "completion",
        "lap_time",
        "wall_contacts",
        "track_limit_violations",
        "caused_contacts",
        "overtakes",
        "reverse_steps",
        "distance",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]):
    # Print smoke statistics without presenting them as final performance claims.
    print("Smoke metrics, not final results")
    labels = sorted({row["policy"] for row in rows})
    for label in labels:
        selected = [row for row in rows if row["policy"] == label]
        completions = [float(row["completion"]) for row in selected]
        violations = [float(row["track_limit_violations"]) for row in selected]
        print(
            f"{label}: completion={mean(completions):.3f} +/- {pstdev(completions):.3f}, "
            f"track_limits={mean(violations):.3f} +/- {pstdev(violations):.3f}"
        )


def circuit_features(track):
    # Build reusable geometry for static and animated circuit previews.
    center = track.sample(0.0, 480)
    outer = track.sample(track.half_width, 480)
    inner = track.sample(-track.half_width, 480)

    start = np.asarray(track.main_straight[0], dtype=np.float64)
    finish = np.asarray(track.main_straight[1], dtype=np.float64)
    straight = finish - start
    direction = straight / np.linalg.norm(straight)
    outside = np.array([direction[1], -direction[0]], dtype=np.float64)
    entry = start + 0.16 * straight
    exit_point = start + 0.84 * straight
    pit = np.asarray(
        [
            entry,
            entry + 1.45 * track.half_width * outside,
            exit_point + 1.45 * track.half_width * outside,
            exit_point,
        ]
    )

    # Alternate curb colours only where signed curvature identifies an inner apex.
    curbs = []
    curb_count = 420
    for index in range(curb_count):
        first_progress = index * track.lap_length / curb_count
        second_progress = (index + 1) * track.lap_length / curb_count
        _, _, curvature = track.geometry(0.5 * (first_progress + second_progress))
        if abs(curvature) < 0.075:
            continue
        inner_offset = math.copysign(track.half_width, curvature)
        first = track.geometry(first_progress, inner_offset)[0]
        second = track.geometry(second_progress, inner_offset)[0]
        colour = "#d62828" if index % 2 == 0 else "#f7f7f2"
        curbs.append((first, second, colour))

    start_finish = np.asarray(
        [
            track.geometry(0.0, -track.half_width)[0],
            track.geometry(0.0, track.half_width)[0],
        ]
    )
    return center, outer, inner, pit, curbs, start_finish


def write_svg(path: Path, trace: list[np.ndarray], track_name: str):
    # Render arbitrary spline bounds instead of assuming an ellipse.
    track = TRACKS[track_name]
    width, height, padding = 1100, 680, 42
    minimum_x, maximum_x, minimum_y, maximum_y = track.bounds
    margin = 1.8
    minimum_x -= margin
    maximum_x += margin
    minimum_y -= margin
    maximum_y += margin
    scale = min(
        (width - 2 * padding) / (maximum_x - minimum_x),
        (height - 2 * padding) / (maximum_y - minimum_y),
    )

    def screen(point):
        return (
            padding + (point[0] - minimum_x) * scale,
            height - padding - (point[1] - minimum_y) * scale,
        )

    def polyline(points, closed=False):
        rendered = list(points)
        if closed:
            rendered.append(points[0])
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in map(screen, rendered))

    center, outer, inner, pit, curbs, start_finish = circuit_features(track)
    asphalt_width = 2.0 * track.half_width * scale
    curb_svg = "\n".join(
        f'<line x1="{screen(first)[0]:.1f}" y1="{screen(first)[1]:.1f}" '
        f'x2="{screen(second)[0]:.1f}" y2="{screen(second)[1]:.1f}" '
        f'stroke="{colour}" stroke-width="7"/>'
        for first, second, colour in curbs
    )
    start_a, start_b = map(screen, start_finish)

    path.parent.mkdir(parents=True, exist_ok=True)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#0b3d25"/>
<polyline points="{polyline(center, closed=True)}" fill="none" stroke="#3f464d" stroke-width="{asphalt_width:.1f}" stroke-linejoin="round" stroke-linecap="round"/>
<polyline points="{polyline(outer, closed=True)}" fill="none" stroke="#d8dee9" stroke-width="3"/>
<polyline points="{polyline(inner, closed=True)}" fill="none" stroke="#d8dee9" stroke-width="3"/>
{curb_svg}
<polyline points="{polyline(pit)}" fill="none" stroke="#f4d35e" stroke-width="3" stroke-dasharray="10 7"/>
<line x1="{start_a[0]:.1f}" y1="{start_a[1]:.1f}" x2="{start_b[0]:.1f}" y2="{start_b[1]:.1f}" stroke="white" stroke-width="6" stroke-dasharray="7 7"/>
<polyline points="{polyline(trace)}" fill="none" stroke="#ff5a5f" stroke-width="3"/>
<text x="25" y="35" fill="white" font-family="sans-serif" font-size="20">Complex circuit smoke trajectory: {track.name}</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def demo_frame(env: RacingEnv) -> dict:
    # Capture positions and headings so the browser can draw both cars.
    ego_position, track_heading, _ = env.track.geometry(
        env.s_total, env.lateral_offset
    )
    opponent_position, _, opponent_offset = env._opponent_state()
    _, opponent_heading, _ = env.track.geometry(env.opponent_total, opponent_offset)
    return {
        "ego": [*ego_position.tolist(), track_heading + env.heading_error],
        "opponent": [*opponent_position.tolist(), opponent_heading],
    }


def collect_demo(model: PPO, seed: int):
    # Replay one deterministic evaluation with the trained car and traffic.
    env = RacingEnv(
        traffic=True,
        opponent_style="weaver",
        reward_profile="v2",
        track="train",
    )
    observation, info = env.reset(seed=seed)
    frames = [demo_frame(env)]
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        frames.append(demo_frame(env))
    env.close()
    return frames, info


def write_demo_html(
    path: Path,
    frames: list[dict],
    info: dict,
    checkpoint: str,
):
    # Package the measured rollout and complete circuit styling into one page.
    track = TRACKS["train"]
    center, outer, inner, pit, curbs, start_finish = circuit_features(track)
    minimum_x, maximum_x, minimum_y, maximum_y = track.bounds
    payload = json.dumps(
        {
            "center": center.tolist(),
            "outer": outer.tolist(),
            "inner": inner.tolist(),
            "pit": pit.tolist(),
            "curbs": [
                [first.tolist(), second.tolist(), colour]
                for first, second, colour in curbs
            ],
            "startFinish": start_finish.tolist(),
            "frames": frames,
            "checkpoint": checkpoint,
            "completion": bool(info["completion"]),
            "bounds": [minimum_x, maximum_x, minimum_y, maximum_y],
            "trackWidth": track.width,
            "intervalMs": 50,
        }
    )
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Traffic-trained PPO replay</title>
<style>
body { margin: 0; background: #0b3d25; color: #f5f7fa; font-family: system-ui, sans-serif; text-align: center; }
canvas { width: min(96vw, 1100px); height: auto; border: 1px solid #313943; background: #0b3d25; }
.legend { display: flex; justify-content: center; gap: 24px; margin: 8px; }
.ego { color: #ff5a5f; } .traffic { color: #56b4e9; }
</style>
</head>
<body>
<h1>Traffic-trained PPO replay</h1>
<p id="status"></p>
<div class="legend"><span class="ego">Trained PPO car</span><span class="traffic">Scripted traffic</span></div>
<canvas width="1100" height="680"></canvas>
<script>
const data = __DATA__;
const canvas = document.querySelector("canvas");
const ctx = canvas.getContext("2d");
const padding = 42;
const [minX, maxX, minY, maxY] = data.bounds;
const scale = Math.min((canvas.width - 2 * padding) / (maxX - minX + 3.6), (canvas.height - 2 * padding) / (maxY - minY + 3.6));
const screen = p => [padding + (p[0] - minX + 1.8) * scale, canvas.height - padding - (p[1] - minY + 1.8) * scale];
function line(points, close = false) {
  ctx.beginPath();
  points.forEach((point, index) => { const [x, y] = screen(point); index ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  if (close) ctx.closePath();
  ctx.stroke();
}
function car(state, color, label) {
  const [x, y] = screen(state);
  ctx.save(); ctx.translate(x, y); ctx.rotate(-state[2]);
  ctx.fillStyle = color; ctx.fillRect(-11, -6, 22, 12);
  ctx.fillStyle = "#101418"; ctx.fillRect(3, -4, 5, 8); ctx.restore();
  ctx.fillStyle = color; ctx.fillText(label, x + 12, y - 10);
}
let index = 0;
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#3f464d"; ctx.lineWidth = data.trackWidth * scale; ctx.lineJoin = "round"; ctx.lineCap = "round"; line(data.center, true);
  ctx.strokeStyle = "#d8dee9"; ctx.lineWidth = 3; line(data.outer, true); line(data.inner, true);
  data.curbs.forEach(curb => { ctx.strokeStyle = curb[2]; ctx.lineWidth = 7; line([curb[0], curb[1]]); });
  ctx.setLineDash([10, 7]); ctx.strokeStyle = "#f4d35e"; ctx.lineWidth = 3; line(data.pit); ctx.setLineDash([]);
  ctx.setLineDash([7, 7]); ctx.strokeStyle = "white"; ctx.lineWidth = 6; line(data.startFinish); ctx.setLineDash([]);
  const frame = data.frames[index];
  car(frame.ego, "#ff5a5f", "PPO"); car(frame.opponent, "#56b4e9", "Traffic");
  document.querySelector("#status").textContent = `${data.checkpoint} | frame ${index + 1}/${data.frames.length} | lap complete: ${data.completion}`;
  index = (index + 1) % data.frames.length;
}
draw(); setInterval(draw, data.intervalMs);
</script>
</body>
</html>""".replace("__DATA__", payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def preview():
    # Validate the environment before writing the first visible circuit result.
    ARTIFACTS.mkdir(exist_ok=True)
    env = RacingEnv(traffic=False, reward_profile="v2")
    check_env(env)
    observation, _ = env.reset(seed=101)
    print(f"Observation shape: {observation.shape}")
    _, trace = evaluate(
        "pure_pursuit",
        None,
        traffic=False,
        opponent_style="steady",
        track="train",
    )
    write_svg(ARTIFACTS / "preview.svg", trace, "train")
    print("Wrote artifacts/preview.svg")


def exploit():
    # Run the deterministic round trip against both reward profiles.
    v1 = RacingEnv(reward_profile="v1")
    v2 = RacingEnv(reward_profile="v2")
    v1_total = 0.0
    v2_total = 0.0
    for delta in [0.1, -0.1] * 10:
        v1_total += v1._reward(delta, False, False, False, False, False)[0]
        v2_total += v2._reward(delta, False, False, False, False, False)[0]
    print(f"v1 round-trip proxy total: {v1_total:.3f}")
    print(f"v2 round-trip proxy total: {v2_total:.3f}")
    print("The v1 proxy pays for progress farming; v2 does not.")


def train_solo():
    # Train three independent solo policies and apply the lap-completion gate.
    ARTIFACTS.mkdir(exist_ok=True)
    all_rows = []
    passing_seeds = []
    saved_trace = None

    baseline_env = RacingEnv(traffic=False, reward_profile="v2")
    untrained = make_model(baseline_env, seed=999)
    rows, _ = evaluate(
        "untrained",
        untrained,
        traffic=False,
        opponent_style="steady",
        track="train",
    )
    all_rows.extend(rows)
    rows, _ = evaluate(
        "pure_pursuit",
        None,
        traffic=False,
        opponent_style="steady",
        track="train",
    )
    all_rows.extend(rows)

    for seed in TRAIN_SEEDS:
        env = RacingEnv(traffic=False, reward_profile="v2")
        model = make_model(env, seed)
        print(f"Training solo seed {seed} for {SOLO_STEPS} steps")
        model.learn(total_timesteps=SOLO_STEPS)
        model.save(str(ARTIFACTS / f"solo_seed_{seed}"))
        rows, trace = evaluate(
            f"ppo_seed_{seed}",
            model,
            traffic=False,
            opponent_style="steady",
            track="train",
        )
        all_rows.extend(rows)
        if sum(row["completion"] for row in rows) >= 2:
            passing_seeds.append(seed)
        if saved_trace is None:
            saved_trace = trace
        env.close()

    gate_passed = len(passing_seeds) >= 2
    summary = {
        "label": "smoke experiment",
        "training_seeds": TRAIN_SEEDS,
        "evaluation_seeds": EVAL_SEEDS,
        "solo_steps_per_seed": SOLO_STEPS,
        "passing_seeds": passing_seeds,
        "traffic_gate_passed": gate_passed,
    }
    write_csv(ARTIFACTS / "solo_smoke.csv", all_rows)
    (ARTIFACTS / "solo_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_svg(ARTIFACTS / "solo_trajectory.svg", saved_trace, "train")
    print_summary(all_rows)
    print(f"Traffic gate passed: {gate_passed}; passing seeds: {passing_seeds}")


def load_solo_summary():
    # Refuse traffic training until the solo experiment has produced a gate result.
    path = ARTIFACTS / "solo_summary.json"
    if not path.exists():
        raise SystemExit("Run train-solo first. Traffic remains disabled.")
    return json.loads(path.read_text(encoding="utf-8"))


def train_traffic():
    # Continue only the solo checkpoints that passed the fixed completion gate.
    summary = load_solo_summary()
    if not summary["traffic_gate_passed"]:
        print("Traffic skipped: fewer than two solo policies passed the lap gate.")
        return

    all_rows = []
    saved_trace = None
    for seed in summary["passing_seeds"]:
        env = RacingEnv(
            traffic=True, opponent_style="steady", reward_profile="v2"
        )
        model = PPO.load(
            str(ARTIFACTS / f"solo_seed_{seed}"), env=env, device="cpu"
        )
        print(f"Continuing seed {seed} with traffic for {TRAFFIC_STEPS} steps")
        model.learn(total_timesteps=TRAFFIC_STEPS, reset_num_timesteps=False)
        model.save(str(ARTIFACTS / f"traffic_seed_{seed}"))
        rows, trace = evaluate(
            f"traffic_ppo_seed_{seed}",
            model,
            traffic=True,
            opponent_style="weaver",
            track="train",
        )
        all_rows.extend(rows)
        if saved_trace is None:
            saved_trace = trace
        env.close()

    baseline_env = RacingEnv(
        traffic=True, opponent_style="weaver", reward_profile="v2"
    )
    untrained = make_model(baseline_env, seed=999)
    rows, _ = evaluate(
        "untrained",
        untrained,
        traffic=True,
        opponent_style="weaver",
        track="train",
    )
    all_rows.extend(rows)
    rows, _ = evaluate(
        "pure_pursuit",
        None,
        traffic=True,
        opponent_style="weaver",
        track="train",
    )
    all_rows.extend(rows)

    write_csv(ARTIFACTS / "traffic_smoke.csv", all_rows)
    write_svg(ARTIFACTS / "traffic_trajectory.svg", saved_trace, "train")
    print_summary(all_rows)


def demo():
    # Use the first traffic-qualified checkpoint for a reproducible visual proof.
    summary = load_solo_summary()
    models = [
        seed
        for seed in summary["passing_seeds"]
        if (ARTIFACTS / f"traffic_seed_{seed}.zip").exists()
    ]
    if not models:
        print("No traffic-qualified checkpoint exists; animated demo skipped.")
        return
    seed = models[0]
    checkpoint = f"traffic_seed_{seed}"
    model = PPO.load(str(ARTIFACTS / checkpoint), device="cpu")
    frames, info = collect_demo(model, seed=EVAL_SEEDS[0])
    write_demo_html(
        ARTIFACTS / "traffic_demo.html",
        frames,
        info,
        checkpoint,
    )
    print("Wrote artifacts/traffic_demo.html")


def heldout():
    # Measure qualified traffic policies on the altered circuit geometry.
    summary = load_solo_summary()
    models = [
        seed
        for seed in summary["passing_seeds"]
        if (ARTIFACTS / f"traffic_seed_{seed}.zip").exists()
    ]
    if not models:
        print("No traffic-qualified checkpoint exists; held-out circuit skipped.")
        return

    rows = []
    saved_trace = None
    for seed in models:
        model = PPO.load(ARTIFACTS / f"traffic_seed_{seed}", device="cpu")
        model_rows, trace = evaluate(
            f"traffic_ppo_seed_{seed}",
            model,
            traffic=True,
            opponent_style="weaver",
            track="heldout",
        )
        rows.extend(model_rows)
        if saved_trace is None:
            saved_trace = trace
    write_csv(ARTIFACTS / "heldout_circuit.csv", rows)
    write_svg(ARTIFACTS / "heldout_trajectory.svg", saved_trace, "heldout")
    print_summary(rows)


def main():
    # Dispatch one reproducible experiment command at a time.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "preview",
            "exploit",
            "train-solo",
            "train-traffic",
            "demo",
            "heldout",
        ],
    )
    args = parser.parse_args()
    {
        "preview": preview,
        "exploit": exploit,
        "train-solo": train_solo,
        "train-traffic": train_traffic,
        "demo": demo,
        "heldout": heldout,
    }[args.command]()


if __name__ == "__main__":
    main()