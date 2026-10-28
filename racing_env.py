from __future__ import annotations

from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def wrap_angle(angle: float) -> float:
    # Keep heading errors inside one signed revolution.
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class TrackSpec:
    name: str
    control_points: tuple[tuple[float, float], ...]
    main_straight: tuple[tuple[float, float], tuple[float, float]]
    half_width: float = 1.0
    samples_per_segment: int = 18

    def __post_init__(self):
        # Convert the design points into a smooth, closed Catmull-Rom centreline.
        controls = np.asarray(self.control_points, dtype=np.float64)
        if controls.shape[0] < 8:
            raise ValueError("a circuit needs at least eight control points")

        samples = []
        tangents = []
        curvatures = []
        for index in range(len(controls)):
            p0 = controls[(index - 1) % len(controls)]
            p1 = controls[index]
            p2 = controls[(index + 1) % len(controls)]
            p3 = controls[(index + 2) % len(controls)]
            for t in np.linspace(0.0, 1.0, self.samples_per_segment, endpoint=False):
                t2 = t * t
                t3 = t2 * t
                position = 0.5 * (
                    2.0 * p1
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                first = 0.5 * (
                    -p0
                    + p2
                    + 2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t
                    + 3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t2
                )
                second = 0.5 * (
                    2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)
                    + 6.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t
                )
                speed = max(float(np.linalg.norm(first)), 1e-9)
                samples.append(position)
                tangents.append(first / speed)
                curvatures.append(
                    float((first[0] * second[1] - first[1] * second[0]) / speed**3)
                )

        samples_array = np.asarray(samples, dtype=np.float64)
        tangents_array = np.asarray(tangents, dtype=np.float64)
        curvature_array = np.asarray(curvatures, dtype=np.float64)
        segment_vectors = np.roll(samples_array, -1, axis=0) - samples_array
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))

        object.__setattr__(self, "_samples", samples_array)
        object.__setattr__(self, "_tangents", tangents_array)
        object.__setattr__(self, "_curvatures", curvature_array)
        object.__setattr__(self, "_segment_vectors", segment_vectors)
        object.__setattr__(self, "_segment_lengths", segment_lengths)
        object.__setattr__(self, "_cumulative", cumulative)
        object.__setattr__(self, "_lap_length", float(cumulative[-1]))

        # Cache both walls for fast global LiDAR ray intersections.
        outer = self.sample(self.half_width, len(samples_array))
        inner = self.sample(-self.half_width, len(samples_array))
        wall_starts = np.concatenate((outer, inner), axis=0)
        wall_ends = np.concatenate(
            (np.roll(outer, -1, axis=0), np.roll(inner, -1, axis=0)), axis=0
        )
        wall_midpoints = 0.5 * (wall_starts + wall_ends)
        wall_half_lengths = 0.5 * np.linalg.norm(wall_ends - wall_starts, axis=1)
        all_edges = np.concatenate((outer, inner), axis=0)
        bounds = (
            float(all_edges[:, 0].min()),
            float(all_edges[:, 0].max()),
            float(all_edges[:, 1].min()),
            float(all_edges[:, 1].max()),
        )
        object.__setattr__(self, "_wall_starts", wall_starts)
        object.__setattr__(self, "_wall_ends", wall_ends)
        object.__setattr__(self, "_wall_midpoints", wall_midpoints)
        object.__setattr__(self, "_wall_half_lengths", wall_half_lengths)
        object.__setattr__(self, "_bounds", bounds)

    @property
    def lap_length(self) -> float:
        return self._lap_length

    @property
    def width(self) -> float:
        return 2.0 * self.half_width

    @property
    def main_straight_length(self) -> float:
        start = np.asarray(self.main_straight[0], dtype=np.float64)
        finish = np.asarray(self.main_straight[1], dtype=np.float64)
        return float(np.linalg.norm(finish - start))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self._bounds

    def geometry(self, progress: float, offset: float = 0.0):
        # Interpolate position, tangent, and signed curvature by arc length.
        wrapped = progress % self.lap_length
        index = int(np.searchsorted(self._cumulative, wrapped, side="right") - 1)
        index = min(index, len(self._samples) - 1)
        length = max(float(self._segment_lengths[index]), 1e-9)
        fraction = (wrapped - self._cumulative[index]) / length
        next_index = (index + 1) % len(self._samples)

        center = self._samples[index] + fraction * self._segment_vectors[index]
        tangent = (
            (1.0 - fraction) * self._tangents[index]
            + fraction * self._tangents[next_index]
        )
        tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        position = center + normal * offset
        heading = math.atan2(tangent[1], tangent[0])
        curvature = float(
            (1.0 - fraction) * self._curvatures[index]
            + fraction * self._curvatures[next_index]
        )
        return position, heading, curvature

    def sample(self, offset: float = 0.0, count: int = 360) -> np.ndarray:
        # Return equally spaced arc-length samples for tests and rendering.
        progress_values = np.linspace(0.0, self.lap_length, count, endpoint=False)
        return np.asarray(
            [self.geometry(progress, offset)[0] for progress in progress_values],
            dtype=np.float64,
        )

    def ray_distance(
        self, origin: np.ndarray, direction: np.ndarray, max_range: float
    ) -> float:
        # Intersect one LiDAR ray with nearby inner and outer wall segments.
        midpoint_distance = np.linalg.norm(self._wall_midpoints - origin, axis=1)
        nearby = midpoint_distance - self._wall_half_lengths <= max_range
        starts = self._wall_starts[nearby]
        segments = self._wall_ends[nearby] - starts
        relative = starts - origin
        denominator = direction[0] * segments[:, 1] - direction[1] * segments[:, 0]
        valid_denominator = np.abs(denominator) > 1e-9

        safe_denominator = np.where(valid_denominator, denominator, 1.0)
        ray_t = (
            relative[:, 0] * segments[:, 1] - relative[:, 1] * segments[:, 0]
        ) / safe_denominator
        segment_u = (
            relative[:, 0] * direction[1] - relative[:, 1] * direction[0]
        ) / safe_denominator
        valid = (
            valid_denominator
            & (ray_t > 1e-4)
            & (ray_t <= max_range)
            & (segment_u >= 0.0)
            & (segment_u <= 1.0)
        )
        if not np.any(valid):
            return max_range
        return float(np.min(ray_t[valid]))


# The bottom control points define a 15-unit start/finish straight.
TRAIN_POINTS = (
    (-7.5, -6.0),
    (0.0, -6.0),
    (7.5, -6.0),
    (10.5, -5.5),
    (13.0, -3.0),
    (13.5, 1.0),
    (13.0, 4.5),
    (11.0, 6.5),
    (8.0, 7.0),
    (6.5, 6.0),
    (6.0, 4.3),
    (7.0, 3.2),
    (9.0, 3.0),
    (10.2, 1.5),
    (10.0, -0.5),
    (8.8, -2.2),
    (7.2, -2.7),
    (5.8, -1.5),
    (2.0, 2.2),
    (0.0, 5.5),
    (-2.0, 2.0),
    (-3.2, -2.8),
    (-5.0, -3.8),
    (-7.2, -3.5),
    (-8.6, -2.2),
    (-8.7, -0.8),
    (-7.5, 0.2),
    (-5.7, 0.3),
    (-4.5, 1.2),
    (-4.6, 2.4),
    (-6.0, 3.1),
    (-8.0, 3.0),
    (-9.2, 3.8),
    (-9.0, 5.5),
    (-10.5, 7.0),
    (-13.0, 6.5),
    (-14.5, 4.5),
    (-15.0, 1.5),
    (-14.8, -2.5),
    (-13.0, -5.0),
    (-10.0, -6.0),
)

# The held-out circuit keeps the same sectors with shifted radii and apex positions.
HELDOUT_POINTS = tuple(
    (
        0.97 * x + 0.18 * math.sin(index * 0.73),
        0.94 * y + 0.15 * math.cos(index * 0.61),
    )
    for index, (x, y) in enumerate(TRAIN_POINTS)
)

TRACKS = {
    "train": TrackSpec(
        "train",
        control_points=TRAIN_POINTS,
        main_straight=((-7.5, -6.0), (7.5, -6.0)),
        half_width=1.0,
    ),
    "heldout": TrackSpec(
        "heldout",
        control_points=HELDOUT_POINTS,
        main_straight=((-7.5, -5.49), (7.5, -5.49)),
        half_width=1.0,
    ),
}


class RacingEnv(gym.Env):
    metadata = {"render_modes": []}

    BEAMS = 24
    MAX_RANGE = 8.0
    MAX_SPEED = 4.0
    MIN_SPEED = -0.8
    MAX_STEER = 0.45
    WHEELBASE = 0.33
    ACCELERATION = 2.5
    DT = 0.1

    def __init__(
        self,
        *,
        traffic: bool = False,
        opponent_style: str = "steady",
        reward_profile: str = "v2",
        track: str = "train",
        max_steps: int = 800,
    ):
        super().__init__()
        if reward_profile not in {"v1", "v2"}:
            raise ValueError("reward_profile must be 'v1' or 'v2'")
        if track not in TRACKS:
            raise ValueError(f"unknown track: {track}")
        if opponent_style not in {"steady", "weaver"}:
            raise ValueError("opponent_style must be 'steady' or 'weaver'")

        # Keep the learning contract independent from the selected circuit.
        self.traffic = traffic
        self.opponent_style = opponent_style
        self.reward_profile = reward_profile
        self.track = TRACKS[track]
        self.max_steps = max_steps
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(33,), dtype=np.float32)
        self.beam_angles = np.linspace(
            -0.75 * math.pi, 0.75 * math.pi, self.BEAMS, dtype=np.float64
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        # Reset race state while retaining deterministic seeded offsets.
        self.s_total = 0.0
        self.lateral_offset = float(self.np_random.uniform(-0.05, 0.05))
        self.heading_error = float(self.np_random.uniform(-0.03, 0.03))
        self.speed = 1.0
        self.steering = 0.0
        self.step_count = 0
        self.wall_contacts = 0
        self.track_limit_violations = 0
        self.caused_contacts = 0
        self.overtakes = 0
        self.reverse_steps = 0
        self.previous_wall = False
        self.previous_out = False
        self.previous_race_difference = -0.25 * self.track.lap_length
        self.opponent_total = 0.25 * self.track.lap_length
        self.opponent_speed = 1.7
        self.last_lap_time = None
        return self._observation(), self._info(completion=False)

    def ego_position(self) -> np.ndarray:
        return self.track.geometry(self.s_total, self.lateral_offset)[0]

    def _opponent_state(self):
        # Scripted traffic either holds its line or weaves across the ribbon.
        if self.opponent_style == "steady":
            speed = 1.7
            offset = 0.0
        else:
            speed = 1.55 + 0.30 * math.sin(0.08 * self.step_count)
            offset = 0.28 * math.sin(0.18 * self.opponent_total)
        position = self.track.geometry(self.opponent_total, offset)[0]
        return position, speed, offset

    def _lidar(self) -> np.ndarray:
        # Cast each beam against the complete circuit so angled walls affect range.
        ego_position, track_heading, _ = self.track.geometry(
            self.s_total, self.lateral_offset
        )
        car_yaw = track_heading + self.heading_error
        ranges = np.full(self.BEAMS, self.MAX_RANGE, dtype=np.float64)
        for index, beam in enumerate(self.beam_angles):
            direction = np.array(
                [math.cos(car_yaw + beam), math.sin(car_yaw + beam)],
                dtype=np.float64,
            )
            ranges[index] = self.track.ray_distance(
                ego_position, direction, self.MAX_RANGE
            )

        # Treat the opponent as a circle that can occlude a LiDAR beam.
        if self.traffic:
            opponent_position, _, _ = self._opponent_state()
            relative = opponent_position - ego_position
            for index, beam in enumerate(self.beam_angles):
                direction = np.array(
                    [math.cos(car_yaw + beam), math.sin(car_yaw + beam)]
                )
                projection = float(np.dot(relative, direction))
                perpendicular_sq = float(np.dot(relative, relative) - projection**2)
                radius = 0.24
                if projection > 0.0 and perpendicular_sq <= radius**2:
                    distance = projection - math.sqrt(
                        max(0.0, radius**2 - perpendicular_sq)
                    )
                    if distance > 0.0:
                        ranges[index] = min(ranges[index], distance)

        return np.clip(ranges, 0.0, self.MAX_RANGE).astype(np.float32)

    def _observation(self) -> np.ndarray:
        # Combine LiDAR, vehicle state, future curvature, and traffic state.
        curvature_values = []
        for lookahead in (0.0, 4.0, 8.0):
            curvature = self.track.geometry(self.s_total + lookahead)[2]
            curvature_values.append(float(np.clip(curvature / 0.35, -1.0, 1.0)))

        if self.traffic:
            gap = (self.opponent_total - self.s_total) % self.track.lap_length
            _, opponent_speed, _ = self._opponent_state()
            relative_speed = (opponent_speed - self.speed) / self.MAX_SPEED
        else:
            gap = self.track.lap_length
            relative_speed = 0.0

        observation = np.concatenate(
            [
                self._lidar() / self.MAX_RANGE,
                np.array(
                    [
                        self.speed / self.MAX_SPEED,
                        self.steering / self.MAX_STEER,
                        self.heading_error / math.pi,
                        self.lateral_offset / self.track.half_width,
                    ],
                    dtype=np.float32,
                ),
                np.asarray(curvature_values, dtype=np.float32),
                np.array(
                    [gap / self.track.lap_length, relative_speed], dtype=np.float32
                ),
            ]
        )
        return np.clip(observation, -1.0, 1.0).astype(np.float32)

    def _hazards(self):
        # Track limits remain tied to the uniform two-unit racing ribbon.
        wall = abs(self.lateral_offset) >= self.track.half_width - 0.12
        out = abs(self.lateral_offset) > self.track.half_width
        collision = False
        caused = False
        if self.traffic:
            opponent_position, opponent_speed, _ = self._opponent_state()
            collision = float(np.linalg.norm(self.ego_position() - opponent_position)) < 0.42
            caused = collision and self.speed > opponent_speed + 0.2
        return wall, out, collision, caused

    def _reward(
        self,
        delta_progress: float,
        lap_event: bool,
        wall: bool,
        out: bool,
        collision: bool,
        caused: bool,
    ):
        # Preserve the flawed v1 proxy beside the corrected v2 contract.
        if self.reward_profile == "v1":
            terms = {
                "progress": 4.0 * abs(delta_progress),
                "lap": 25.0 * float(lap_event),
                "wall": 0.0,
                "out": -2.0 * float(out),
                "collision": -2.0 * float(collision),
                "caused": 0.0,
                "reverse": 0.0,
            }
        else:
            terms = {
                "progress": 4.0 * delta_progress,
                "lap": 25.0 * float(lap_event),
                "wall": -1.5 * float(wall),
                "out": -20.0 * float(out),
                "collision": -25.0 * float(collision),
                "caused": -10.0 * float(caused),
                "reverse": -2.0 * float(delta_progress < 0.0),
            }
        return float(sum(terms.values())), terms

    def step(self, action):
        # Apply bounded steering and acceleration in the local Frenet frame.
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target_steering = float(action[0]) * self.MAX_STEER
        self.steering = 0.7 * self.steering + 0.3 * target_steering
        self.speed = float(
            np.clip(
                self.speed + float(action[1]) * self.ACCELERATION * self.DT,
                self.MIN_SPEED,
                self.MAX_SPEED,
            )
        )

        _, _, curvature = self.track.geometry(self.s_total)
        denominator = max(0.5, 1.0 - curvature * self.lateral_offset)
        longitudinal_rate = self.speed * math.cos(self.heading_error) / denominator
        lateral_rate = self.speed * math.sin(self.heading_error)
        yaw_rate = self.speed * math.tan(self.steering) / self.WHEELBASE

        previous_progress = self.s_total
        previous_laps = math.floor(max(0.0, previous_progress) / self.track.lap_length)
        self.s_total += longitudinal_rate * self.DT
        self.lateral_offset += lateral_rate * self.DT
        self.heading_error = wrap_angle(
            self.heading_error
            + (yaw_rate - curvature * longitudinal_rate) * self.DT
        )
        delta_progress = self.s_total - previous_progress
        current_laps = math.floor(max(0.0, self.s_total) / self.track.lap_length)
        lap_event = current_laps > previous_laps

        if self.traffic:
            _, self.opponent_speed, _ = self._opponent_state()
            self.opponent_total += self.opponent_speed * self.DT

        # Record behavioural metrics independently from episode reward.
        self.step_count += 1
        wall, out, collision, caused = self._hazards()
        if wall and not self.previous_wall:
            self.wall_contacts += 1
        if out and not self.previous_out:
            self.track_limit_violations += 1
        if caused:
            self.caused_contacts += 1
        if delta_progress < -1e-4:
            self.reverse_steps += 1

        if self.traffic:
            difference = self.s_total - self.opponent_total
            if self.previous_race_difference <= 0.0 < difference:
                self.overtakes += 1
            self.previous_race_difference = difference

        reward, reward_terms = self._reward(
            delta_progress, lap_event, wall, out, collision, caused
        )
        terminated = bool(lap_event or out or collision)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        completion = bool(lap_event)
        if completion:
            self.last_lap_time = self.step_count * self.DT

        self.previous_wall = wall
        self.previous_out = out
        info = self._info(completion=completion)
        info["reward_terms"] = reward_terms
        return self._observation(), reward, terminated, truncated, info

    def _info(self, *, completion: bool):
        # Expose evaluation evidence without using the reward total as a result.
        return {
            "completion": completion,
            "lap_time": self.last_lap_time,
            "wall_contacts": self.wall_contacts,
            "track_limit_violations": self.track_limit_violations,
            "caused_contacts": self.caused_contacts,
            "overtakes": self.overtakes,
            "reverse_steps": self.reverse_steps,
            "distance": self.s_total,
        }