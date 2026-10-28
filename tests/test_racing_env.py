import numpy as np
import pytest

from racing_env import RacingEnv, TRACKS


def segments_intersect(a, b, c, d):
    # Detect proper crossings while allowing neighbouring segments to share endpoints.
    first = b - a
    second = d - c
    denominator = first[0] * second[1] - first[1] * second[0]
    if abs(denominator) < 1e-9:
        return False
    relative = c - a
    first_fraction = (
        relative[0] * second[1] - relative[1] * second[0]
    ) / denominator
    second_fraction = (
        relative[0] * first[1] - relative[1] * first[0]
    ) / denominator
    return 1e-6 < first_fraction < 1.0 - 1e-6 and 1e-6 < second_fraction < 1.0 - 1e-6


def assert_simple_closed_polyline(points):
    # Compare every non-neighbouring segment to enforce one continuous loop.
    segment_count = len(points)
    for first_index in range(segment_count):
        first_next = (first_index + 1) % segment_count
        for second_index in range(first_index + 1, segment_count):
            second_next = (second_index + 1) % segment_count
            neighbours = {
                first_index,
                first_next,
            } & {
                second_index,
                second_next,
            }
            if neighbours:
                continue
            assert not segments_intersect(
                points[first_index],
                points[first_next],
                points[second_index],
                points[second_next],
            )


def test_observation_and_action_contract():
    env = RacingEnv()
    observation, _ = env.reset(seed=7)
    assert observation.shape == (33,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert env.action_space.shape == (2,)


def test_reset_is_seeded():
    first = RacingEnv()
    second = RacingEnv()
    observation_a, _ = first.reset(seed=42)
    observation_b, _ = second.reset(seed=42)
    assert np.allclose(observation_a, observation_b)


def test_track_geometry_contract():
    track = TRACKS["train"]
    assert track.width == pytest.approx(2.0)
    assert track.main_straight_length == pytest.approx(15.0)
    assert np.allclose(track.geometry(0.0)[0], track.geometry(track.lap_length)[0])

    # Check uniform width and alternating turn directions around the complete lap.
    progress_values = np.linspace(0.0, track.lap_length, 64, endpoint=False)
    curvatures = []
    for progress in progress_values:
        left = track.geometry(progress, track.half_width)[0]
        right = track.geometry(progress, -track.half_width)[0]
        assert np.linalg.norm(left - right) == pytest.approx(2.0, abs=1e-6)
        curvatures.append(track.geometry(progress)[2])
    assert max(curvatures) > 0.03
    assert min(curvatures) < -0.03


def test_track_has_no_self_intersections():
    track = TRACKS["train"]
    assert_simple_closed_polyline(track.sample(count=180))


def test_lidar_reads_complex_wall_angles():
    env = RacingEnv()
    env.reset(seed=5)
    ranges = env._lidar()
    assert ranges.shape == (24,)
    assert np.all(ranges > 0.0)
    assert np.all(ranges <= env.MAX_RANGE)
    assert np.ptp(ranges) > 0.1


def test_track_limit_and_collision_logic():
    solo = RacingEnv()
    solo.reset(seed=1)
    solo.lateral_offset = solo.track.half_width + 0.01
    _, out, collision, _ = solo._hazards()
    assert out
    assert not collision

    traffic = RacingEnv(traffic=True)
    traffic.reset(seed=1)
    traffic.opponent_total = traffic.s_total
    _, _, collision, _ = traffic._hazards()
    assert collision


def test_reward_terms_remove_round_trip_exploit():
    v1 = RacingEnv(reward_profile="v1")
    v2 = RacingEnv(reward_profile="v2")
    v1_round_trip = sum(
        v1._reward(delta, False, False, False, False, False)[0]
        for delta in (0.1, -0.1)
    )
    v2_round_trip = sum(
        v2._reward(delta, False, False, False, False, False)[0]
        for delta in (0.1, -0.1)
    )
    assert v1_round_trip > 0.0
    assert v2_round_trip < 0.0


def test_lap_completion_terminates():
    env = RacingEnv()
    env.reset(seed=2)
    env.s_total = env.track.lap_length - 0.02
    env.speed = 1.5
    _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
    assert terminated
    assert not truncated
    assert info["completion"]
    assert info["lap_time"] is not None


def test_time_limit_truncates_without_termination():
    env = RacingEnv(max_steps=1)
    env.reset(seed=3)
    env.speed = 0.0
    _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
    assert not terminated
    assert truncated
    assert not info["completion"]