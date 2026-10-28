# Reward Contract

## Intended behaviour

Complete fast laps while staying inside track limits, avoiding walls, avoiding caused contacts, and making forward progress. Reward is a training proxy, not the evaluation target.

## Observation and action contract

The observation is a 33-value float vector: 24 normalized LiDAR-like ranges, speed, steering angle, heading error, lateral offset, three curvature look-ahead values, opponent gap, and opponent relative speed.

The action is `[steering_command, acceleration_command]`, with both values clipped to `[-1, 1]`.

## Reward v1: deliberately flawed smoke probe

- Pays for absolute centreline progress.
- Pays a lap bonus.
- Applies weak track-limit and collision penalties.

Known failure: moving forward and backward over the same section can collect positive proxy reward without completing a lap.

## Reward v2: training contract

- Pays signed centreline progress.
- Pays a lap-completion bonus.
- Penalizes reverse progress, wall proximity, leaving the track, collisions, and contacts caused while closing faster than the opponent.

## Behavioural evaluation

Do not use episode reward as the result. Report completion, lap time, wall contacts, track-limit violations, caused contacts, overtakes, and reverse steps. This project is a smoke experiment with fixed small budgets, not a performance claim.