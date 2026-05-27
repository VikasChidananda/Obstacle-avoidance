# Terrain Obstacle Environment

This document describes the procedural terrain obstacle task used by `Quadcontrol-TerrainObstacle-v0`.

## Environment Summary

The task places a quadrotor on a procedurally generated heightfield with discrete obstacles and uses a lidar raycaster
for obstacle sensing. The policy observes body-frame kinematics, gravity, relative waypoint target, and lidar ranges.
Waypoints progress along the terrain row direction with a per-env lateral offset to create distinct lanes.

## Key Components

- Terrain generator: `TerrainImporterCfg` with `terrain_type="generator"` and `HfDiscreteObstaclesTerrainCfg`.
- Lidar: `RayCaster` with a 5-ray horizontal scan (`lidar_num_rays=5`, `horizontal_fov_range=(-90, 90)`).
- Waypoints: derived from terrain row origins; per-env `y` offsets are randomized on reset.
- Hover delay: waypoint advancement is gated by `waypoint_hover_steps` after reset.

## Default Parameters

- Terrain grid: `num_rows=16`, `num_cols=16`, `size=(4.0, 4.0)`.
- Obstacles: width range `(0.2, 0.6)`, height range `(0.2, 1.0)`, `num_obstacles=30`.
- Waypoints: `waypoint_stride=1`, `waypoint_threshold=0.6`, `waypoint_hover_steps=120`.
- Lidar: `lidar_max_range=3.0`, mesh path `/World/ground/terrain`.

## Config Location

- [source/QuadControl/QuadControl/tasks/direct/quadcontrol/quadcontrol_terrain_obstacle_env_cfg.py](source/QuadControl/QuadControl/tasks/direct/quadcontrol/quadcontrol_terrain_obstacle_env_cfg.py)

## Task Notes

- Increase `num_rows` or decrease `size[0]` to get denser waypoint spacing.
- Adjust `waypoint_hover_steps` to control the initial hover duration.
