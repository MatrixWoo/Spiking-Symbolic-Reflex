# MuJoCo Migration Plan

## Goal

Migrate the toy-demo Event → Concept → Rule / SNN → Reflex Action pipeline to a MuJoCo physics simulation, keeping the same interface so the core modules (event_detector, concept_layer, rule_layer, snn_reflex) can be reused with minimal changes.

## Minimum Viable Scene

- Robot arm with gripper (e.g., UR5e + Robotiq 2F-85)
- Cube object on table
- External force impulse as slip disturbance during transport

## Module Reuse

| Toy Demo Module | MuJoCo Equivalent |
|----------------|-------------------|
| `environment.py` | `mujoco_env.py` — wraps MuJoCo model, exposes same `step(action, reflex_action, grip_force_delta)` interface |
| `event_detector.py` | **reuse as-is** — reads state vector, outputs events |
| `concept_layer.py` | **reuse as-is** |
| `rule_layer.py` | **reuse as-is** |
| `snn_reflex.py` | **reuse as-is** |
| `visualization.py` | `mujoco_visualization.py` — render 3D + trajectory plots |

## Phases

1. **Minimal scene** — gripper + cube, no arm. Test slip + recovery.
2. **Add arm** — UR5e pick-and-place trajectory.
3. **Full pipeline** — Event/Concept/Rule/SNN with MuJoCo physics.
4. **Video export** — Side-by-side: 3D render + event/concept/reflex plots.
