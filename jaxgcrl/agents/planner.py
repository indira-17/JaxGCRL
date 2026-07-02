from typing import Literal, Tuple

import jax.numpy as jnp

PlannerMode = Literal["none", "ant_xy_oracle"]


def validate_planner_config(
    planner_mode: str,
    *,
    goal_indices: Tuple[int, ...],
    goal_size: int,
) -> None:
    if planner_mode == "none":
        return
    if planner_mode != "ant_xy_oracle":
        raise ValueError(f"Unknown planner_mode: {planner_mode}")
    if goal_size != 2 or tuple(goal_indices) != (0, 1):
        raise ValueError(
            "planner_mode='ant_xy_oracle' currently supports only Ant-style XY goals "
            f"with goal_indices=(0, 1), got goal_size={goal_size}, goal_indices={goal_indices}"
        )


def oracle_subgoal(
    state: jnp.ndarray,
    final_goal: jnp.ndarray,
    goal_indices: Tuple[int, ...],
    planner_mode: str,
    planner_step_size: float,
) -> jnp.ndarray:
    if planner_mode == "none":
        return final_goal
    if planner_mode != "ant_xy_oracle":
        raise ValueError(f"Unknown planner_mode: {planner_mode}")

    current_goal_state = state[:, jnp.asarray(goal_indices)]
    delta = final_goal - current_goal_state
    dist = jnp.linalg.norm(delta, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, planner_step_size / jnp.maximum(dist, 1e-6))
    return current_goal_state + scale * delta


def planner_metrics(
    state: jnp.ndarray,
    final_goal: jnp.ndarray,
    subgoal: jnp.ndarray,
    goal_indices: Tuple[int, ...],
    planner_step_size: float,
) -> dict:
    current_goal_state = state[:, jnp.asarray(goal_indices)]
    return {
        "planner/subgoal_dist": jnp.mean(jnp.linalg.norm(subgoal - current_goal_state, axis=-1)),
        "planner/final_goal_dist": jnp.mean(jnp.linalg.norm(final_goal - current_goal_state, axis=-1)),
        "planner/step_size": jnp.asarray(planner_step_size, dtype=jnp.float32),
    }
