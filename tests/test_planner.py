import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from jaxgcrl.agents.planner import oracle_subgoal, validate_planner_config


def test_ant_xy_oracle_points_toward_goal():
    state = jnp.array([[0.0, 0.0, 3.0]])
    final_goal = jnp.array([[3.0, 4.0]])

    subgoal = oracle_subgoal(state, final_goal, (0, 1), "ant_xy_oracle", 2.0)

    assert jnp.allclose(subgoal, jnp.array([[1.2, 1.6]]), atol=1e-6)


def test_ant_xy_oracle_clips_to_step_size():
    state = jnp.array([[1.0, 2.0]])
    final_goal = jnp.array([[11.0, 2.0]])

    subgoal = oracle_subgoal(state, final_goal, (0, 1), "ant_xy_oracle", 2.5)

    assert jnp.allclose(subgoal, jnp.array([[3.5, 2.0]]), atol=1e-6)


def test_ant_xy_oracle_zero_distance_is_finite():
    state = jnp.array([[1.0, 2.0]])
    final_goal = jnp.array([[1.0, 2.0]])

    subgoal = oracle_subgoal(state, final_goal, (0, 1), "ant_xy_oracle", 2.5)

    assert jnp.all(jnp.isfinite(subgoal))
    assert jnp.allclose(subgoal, final_goal)


def test_ant_xy_oracle_rejects_non_ant_xy_config():
    with pytest.raises(ValueError, match="Ant-style XY"):
        validate_planner_config("ant_xy_oracle", goal_indices=(2, 3), goal_size=2)
