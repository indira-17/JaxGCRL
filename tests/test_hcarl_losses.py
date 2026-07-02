import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from jaxgcrl.agents.hcarl.hcarl import Transition
from jaxgcrl.agents.hcarl.losses import (
    update_actor_and_alpha,
    update_critic,
    update_high_actor,
)


class _Encoder:
    def __init__(self, rep_dim=2):
        self.rep_dim = rep_dim

    def apply(self, params, x):
        return params["w"] * x[:, : self.rep_dim]


class _ActionEncoder:
    def __init__(self, rep_dim=2):
        self.rep_dim = rep_dim

    def apply(self, params, x):
        return params["w"] * x[:, : self.rep_dim]


class _Value:
    def apply(self, params, state, goal):
        return params["w"] * (
            jnp.sum(state, axis=-1, keepdims=True) + jnp.sum(goal, axis=-1, keepdims=True)
        )


class _RaiseValue:
    def apply(self, params, state, goal):
        del params, state, goal
        raise AssertionError("wrong value module was used")


class _Actor:
    def __init__(self, action_size):
        self.action_size = action_size

    def apply(self, params, obs):
        batch_size = obs.shape[0]
        mean = params["w"] * jnp.ones((batch_size, self.action_size), dtype=obs.dtype)
        log_std = -jnp.ones_like(mean)
        return mean, log_std


def _transition(batch_size=4, state_size=3, goal_size=2, action_size=2, subgoal_steps=2):
    state = jnp.arange(batch_size * state_size, dtype=jnp.float32).reshape(batch_size, state_size) / 10.0
    low_goal = jnp.ones((batch_size, goal_size), dtype=jnp.float32) * 0.25
    high_goal = jnp.ones((batch_size, goal_size), dtype=jnp.float32) * 0.75
    next_state = state + 0.1
    target_state = state + 0.2

    return Transition(
        observation=jnp.concatenate([state, low_goal], axis=-1),
        action=jnp.zeros((batch_size, action_size), dtype=jnp.float32),
        reward=jnp.zeros((batch_size,), dtype=jnp.float32),
        discount=jnp.ones((batch_size,), dtype=jnp.float32),
        extras={
            "next_state": next_state,
            "state": state,
            "future_state": target_state,
            "value_goal": high_goal,
            "low_value_goal": low_goal,
            "low_value_reward": jnp.zeros((batch_size,), dtype=jnp.float32),
            "low_value_discount": jnp.ones((batch_size,), dtype=jnp.float32),
            "low_actor_goal": low_goal,
            "low_actor_state": target_state,
            "action_sequence": jnp.ones(
                (batch_size, subgoal_steps * action_size), dtype=jnp.float32
            ),
            "high_actor_goal": high_goal,
            "high_value_goal": high_goal,
            "high_value_reward": jnp.ones((batch_size,), dtype=jnp.float32),
            "high_value_discount": jnp.ones((batch_size,), dtype=jnp.float32),
            "high_actor_target_goal": low_goal,
            "high_actor_target_state": target_state,
            "hiql_value_goal_success": jnp.zeros((batch_size,), dtype=jnp.float32),
            "low_value_goal_success": jnp.zeros((batch_size,), dtype=jnp.float32),
            "high_value_goal_success": jnp.zeros((batch_size,), dtype=jnp.float32),
        },
    )


def _training_state(action_size=2):
    critic_params = {
        "sg_encoder": {"w": jnp.array(0.5, dtype=jnp.float32)},
        "a_encoder": {"w": jnp.array(0.25, dtype=jnp.float32)},
        "value_low": {"w": jnp.array(0.1, dtype=jnp.float32)},
        "value_high": {"w": jnp.array(0.2, dtype=jnp.float32)},
    }
    return SimpleNamespace(
        critic_state=TrainState.create(
            apply_fn=None,
            params=critic_params,
            tx=optax.sgd(0.01),
        ),
        target_critic_params=critic_params,
        actor_state=TrainState.create(
            apply_fn=None,
            params={"w": jnp.array(0.0, dtype=jnp.float32)},
            tx=optax.sgd(0.01),
        ),
        high_actor_state=TrainState.create(
            apply_fn=None,
            params={"w": jnp.array(0.0, dtype=jnp.float32)},
            tx=optax.sgd(0.01),
        ),
        alpha_state=SimpleNamespace(params={"log_alpha": jnp.array(0.0, dtype=jnp.float32)}),
        replace=lambda **kwargs: None,
    )


def _with_replace(state):
    def replace(**kwargs):
        values = state.__dict__.copy()
        values.update(kwargs)
        new_state = SimpleNamespace(**values)
        return _with_replace(new_state)

    state.replace = replace
    return state


def _config():
    return {
        "state_size": 3,
        "goal_indices": (0, 1),
        "discount": 0.99,
        "target_update_rate": 0.005,
        "expectile": 0.7,
        "value_loss_coeff": 1.0,
        "energy_fn": "dot",
        "contrastive_loss_fn": "fwd_infonce",
        "logsumexp_penalty_coeff": 0.0,
        "low_actor_beta": 1.0,
        "low_actor_max_weight": 20.0,
        "high_actor_beta": 1.0,
        "high_actor_max_weight": 20.0,
    }


def test_update_critic_updates_split_values_and_reports_metrics():
    networks = {
        "sg_encoder": _Encoder(),
        "a_encoder": _ActionEncoder(),
        "value_low_module": _Value(),
        "value_high_module": _Value(),
    }
    state = _with_replace(_training_state())

    new_state, metrics = update_critic(
        _config(), networks, _transition(), state, jax.random.PRNGKey(0)
    )

    assert jnp.isfinite(metrics["value_low_loss"])
    assert jnp.isfinite(metrics["value_high_loss"])
    assert not jnp.allclose(
        new_state.critic_state.params["value_low"]["w"],
        state.critic_state.params["value_low"]["w"],
    )
    assert not jnp.allclose(
        new_state.critic_state.params["value_high"]["w"],
        state.critic_state.params["value_high"]["w"],
    )
    assert float(metrics["value_high_reward_mean"]) > 0.0


def test_low_actor_uses_low_value_not_high_value():
    networks = {
        "sg_encoder": _Encoder(),
        "actor": _Actor(action_size=2),
        "value_low_module": _Value(),
        "value_high_module": _RaiseValue(),
    }
    state = _with_replace(_training_state())

    _, metrics = update_actor_and_alpha(
        _config(), networks, _transition(), state, jax.random.PRNGKey(0)
    )

    assert jnp.isfinite(metrics["low_actor_adv"])


def test_high_actor_uses_high_value_not_low_value():
    networks = {
        "sg_encoder": _Encoder(),
        "high_actor": _Actor(action_size=2),
        "value_low_module": _RaiseValue(),
        "value_high_module": _Value(),
    }
    state = _with_replace(_training_state())

    _, metrics = update_high_actor(
        _config(), networks, _transition(), state, jax.random.PRNGKey(0)
    )

    assert jnp.isfinite(metrics["high_actor_adv"])
