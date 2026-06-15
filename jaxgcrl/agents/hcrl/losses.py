"""HCRL losses that reuse original CRL losses.

The important point is that the low-level actor and critic are not copied here.
They are imported from jaxgcrl.agents.crl.losses, so flat_policy=True uses the
same update functions as CRL.

This file only adds:
  - tanh/normal sampling helpers for HCRL rollout
  - update_high_actor for the hierarchical subgoal predictor
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

# Reuse the real CRL losses. Do not duplicate them in HCRL.
from jaxgcrl.agents.crl.losses import (  # noqa: F401
    update_actor_and_alpha,
    update_critic,
)


def _normal_sample(mean, log_std, key):
    std = jnp.exp(log_std)
    x = mean + std * jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
    log_prob = jax.scipy.stats.norm.logpdf(x, loc=mean, scale=std).sum(-1)
    return x, log_prob


def _tanh_normal_sample(mean, log_std, key):
    std = jnp.exp(log_std)
    x = mean + std * jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
    action = nn.tanh(x)
    log_prob = jax.scipy.stats.norm.logpdf(x, loc=mean, scale=std)
    log_prob -= 2.0 * (jnp.log(2.0) - x - nn.softplus(-2.0 * x))
    log_prob = log_prob.sum(-1)
    return action, log_prob


def update_high_actor(config, networks, transitions, training_state, key):
    """Supervised high-level waypoint predictor.

    This is the only new loss relative to CRL.

    Input:
        current state s_t
        final / hindsight goal g_j sampled with CRL future-goal sampling

    Target:
        k-step waypoint goal g_{min(t+k, j)}

    The low-level actor and CRL critic are not touched here.
    """
    del key, config

    def high_actor_loss(high_actor_params):
        state = transitions.extras["state"]
        final_goal = transitions.extras["high_actor_goal"]
        target_subgoal = transitions.extras["high_actor_target_goal"]

        high_obs = jnp.concatenate([state, final_goal], axis=-1)
        mean, log_std = networks["high_actor"].apply(high_actor_params, high_obs)
        std = jnp.exp(log_std)

        log_prob = jax.scipy.stats.norm.logpdf(
            target_subgoal,
            loc=mean,
            scale=std,
        ).sum(-1)

        loss = -jnp.mean(log_prob)
        mse = jnp.mean((mean - target_subgoal) ** 2)
        std_mean = jnp.mean(std)
        return loss, (log_prob, mse, std_mean)

    (loss, (log_prob, mse, std_mean)), grads = jax.value_and_grad(
        high_actor_loss,
        has_aux=True,
    )(training_state.high_actor_state.params)

    new_high_actor_state = training_state.high_actor_state.apply_gradients(grads=grads)
    training_state = training_state.replace(high_actor_state=new_high_actor_state)

    return training_state, {
        "high_actor_loss": loss,
        "high_actor_log_prob": jnp.mean(log_prob),
        "high_actor_mse": mse,
        "high_actor_std": std_mean,
    }