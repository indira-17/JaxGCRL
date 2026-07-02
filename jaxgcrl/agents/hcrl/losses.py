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
    contrastive_loss_fn,
    energy_fn,
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


def _tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def update_carl_aux(config, networks, transitions, training_state):
    """Updates the auxiliary CARL encoders from replay only."""
    state = transitions.extras["state"]
    carl_goal = transitions.extras["carl_goal"]
    action_sequence = transitions.extras["action_sequence"]

    def carl_loss(carl_params):
        sg_repr = networks["sg_encoder"].apply(
            carl_params["sg_encoder"],
            jnp.concatenate([state, carl_goal], axis=-1),
        )
        a_repr = networks["a_encoder"].apply(carl_params["a_encoder"], action_sequence)

        logits = energy_fn(config["energy_fn"], sg_repr[:, None, :], a_repr[None, :, :])
        infonce_loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        logsumexp_penalty = jnp.mean(logsumexp**2)
        loss = infonce_loss + config["logsumexp_penalty_coeff"] * logsumexp_penalty

        eye = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.arange(logits.shape[0])
        logits_pos = jnp.sum(logits * eye) / jnp.sum(eye)
        logits_neg = jnp.sum(logits * (1.0 - eye)) / jnp.sum(1.0 - eye)

        return loss, {
            "carl_loss": loss,
            "carl_infonce_loss": infonce_loss,
            "carl_logsumexp_penalty": logsumexp_penalty,
            "carl_categorical_accuracy": jnp.mean(correct),
            "carl_logits_pos": logits_pos,
            "carl_logits_neg": logits_neg,
            "carl_logit_gap": logits_pos - logits_neg,
        }

    (_, metrics), grad = jax.value_and_grad(carl_loss, has_aux=True)(
        training_state.carl_state.params
    )
    training_state = training_state.replace(
        carl_state=training_state.carl_state.apply_gradients(grads=grad)
    )
    return training_state, metrics


def update_actor_and_alpha_carl(config, networks, transitions, training_state, key):
    """CRL actor loss with CARL latent actor conditioning."""

    def actor_loss(actor_params, sg_params, critic_params, log_alpha, transitions, key):
        obs = transitions.observation
        state = obs[:, : config["state_size"]]
        raw_goal = obs[:, config["state_size"] :]

        z = networks["sg_encoder"].apply(
            sg_params,
            jnp.concatenate([state, raw_goal], axis=-1),
        )
        if not config.get("carl_actor_encoder_grad", False):
            z = jax.lax.stop_gradient(z)

        actor_obs = jnp.concatenate([state, z], axis=-1)
        means, log_stds = networks["actor"].apply(actor_params, actor_obs)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= 2.0 * (jnp.log(2.0) - x_ts - nn.softplus(-2.0 * x_ts))
        log_prob = log_prob.sum(-1)

        sa_repr = networks["sa_encoder"].apply(
            critic_params["sa_encoder"],
            jnp.concatenate([state, action], axis=-1),
        )
        g_repr = networks["g_encoder"].apply(critic_params["g_encoder"], raw_goal)
        qf_pi = energy_fn(config["energy_fn"], sa_repr, g_repr)
        loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)
        return loss, log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        return jnp.mean(
            alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - config["target_entropy"]))
        )

    sg_params = training_state.carl_state.params["sg_encoder"]
    (actor_loss_value, log_prob), grad = jax.value_and_grad(actor_loss, argnums=(0, 1), has_aux=True)(
        training_state.actor_state.params,
        sg_params,
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
        transitions,
        key,
    )
    actor_grad, sg_grad = grad
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    encoder_grad_norm = jnp.where(
        config.get("carl_actor_encoder_grad", False),
        _tree_l2_norm(sg_grad),
        jnp.array(0.0),
    )
    if config.get("carl_actor_encoder_grad", False):
        zero_a_grad = jax.tree_util.tree_map(
            jnp.zeros_like,
            training_state.carl_state.params["a_encoder"],
        )
        old_a_params = training_state.carl_state.params["a_encoder"]
        carl_grad = {"sg_encoder": sg_grad, "a_encoder": zero_a_grad}
        new_carl_state = training_state.carl_state.apply_gradients(grads=carl_grad)
        new_carl_state = new_carl_state.replace(
            params={**new_carl_state.params, "a_encoder": old_a_params}
        )
    else:
        new_carl_state = training_state.carl_state

    alpha_loss_value, alpha_grad = jax.value_and_grad(alpha_loss)(
        training_state.alpha_state.params,
        log_prob,
    )
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(
        actor_state=new_actor_state,
        alpha_state=new_alpha_state,
        carl_state=new_carl_state,
    )
    return training_state, {
        "entropy": -log_prob,
        "actor_loss": actor_loss_value,
        "alpha_loss": alpha_loss_value,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
        "carl_actor_encoder_grad_norm": encoder_grad_norm,
    }


def update_high_actor_carl(config, networks, transitions, training_state, key):
    """High actor predicts latent CARL subgoals."""
    del key

    def high_actor_loss(high_actor_params, sg_params):
        state = transitions.extras["state"]
        final_goal = transitions.extras["high_actor_goal"]
        target_subgoal = transitions.extras["high_actor_target_goal"]

        z_target = networks["sg_encoder"].apply(
            sg_params,
            jnp.concatenate([state, target_subgoal], axis=-1),
        )
        z_target = jax.lax.stop_gradient(z_target)

        high_obs = jnp.concatenate([state, final_goal], axis=-1)
        mean, log_std = networks["high_actor"].apply(high_actor_params, high_obs)
        std = jnp.exp(log_std)
        log_prob = jax.scipy.stats.norm.logpdf(z_target, loc=mean, scale=std).sum(-1)

        loss = -jnp.mean(log_prob)
        mse = jnp.mean((mean - z_target) ** 2)
        std_mean = jnp.mean(std)
        return loss, (log_prob, mse, std_mean)

    (loss, (log_prob, mse, std_mean)), grads = jax.value_and_grad(
        high_actor_loss,
        has_aux=True,
    )(
        training_state.high_actor_state.params,
        training_state.carl_state.params["sg_encoder"],
    )
    training_state = training_state.replace(
        high_actor_state=training_state.high_actor_state.apply_gradients(grads=grads)
    )
    return training_state, {
        "high_actor_loss": loss,
        "high_actor_log_prob": jnp.mean(log_prob),
        "high_actor_mse": mse,
        "high_actor_std": std_mean,
    }
