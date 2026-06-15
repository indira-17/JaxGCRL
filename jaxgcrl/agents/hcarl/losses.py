import flax.linen as nn
import jax
import jax.numpy as jnp

import flax.linen as nn
import jax
import jax.numpy as jnp

# Reuse the real CRL losses.
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


def energy_fn(name, x, y):
    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    elif name == "dot":
        return jnp.sum(x * y, axis=-1)
    elif name == "cosine":
        return jnp.sum(x * y, axis=-1) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-6)
    elif name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    else:
        raise ValueError(f"Unknown energy function: {name}")


def contrastive_loss_fn(name, logits):
    if name == "fwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
    elif name == "bwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0))
    elif name == "sym_infonce":
        critic_loss = -jnp.mean(
            2 * jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1) - jax.nn.logsumexp(logits, axis=0)
        )
    elif name == "binary_nce":
        critic_loss = -jnp.mean(jax.nn.sigmoid(logits))
    else:
        raise ValueError(f"Unknown contrastive loss function: {name}")
    return critic_loss

def make_positive_negative_actions_from_batch(config, transitions, key):
    obs = transitions.observation
    state = obs[:, : config["state_size"]]

    future_state = transitions.extras["future_state"]
    goal = future_state[:, config["goal_indices"]]

    pos_action = transitions.action

    batch_size = state.shape[0]

    # Use a random circular shift so neg_action_i != pos_action_i.
    shift = jax.random.randint(key, shape=(), minval=1, maxval=batch_size)
    neg_idx = (jnp.arange(batch_size) + shift) % batch_size

    neg_action = pos_action[neg_idx]

    return {
        "state": state,
        "goal": goal,
        "pos_action": pos_action,
        "neg_action": neg_action,
    }


def update_actor_and_alpha(config, networks, transitions, training_state, key):
    def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
        obs = transitions.observation  # expected_shape = self.batch_size, obs_size + goal_size
        state = obs[:, : config["state_size"]]
        future_state = transitions.extras["future_state"]
        goal = future_state[:, config["goal_indices"]]
        observation = jnp.concatenate([state, goal], axis=1)

        means, log_stds = networks["actor"].apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= 2 * (jnp.log(2.0) - x_ts - nn.softplus(-2.0 * x_ts))
        log_prob = log_prob.sum(-1)  # dimension = B

        sg_encoder_params, a_encoder_params = (
            critic_params["sg_encoder"],
            critic_params["a_encoder"],
        )
        sg_repr = networks["sg_encoder"].apply(sg_encoder_params, jnp.concatenate([state, goal], axis=-1))
        a_repr = networks["a_encoder"].apply(a_encoder_params, action)

        qf_pi = energy_fn(config["energy_fn"], sg_repr, a_repr)

        actor_loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)

        return actor_loss, log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - config["target_entropy"]))
        return jnp.mean(alpha_loss)

    (actor_loss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
        transitions,
        key,
    )
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

    metrics = {
        "entropy": -log_prob,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
    }

    return training_state, metrics

# InfoNCE loss on (s, g, a) pairs as in original CRL.
# def update_critic(config, networks, transitions, training_state, key):
#     def critic_loss(critic_params, transitions, key):
#         del key

#         sg_encoder_params, a_encoder_params = (critic_params["sg_encoder"], critic_params["a_encoder"])

#         obs = transitions.observation
#         state = obs[:, : config["state_size"]]
#         future_state = transitions.extras["future_state"]
#         goal = future_state[:, config["goal_indices"]]
#         action = transitions.action

#         sg_repr = networks["sg_encoder"].apply(sg_encoder_params, jnp.concatenate([state, goal], axis=-1))
#         a_repr = networks["a_encoder"].apply(a_encoder_params, action)

#         logits = energy_fn(config["energy_fn"], sg_repr[:, None, :], a_repr[None, :, :])

#         critic_loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)

#         logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
#         critic_loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp**2)

#         I = jnp.eye(logits.shape[0])
#         correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
#         logits_pos = jnp.sum(logits * I) / jnp.sum(I)
#         logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

#         return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg)

#     (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
#         critic_loss, has_aux=True
#     )(training_state.critic_state.params, transitions, key)

#     new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
#     training_state = training_state.replace(critic_state=new_critic_state)

#     metrics = {
#         "categorical_accuracy": jnp.mean(correct),
#         "logits_pos": logits_pos,
#         "logits_neg": logits_neg,
#         "logsumexp": logsumexp.mean(),
#         "critic_loss": loss,
#     }

#     return training_state, metrics

# Multi Positive InfoNCE
# def update_critic(config, networks, transitions, training_state, key):
#     def critic_loss(critic_params, transitions, key):
#         del key

#         sg_encoder_params, a_encoder_params = (critic_params["sg_encoder"], critic_params["a_encoder"])

#         obs = transitions.observation
#         state = obs[:, : config["state_size"]]
#         future_state = transitions.extras["future_state"]
#         goal = future_state[:, config["goal_indices"]]
#         action = transitions.action

#         sg_repr = networks["sg_encoder"].apply(sg_encoder_params, jnp.concatenate([state, goal], axis=-1))
#         a_repr = networks["a_encoder"].apply(a_encoder_params, action)

#         # InfoNCE / contrastive loss.
#         # logits[i, j] = score between state-goal i and action j
#         logits = energy_fn(config["energy_fn"],
#             sg_repr[:, None, :],
#             a_repr[None, :, :],
#         )

#         batch_size = logits.shape[0]

#         goal_dist = jnp.linalg.norm(goal[:, None, :] - goal[None, :, :], axis=-1)

#         eps = config.get("positive_goal_eps", 0.5)

#         pos_mask = goal_dist < eps
#         pos_mask = jnp.logical_or(pos_mask, jnp.eye(batch_size, dtype=bool))
#         neg_mask = jnp.logical_not(pos_mask)
#         pos_logits = jnp.where(pos_mask, logits, -1e9)

#         log_num = jax.nn.logsumexp(pos_logits, axis=1)
#         log_den = jax.nn.logsumexp(logits, axis=1)

#         critic_loss = -jnp.mean(log_num - log_den)

#         # for logging
#         avg_logits_pos = jnp.sum(jnp.where(pos_mask, logits, 0.0)) / jnp.maximum(jnp.sum(pos_mask), 1)
#         avg_logits_neg = jnp.sum(jnp.where(neg_mask, logits, 0.0)) / jnp.maximum(jnp.sum(neg_mask), 1)

#         return critic_loss, (log_num, log_den, avg_logits_pos, avg_logits_neg)

#     (loss, (log_num, log_den, avg_logits_pos, avg_logits_neg)), grad = jax.value_and_grad(
#         critic_loss, has_aux=True
#     )(training_state.critic_state.params, transitions, key)

#     new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
#     training_state = training_state.replace(critic_state=new_critic_state)

#     metrics = {
#         "log_num": jnp.mean(log_num),
#         "log_den": jnp.mean(log_den),
#         "avg_logits_pos": avg_logits_pos,
#         "avg_logits_neg": avg_logits_neg,
#         "critic_loss": loss,
#     }

#     return training_state, metrics

# # NLL Loss on original CRL
# def update_high_actor(config, networks, transitions, training_state, key):
#     """Supervised high-level waypoint predictor.

#     This is the only new loss relative to CRL.

#     Input:
#         current state s_t
#         final / hindsight goal g_j sampled with CRL future-goal sampling

#     Target:
#         k-step waypoint goal g_{min(t+k, j)}

#     The low-level actor and CRL critic are not touched here.
#     """
#     del key, config

#     def high_actor_loss(high_actor_params):
#         state = transitions.extras["state"]
#         final_goal = transitions.extras["high_actor_goal"]
#         target_subgoal = transitions.extras["high_actor_target_goal"]

#         high_obs = jnp.concatenate([state, final_goal], axis=-1)
#         mean, log_std = networks["high_actor"].apply(high_actor_params, high_obs)
#         std = jnp.exp(log_std)

#         log_prob = jax.scipy.stats.norm.logpdf(
#             target_subgoal,
#             loc=mean,
#             scale=std,
#         ).sum(-1)

#         loss = -jnp.mean(log_prob)
#         mse = jnp.mean((mean - target_subgoal) ** 2)
#         std_mean = jnp.mean(std)
#         return loss, (log_prob, mse, std_mean)

#     (loss, (log_prob, mse, std_mean)), grads = jax.value_and_grad(
#         high_actor_loss,
#         has_aux=True,
#     )(training_state.high_actor_state.params)

#     new_high_actor_state = training_state.high_actor_state.apply_gradients(grads=grads)
#     training_state = training_state.replace(high_actor_state=new_high_actor_state)

#     return training_state, {
#         "high_actor_loss": loss,
#         "high_actor_log_prob": jnp.mean(log_prob),
#         "high_actor_mse": mse,
#         "high_actor_std": std_mean,
#     }

# policy gradient version of the high-level actor loss.
def update_high_actor(config, networks, transitions, training_state, key):
    """CARL-advantage-weighted high-level waypoint predictor.

    High actor:
        pi_H(z | s, g_final)

    Weight:
        delta = Q_CARL(s, z, a) - Q_CARL(s, g_final, a)

    Loss:
        L_H = - exp(beta * stopgrad(delta)) * log pi_H(z | s, g_final)
    """
    del key

    beta = config.get("high_actor_beta", 1.0)
    max_weight = config.get("high_actor_max_weight", 20.0)

    def high_actor_loss(high_actor_params, critic_params):
        state = transitions.extras["state"]
        final_goal = transitions.extras["high_actor_goal"]
        target_subgoal = transitions.extras["high_actor_target_goal"]
        action = transitions.action

        # log pi_H(z | s, g_final)
        high_obs = jnp.concatenate([state, final_goal], axis=-1)
        mean, log_std = networks["high_actor"].apply(high_actor_params, high_obs)
        std = jnp.exp(log_std)
        log_prob = jax.scipy.stats.norm.logpdf( target_subgoal, loc=mean, scale=std).sum(-1)

        # Q(s, z, a) - Q(s, g_final, a)
        sg_encoder_params = critic_params["sg_encoder"]
        a_encoder_params = critic_params["a_encoder"]

        sg_sub_repr = networks["sg_encoder"].apply(sg_encoder_params,jnp.concatenate([state, target_subgoal], axis=-1))
        sg_final_repr = networks["sg_encoder"].apply(sg_encoder_params, jnp.concatenate([state, final_goal], axis=-1))
        a_repr = networks["a_encoder"].apply(a_encoder_params, action)

        q_sub = energy_fn(config["energy_fn"], sg_sub_repr, a_repr)
        q_final = energy_fn(config["energy_fn"], sg_final_repr, a_repr)

        delta = q_sub - q_final

        # HIQL-style advantage weighting
        weight = jnp.exp(beta * jax.lax.stop_gradient(delta))
        weight = jnp.clip(weight, 0.0, max_weight)

        loss = -jnp.mean(weight * log_prob)

        mse = jnp.mean((mean - target_subgoal) ** 2)
        std_mean = jnp.mean(std)

        return loss, (log_prob, mse, std_mean, delta, weight, q_sub, q_final)

    (loss, (log_prob, mse, std_mean, delta, weight, q_sub, q_final)), grads = (
        jax.value_and_grad(high_actor_loss, has_aux=True)
        (
            training_state.high_actor_state.params,
            training_state.critic_state.params,
        )
    )

    new_high_actor_state = training_state.high_actor_state.apply_gradients(grads=grads)

    training_state = training_state.replace(high_actor_state=new_high_actor_state)

    return training_state, {
        "high_actor_loss": loss,
        "high_actor_log_prob": jnp.mean(log_prob),
        "high_actor_mse": mse,
        "high_actor_std": std_mean,
        "high_actor_delta": jnp.mean(delta),
        "high_actor_weight": jnp.mean(weight),
        "high_actor_q_sub": jnp.mean(q_sub),
        "high_actor_q_final": jnp.mean(q_final),
    }