import flax.linen as nn
import jax
import jax.numpy as jnp


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
    if name == "dot":
        return jnp.sum(x * y, axis=-1)
    if name == "cosine":
        x_norm = jnp.linalg.norm(x, axis=-1)
        y_norm = jnp.linalg.norm(y, axis=-1)
        return jnp.sum(x * y, axis=-1) / (x_norm * y_norm + 1e-6)
    if name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    raise ValueError(f"Unknown energy function: {name}")


def contrastive_loss_fn(name, logits):
    if name == "fwd_infonce":
        return -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
    if name == "bwd_infonce":
        return -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0))
    if name == "sym_infonce":
        return -jnp.mean(
            2.0 * jnp.diag(logits)
            - jax.nn.logsumexp(logits, axis=1)
            - jax.nn.logsumexp(logits, axis=0)
        )
    if name == "binary_nce":
        eye = jnp.eye(logits.shape[0])
        pos_loss = -jnp.sum(jax.nn.log_sigmoid(logits) * eye) / jnp.sum(eye)
        neg_loss = -jnp.sum(jax.nn.log_sigmoid(-logits) * (1.0 - eye)) / jnp.sum(1.0 - eye)
        return pos_loss + neg_loss
    raise ValueError(f"Unknown contrastive loss function: {name}")


def update_critic(config, networks, transitions, training_state, key):
    """CARL representation update plus split low/high HIQL-style values."""
    del key

    obs = transitions.observation
    state = obs[:, : config["state_size"]]
    low_goal = obs[:, config["state_size"] :]
    low_value_goal = transitions.extras["low_value_goal"]
    high_value_goal = transitions.extras["high_value_goal"]
    next_state = transitions.extras["next_state"]
    action_sequence = transitions.extras["action_sequence"]

    critic_params = training_state.critic_state.params
    rep_params = {
        "sg_encoder": critic_params["sg_encoder"],
        "a_encoder": critic_params["a_encoder"],
    }
    value_low_params = critic_params["value_low"]
    value_high_params = critic_params["value_high"]
    target_value_low_params = training_state.target_critic_params["value_low"]
    target_value_high_params = training_state.target_critic_params["value_high"]

    def carl_loss_fn(rep_params):
        sg_encoder_params = rep_params["sg_encoder"]
        a_encoder_params = rep_params["a_encoder"]

        # CARL representation loss: InfoNCE on (state, low_goal, action_sequence)
        sg_repr = networks["sg_encoder"].apply(
            sg_encoder_params,
            jnp.concatenate([state, low_goal], axis=-1),
        )
        a_repr = networks["a_encoder"].apply(a_encoder_params, action_sequence)
        logits = energy_fn(
            config["energy_fn"],
            sg_repr[:, None, :],
            a_repr[None, :, :],
        )

        carl_loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        carl_loss += config.get("logsumexp_penalty_coeff", 0.0) * jnp.mean(logsumexp**2)

        eye = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.arange(logits.shape[0])
        logits_pos = jnp.sum(logits * eye) / jnp.sum(eye)
        logits_neg = jnp.sum(logits * (1.0 - eye)) / jnp.sum(1.0 - eye)

        return carl_loss, {
            "carl_loss": carl_loss,
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logsumexp": jnp.mean(logsumexp),
            "action_seq_norm": jnp.mean(jnp.linalg.norm(action_sequence, axis=-1)),
        }

    def low_value_loss_fn(value_params, rep_params, target_value_params):
        del rep_params

        # Low-level value is conditioned on the raw local/planner subgoal coordinates,
        # not on CARL phi(s, g). This keeps CARL actor conditioning separate from
        # the HIQL-style value target.
        raw_goal = low_value_goal
        v_curr = networks["value_low_module"].apply(value_params, state, raw_goal).squeeze(-1)
        v_next = networks["value_low_module"].apply(target_value_params, next_state, raw_goal).squeeze(-1)

        target_v = (
            transitions.extras["low_value_reward"]
            + transitions.extras["low_value_discount"]
            * config.get("discount", 0.99)
            * jax.lax.stop_gradient(v_next)
        )

        diff = target_v - v_curr
        expectile = config.get("expectile", 0.7)
        value_weight = jnp.where(diff > 0.0, expectile, 1.0 - expectile)
        value_loss = jnp.mean(value_weight * diff**2)
        value_update_loss = config.get("value_loss_coeff", 1.0) * value_loss

        return value_update_loss, {
            "value_low_loss": value_loss,
            "value_low_mean": jnp.mean(v_curr),
            "value_low_target_mean": jnp.mean(target_v),
            "value_low_adv_mean": jnp.mean(diff),
            "value_low_reward_mean": jnp.mean(transitions.extras["low_value_reward"]),
            "value_low_goal_success": jnp.mean(transitions.extras["low_value_goal_success"]),
            "target_value_low_mean": jnp.mean(v_next),
        }

    def high_value_loss_fn(value_params, target_value_params):
        # High-level value receives the raw final/long-horizon goal coordinates.
        v_curr = networks["value_high_module"].apply(value_params, state, high_value_goal).squeeze(-1)
        v_next = networks["value_high_module"].apply(
            target_value_params, next_state, high_value_goal
        ).squeeze(-1)

        target_v = (
            transitions.extras["high_value_reward"]
            + transitions.extras["high_value_discount"]
            * config.get("discount", 0.99)
            * jax.lax.stop_gradient(v_next)
        )

        diff = target_v - v_curr
        expectile = config.get("expectile", 0.7)
        value_weight = jnp.where(diff > 0.0, expectile, 1.0 - expectile)
        value_loss = jnp.mean(value_weight * diff**2)
        value_update_loss = config.get("value_loss_coeff", 1.0) * value_loss

        return value_update_loss, {
            "value_high_loss": value_loss,
            "value_high_mean": jnp.mean(v_curr),
            "value_high_target_mean": jnp.mean(target_v),
            "value_high_adv_mean": jnp.mean(diff),
            "value_high_reward_mean": jnp.mean(transitions.extras["high_value_reward"]),
            "value_high_goal_success": jnp.mean(transitions.extras["high_value_goal_success"]),
            "target_value_high_mean": jnp.mean(v_next),
        }

    (carl_loss, carl_metrics), rep_grad = jax.value_and_grad(carl_loss_fn, has_aux=True)(rep_params)
    (value_low_update_loss, value_low_metrics), value_low_grad = jax.value_and_grad(
        low_value_loss_fn, has_aux=True
    )(
        value_low_params,
        rep_params,
        target_value_low_params,
    )
    # The high value is only used to weight the learned high-actor AWR loss.
    # With a planner, that actor is absent, so do not train or log this unused
    # high-level value branch.
    train_high_value = config.get("train_high_value", True)
    if train_high_value:
        (value_high_update_loss, value_high_metrics), value_high_grad = jax.value_and_grad(
            high_value_loss_fn, has_aux=True
        )(
            value_high_params,
            target_value_high_params,
        )
    else:
        value_high_update_loss = jnp.array(0.0, dtype=value_low_update_loss.dtype)
        value_high_metrics = {}
        value_high_grad = jax.tree_util.tree_map(jnp.zeros_like, value_high_params)

    grad = {
        "sg_encoder": rep_grad["sg_encoder"],
        "a_encoder": rep_grad["a_encoder"],
        "value_low": value_low_grad,
        "value_high": value_high_grad,
    }
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)

    target_tau = config.get("target_update_rate", 0.005)
    new_target_critic_params = dict(training_state.target_critic_params)
    value_names = ("value_low", "value_high") if train_high_value else ("value_low",)
    for value_name in value_names:
        new_target_critic_params[value_name] = jax.tree_util.tree_map(
            lambda target, online: (1.0 - target_tau) * target + target_tau * online,
            training_state.target_critic_params[value_name],
            new_critic_state.params[value_name],
        )

    training_state = training_state.replace(
        critic_state=new_critic_state,
        target_critic_params=new_target_critic_params,
    )

    metrics = {"critic_loss": carl_loss + value_low_update_loss + value_high_update_loss}
    metrics.update(carl_metrics)
    metrics.update(value_low_metrics)
    if train_high_value:
        metrics.update(value_high_metrics)
    return training_state, metrics

def update_actor_and_alpha(config, networks, transitions, training_state, key):
    """Low AWR update; return the state-goal-encoder gradient from the actor loss."""
    del key

    def actor_loss(actor_params, sg_encoder_params, value_params, transitions):
        obs = transitions.observation
        state = obs[:, : config["state_size"]]
        low_goal = obs[:, config["state_size"] :]
        next_state = transitions.extras["next_state"]
        action = transitions.action

        # Keep the encoder differentiable along the actor-conditioning path:
        # phi(s, g) -> low actor -> AWR NLL.  The AWR weight remains fixed.
        z_curr = networks["sg_encoder"].apply(
            sg_encoder_params,
            jnp.concatenate([state, low_goal], axis=-1),
        )

        mean, log_std = networks["actor"].apply(
            actor_params,
            jnp.concatenate([state, z_curr], axis=-1),
        )
        std = jnp.exp(log_std)
        clipped_action = jnp.clip(action, -1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh_action = jnp.arctanh(clipped_action)
        log_prob = jax.scipy.stats.norm.logpdf(
            pre_tanh_action,
            loc=mean,
            scale=std,
        )
        log_prob -= 2.0 * (
            jnp.log(2.0) - pre_tanh_action - nn.softplus(-2.0 * pre_tanh_action)
        )
        log_prob = log_prob.sum(-1)

        # The advantage defines an AWR weight only. The low value is conditioned on the raw local/planner subgoal coordinates, not on CARL phi(s, g).
        raw_goal_for_value = low_goal
        v_curr = networks["value_low_module"].apply(
            value_params, state, raw_goal_for_value
        ).squeeze(-1)
        v_next = networks["value_low_module"].apply(
            value_params, next_state, raw_goal_for_value
        ).squeeze(-1)
        adv = v_next - v_curr

        beta = config.get("low_actor_beta", config.get("actor_beta", 1.0))
        max_weight = config.get(
            "low_actor_max_weight", config.get("actor_max_weight", 20.0)
        )
        weight = jnp.exp(beta * jax.lax.stop_gradient(adv))
        weight = jnp.clip(weight, 0.0, max_weight)

        loss = -jnp.mean(weight * log_prob)

        mean_action = nn.tanh(mean)
        actor_sample_noise = jnp.mean(jnp.abs(action - mean_action))
        actor_action_abs_mean = jnp.mean(jnp.abs(action))

        return loss, {
            "actor_loss": loss,
            "actor_log_prob": jnp.mean(log_prob),
            "entropy": -jnp.mean(log_prob),
            "actor_weight": jnp.mean(weight),
            "actor_adv": jnp.mean(adv),
            "low_actor_adv": jnp.mean(adv),
            "actor_v_curr": jnp.mean(v_curr),
            "actor_v_next": jnp.mean(v_next),
            "actor_mse": jnp.mean((nn.tanh(mean) - action) ** 2),
            "actor_std": jnp.mean(std),
            "alpha_loss": jnp.array(0.0),
            "log_alpha": training_state.alpha_state.params["log_alpha"],
            "actor_sample_noise": actor_sample_noise,
            "actor_action_abs_mean": actor_action_abs_mean,
        }

    (loss, metrics), (actor_grad, sg_actor_grad) = jax.value_and_grad(
        actor_loss,
        argnums=(0, 1),
        has_aux=True,
    )(
        training_state.actor_state.params,
        training_state.critic_state.params["sg_encoder"],
        training_state.critic_state.params["value_low"],
        transitions,
    )
    del loss
    training_state = training_state.replace(
        actor_state=training_state.actor_state.apply_gradients(grads=actor_grad)
    )
    return training_state, metrics, sg_actor_grad


def update_high_actor(config, networks, transitions, training_state, key):
    """High AWR update; return the state-goal-encoder gradient from its latent NLL."""
    del key

    def high_actor_loss(high_actor_params, sg_encoder_params, value_params, transitions):
        state = transitions.extras["state"]
        final_goal = transitions.extras["high_actor_goal"]
        target_subgoal = transitions.extras["high_actor_target_goal"]
        target_subgoal_state = transitions.extras["high_actor_target_state"]

        # The latent target is CARL's phi(s, s_{t+k}); keep it differentiable so
        # the high AWR loss co-trains the state-goal encoder.
        z_target = networks["sg_encoder"].apply(
            sg_encoder_params,
            jnp.concatenate([state, target_subgoal], axis=-1),
        )

        mean, log_std = networks["high_actor"].apply(
            high_actor_params,
            jnp.concatenate([state, final_goal], axis=-1),
        )
        std = jnp.exp(log_std)
        log_prob = jax.scipy.stats.norm.logpdf(
            z_target, loc=mean, scale=std
        ).sum(-1)

        # The high AWR weight is fixed with respect to the high actor and CARL.
        v_curr = networks["value_high_module"].apply(
            value_params, state, final_goal
        ).squeeze(-1)
        v_next = networks["value_high_module"].apply(
            value_params, target_subgoal_state, final_goal
        ).squeeze(-1)
        adv = v_next - v_curr

        beta = config.get("high_actor_beta", config.get("actor_beta", 1.0))
        max_weight = config.get(
            "high_actor_max_weight", config.get("actor_max_weight", 20.0)
        )
        weight = jnp.exp(beta * jax.lax.stop_gradient(adv))
        weight = jnp.clip(weight, 0.0, max_weight)

        loss = -jnp.mean(weight * log_prob)
        high_latent_noise_mean = jnp.mean(std)
        high_log_std_mean = jnp.mean(log_std)

        return loss, {
            "high_actor_loss": loss,
            "high_actor_log_prob": jnp.mean(log_prob),
            "high_actor_mse": jnp.mean((mean - z_target) ** 2),
            "high_actor_std": jnp.mean(std),
            "high_actor_weight": jnp.mean(weight),
            "high_actor_adv": jnp.mean(adv),
            "high_actor_v_curr": jnp.mean(v_curr),
            "high_actor_v_next": jnp.mean(v_next),
            "high_actor_log_std_mean": high_log_std_mean,
            "high_actor_latent_noise_mean": high_latent_noise_mean,
        }

    (loss, metrics), (high_actor_grad, sg_high_actor_grad) = jax.value_and_grad(
        high_actor_loss,
        argnums=(0, 1),
        has_aux=True,
    )(
        training_state.high_actor_state.params,
        training_state.critic_state.params["sg_encoder"],
        training_state.critic_state.params["value_high"],
        transitions,
    )
    del loss
    training_state = training_state.replace(
        high_actor_state=training_state.high_actor_state.apply_gradients(
            grads=high_actor_grad
        )
    )
    return training_state, metrics, sg_high_actor_grad