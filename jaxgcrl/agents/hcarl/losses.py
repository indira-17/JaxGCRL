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
    """CARL representation loss + HIQL-style value loss with target V network."""

    def critic_loss(critic_params, target_critic_params, transitions, key):
        del key

        obs = transitions.observation
        state = obs[:, : config["state_size"]]
        low_goal = obs[:, config["state_size"] :]
        value_goal = transitions.extras["value_goal"]
        next_state = transitions.extras["next_state"]
        # Primitive action is used by the low actor.
        # CARL uses the k-step action sequence that actually led toward the sampled subgoal.
        action = transitions.action
        action_sequence = transitions.extras["action_sequence"]

        sg_encoder_params = critic_params["sg_encoder"]
        a_encoder_params = critic_params["a_encoder"]
        value_params = critic_params["value1"]
        target_value_params = target_critic_params["value1"]

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

        # HIQL value loss: V(s_t, phi(s_t, g)) with target network V_bar. g is the sampled final goal, not the low-level waypoint.
        z_curr = networks["sg_encoder"].apply(
            sg_encoder_params,
            jnp.concatenate([state, value_goal], axis=-1),
        )
        z_next = networks["sg_encoder"].apply(
            sg_encoder_params,
            jnp.concatenate([next_state, value_goal], axis=-1),
        )
        if config.get("stop_value_encoder_grad", True):
            z_curr = jax.lax.stop_gradient(z_curr)
            z_next = jax.lax.stop_gradient(z_next)

        v_curr = networks["value_module"].apply(value_params, state, z_curr).squeeze(-1)
        v_next = networks["value_module"].apply(target_value_params, next_state, z_next).squeeze(-1)

        # HIQL-style relabelled reward/mask are already produced by the
        # GCSDataset-style sampler in flatten_batch_hcrl:
        #   reward = success * reward_scale + reward_shift
        #   discount/mask = 1 - success if terminal=True, else 1
        target_v = (
            transitions.reward
            + transitions.discount
            * config.get("discount", 0.99)
            * jax.lax.stop_gradient(v_next)
        )

        diff = target_v - v_curr
        expectile = config.get("expectile", 0.7)
        value_weight = jnp.where(diff > 0.0, expectile, 1.0 - expectile)
        value_loss = jnp.mean(value_weight * diff**2)

        total_loss = carl_loss + config.get("value_loss_coeff", 1.0) * value_loss

        return total_loss, {
            "critic_loss": total_loss,
            "carl_loss": carl_loss,
            "value_loss": value_loss,
            "value_mean": jnp.mean(v_curr),
            "value_target_mean": jnp.mean(target_v),
            "value_adv_mean": jnp.mean(diff),
            "value_reward_mean": jnp.mean(transitions.reward),
            "value_discount_mean": jnp.mean(transitions.discount),
            "value_goal_success": jnp.mean(transitions.extras["hiql_value_goal_success"]),
            "action_seq_norm": jnp.mean(jnp.linalg.norm(action_sequence, axis=-1)),
            "target_value_mean": jnp.mean(v_next),
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logsumexp": jnp.mean(logsumexp),
        }

    (loss, metrics), grad = jax.value_and_grad(critic_loss, has_aux=True)(
        training_state.critic_state.params,
        training_state.target_critic_params,
        transitions,
        key,
    )
    del loss

    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    target_tau = config.get("target_update_rate", 0.005)
    new_target_critic_params = jax.tree_util.tree_map(
        lambda target, online: (1.0 - target_tau) * target + target_tau * online,
        training_state.target_critic_params,
        new_critic_state.params,
    )

    training_state = training_state.replace(
        critic_state=new_critic_state,
        target_critic_params=new_target_critic_params,
    )
    return training_state, metrics


def update_actor_and_alpha(config, networks, transitions, training_state, key):
    """Low actor AWR loss: weighted NLL on replay action, no SAC-Q update."""
    del key

    def actor_loss(actor_params, critic_params, transitions):
        obs = transitions.observation
        state = obs[:, : config["state_size"]]
        low_goal = obs[:, config["state_size"] :]
        next_state = transitions.extras["next_state"]
        action = transitions.action

        sg_encoder_params = critic_params["sg_encoder"]
        value_params = critic_params["value1"]

        z_curr = networks["sg_encoder"].apply(
            sg_encoder_params,
            jnp.concatenate([state, low_goal], axis=-1),
        )
        z_next = networks["sg_encoder"].apply(
            sg_encoder_params,
            jnp.concatenate([next_state, low_goal], axis=-1),
        )
        z_curr = jax.lax.stop_gradient(z_curr)
        z_next = jax.lax.stop_gradient(z_next)

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

        v_curr = networks["value_module"].apply(value_params, state, z_curr).squeeze(-1)
        v_next = networks["value_module"].apply(value_params, next_state, z_next).squeeze(-1)
        adv = v_next - v_curr

        beta = config.get("low_actor_beta", config.get("actor_beta", 1.0))
        max_weight = config.get("low_actor_max_weight", config.get("actor_max_weight", 20.0))
        weight = jnp.exp(beta * jax.lax.stop_gradient(adv))
        weight = jnp.clip(weight, 0.0, max_weight)

        loss = -jnp.mean(weight * log_prob)

        return loss, {
            "actor_loss": loss,
            "actor_log_prob": jnp.mean(log_prob),
            "entropy": -jnp.mean(log_prob),
            "actor_weight": jnp.mean(weight),
            "actor_adv": jnp.mean(adv),
            "actor_v_curr": jnp.mean(v_curr),
            "actor_v_next": jnp.mean(v_next),
            "actor_mse": jnp.mean((nn.tanh(mean) - action) ** 2),
            "actor_std": jnp.mean(std),
            "alpha_loss": jnp.array(0.0),
            "log_alpha": training_state.alpha_state.params["log_alpha"],
        }

    (loss, metrics), grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        transitions,
    )
    del loss
    training_state = training_state.replace(
        actor_state=training_state.actor_state.apply_gradients(grads=grad)
    )
    return training_state, metrics


def update_high_actor(config, networks, transitions, training_state, key):
    """High actor AWR loss: weighted NLL to latent target phi(s, s_{t+k})."""
    del key

    def high_actor_loss(high_actor_params, critic_params, transitions):
        state = transitions.extras["state"]
        final_goal = transitions.extras["high_actor_goal"]
        target_subgoal = transitions.extras["high_actor_target_goal"]
        target_subgoal_state = transitions.extras["high_actor_target_state"]

        sg_encoder_params = critic_params["sg_encoder"]
        value_params = critic_params["value1"]

        z_target = networks["sg_encoder"].apply(
            sg_encoder_params,
            jnp.concatenate([state, target_subgoal], axis=-1),
        )
        z_final_curr = networks["sg_encoder"].apply(
            sg_encoder_params,
            jnp.concatenate([state, final_goal], axis=-1),
        )
        z_final_next = networks["sg_encoder"].apply(
            sg_encoder_params,
            jnp.concatenate([target_subgoal_state, final_goal], axis=-1),
        )
        z_target = jax.lax.stop_gradient(z_target)
        z_final_curr = jax.lax.stop_gradient(z_final_curr)
        z_final_next = jax.lax.stop_gradient(z_final_next)

        mean, log_std = networks["high_actor"].apply(
            high_actor_params,
            jnp.concatenate([state, final_goal], axis=-1),
        )
        std = jnp.exp(log_std)
        log_prob = jax.scipy.stats.norm.logpdf(z_target, loc=mean, scale=std).sum(-1)

        v_curr = networks["value_module"].apply(value_params, state, z_final_curr).squeeze(-1)
        v_next = networks["value_module"].apply(value_params, target_subgoal_state, z_final_next).squeeze(-1)
        adv = v_next - v_curr

        beta = config.get("high_actor_beta", config.get("actor_beta", 1.0))
        max_weight = config.get("high_actor_max_weight", config.get("actor_max_weight", 20.0))
        weight = jnp.exp(beta * jax.lax.stop_gradient(adv))
        weight = jnp.clip(weight, 0.0, max_weight)

        loss = -jnp.mean(weight * log_prob)

        return loss, {
            "high_actor_loss": loss,
            "high_actor_log_prob": jnp.mean(log_prob),
            "high_actor_mse": jnp.mean((mean - z_target) ** 2),
            "high_actor_std": jnp.mean(std),
            "high_actor_weight": jnp.mean(weight),
            "high_actor_adv": jnp.mean(adv),
            "high_actor_v_curr": jnp.mean(v_curr),
            "high_actor_v_next": jnp.mean(v_next),
        }

    (loss, metrics), grad = jax.value_and_grad(high_actor_loss, has_aux=True)(
        training_state.high_actor_state.params,
        training_state.critic_state.params,
        transitions,
    )
    del loss
    training_state = training_state.replace(
        high_actor_state=training_state.high_actor_state.apply_gradients(grads=grad)
    )
    return training_state, metrics