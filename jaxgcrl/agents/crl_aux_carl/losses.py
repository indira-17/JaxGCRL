import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


def energy_fn(name, x, y):
    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    elif name == "dot":
        return jnp.sum(x * y, axis=-1)
    elif name == "cosine":
        return jnp.sum(x * y, axis=-1) / (
            jnp.linalg.norm(x, axis=-1) * jnp.linalg.norm(y, axis=-1) + 1e-6
        )
    elif name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    raise ValueError(f"Unknown energy function: {name}")


def contrastive_loss_fn(name, logits):
    if name == "fwd_infonce":
        return -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
    elif name == "bwd_infonce":
        return -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0))
    elif name == "sym_infonce":
        return -jnp.mean(
            2 * jnp.diag(logits)
            - jax.nn.logsumexp(logits, axis=1)
            - jax.nn.logsumexp(logits, axis=0)
        )
    elif name == "binary_nce":
        return -jnp.mean(jax.nn.sigmoid(logits))
    raise ValueError(f"Unknown contrastive loss function: {name}")


def _masked_mean(value, mask):
    mask = mask.astype(value.dtype)
    return jnp.sum(value * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def update_forward_dynamics(config, networks, transitions, training_state):
    """Fit F(s, A) -> g only on real replay transitions."""
    del config

    def loss_fn(fd_params):
        pred_goal = networks["forward_dynamics"].apply(
            fd_params,
            transitions.extras["state"],
            transitions.extras["action_sequence"],
        )
        target_goal = jax.lax.stop_gradient(transitions.extras["carl_goal"])
        valid = transitions.extras["carl_valid"]
        error = pred_goal - target_goal
        per_sample_loss = jnp.mean(error**2, axis=-1)
        return _masked_mean(per_sample_loss, valid), jnp.linalg.norm(error, axis=-1)

    (loss, error), grad = jax.value_and_grad(loss_fn, has_aux=True)(
        training_state.fd_state.params
    )
    training_state = training_state.replace(
        fd_state=training_state.fd_state.apply_gradients(grads=grad)
    )
    return training_state, {
        "fd_loss": loss,
        "fd_goal_error": jnp.mean(error),
        "fd_grad_norm": optax.global_norm(grad),
    }


def update_backward_dynamics(config, networks, transitions, training_state):
    """Fit q(A | s, g) only on real replay tuples."""
    del config

    def loss_fn(bd_params):
        mean, log_std = networks["backward_dynamics"].apply(
            bd_params,
            transitions.extras["state"],
            transitions.extras["carl_goal"],
        )
        target = jax.lax.stop_gradient(transitions.extras["action_sequence"])
        valid = transitions.extras["carl_valid"]
        inv_var = jnp.exp(-2.0 * log_std)
        nll = jnp.sum(0.5 * (target - mean) ** 2 * inv_var + log_std, axis=-1)
        mse = jnp.mean((target - mean) ** 2, axis=-1)
        return _masked_mean(nll, valid), _masked_mean(mse, valid)

    (loss, mse), grad = jax.value_and_grad(loss_fn, has_aux=True)(
        training_state.bd_state.params
    )
    training_state = training_state.replace(
        bd_state=training_state.bd_state.apply_gradients(grads=grad)
    )
    return training_state, {
        "bd_loss": loss,
        "bd_action_mse": mse,
        "bd_grad_norm": optax.global_norm(grad),
    }


def update_actor_and_alpha(config, networks, transitions, training_state, key):
    """Actor uses CARL features, but actor gradients do not update CARL."""

    def actor_loss(actor_params, sg_encoder_params, critic_params, log_alpha):
        state = transitions.observation[:, : config["state_size"]]
        local_goal = transitions.observation[:, config["state_size"] :]

        sg_repr = networks["sg_encoder"].apply(
            sg_encoder_params,
            jnp.concatenate([state, local_goal], axis=-1),
        )
        sg_repr = jax.lax.stop_gradient(sg_repr)
        observation = jnp.concatenate([state, local_goal, sg_repr], axis=-1)

        means, log_stds = networks["actor"].apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= 2 * (jnp.log(2.0) - x_ts - nn.softplus(-2.0 * x_ts))
        log_prob = log_prob.sum(-1)

        sa_repr = networks["sa_encoder"].apply(
            critic_params["sa_encoder"],
            jnp.concatenate([state, action], axis=-1),
        )
        g_repr = networks["g_encoder"].apply(critic_params["g_encoder"], local_goal)
        qf_pi = energy_fn(config["energy_fn"], sa_repr, g_repr)
        return jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi), log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        return jnp.mean(
            alpha * jax.lax.stop_gradient(-log_prob - config["target_entropy"])
        )

    (actor_loss_value, log_prob), actor_grad = jax.value_and_grad(
        actor_loss, argnums=0, has_aux=True
    )(
        training_state.actor_state.params,
        training_state.carl_state.params["sg_encoder"],
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
    )
    actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    alpha_loss_value, alpha_grad = jax.value_and_grad(alpha_loss)(
        training_state.alpha_state.params, log_prob
    )
    alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(actor_state=actor_state, alpha_state=alpha_state)
    return training_state, {
        "entropy": -log_prob,
        "actor_loss": actor_loss_value,
        "alpha_loss": alpha_loss_value,
        "log_alpha": alpha_state.params["log_alpha"],
        "actor_carl_sg_grad_norm": jnp.asarray(0.0, dtype=jnp.float32),
    }


def update_critic(config, networks, transitions, training_state, key):
    def critic_loss(critic_params):
        state = transitions.observation[:, : config["state_size"]]
        action = transitions.action
        goal = transitions.observation[:, config["state_size"] :]

        sa_repr = networks["sa_encoder"].apply(
            critic_params["sa_encoder"], jnp.concatenate([state, action], axis=-1)
        )
        g_repr = networks["g_encoder"].apply(critic_params["g_encoder"], goal)
        logits = energy_fn(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
        loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)

        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp**2)

        eye = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(eye, axis=1)
        logits_pos = jnp.sum(logits * eye) / jnp.sum(eye)
        logits_neg = jnp.sum(logits * (1 - eye)) / jnp.sum(1 - eye)
        return loss, (logsumexp, correct, logits_pos, logits_neg)

    (loss, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
        critic_loss, has_aux=True
    )(training_state.critic_state.params)
    training_state = training_state.replace(
        critic_state=training_state.critic_state.apply_gradients(grads=grad)
    )
    return training_state, {
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
        "critic_loss": loss,
    }


def update_carl_aux(config, networks, carl_pairs, training_state):
    """CARL on real anchors plus forward-validated generated pairs.

    Every row has a guaranteed real positive. A generated positive contributes
    only when F(s, A_gen) is close to g. A borrowed action contributes as a
    negative only when F(s, A_neg) is clearly far from g.
    """

    def carl_loss(carl_params):
        sg_repr = networks["sg_encoder"].apply(
            carl_params["sg_encoder"],
            jnp.concatenate([carl_pairs.state, carl_pairs.goal], axis=-1),
        )
        real_repr = networks["a_encoder"].apply(
            carl_params["a_encoder"], carl_pairs.real_action_sequence
        )
        generated_repr = networks["a_encoder"].apply(
            carl_params["a_encoder"], carl_pairs.generated_action_sequence
        )
        negative_repr = networks["a_encoder"].apply(
            carl_params["a_encoder"], carl_pairs.negative_action_sequence
        )

        real_score = energy_fn(config["energy_fn"], sg_repr, real_repr)
        generated_score = energy_fn(config["energy_fn"], sg_repr, generated_repr)
        negative_score = energy_fn(config["energy_fn"], sg_repr, negative_repr)

        # Real replay tuples are always the anchor. Generated positives are only
        # an auxiliary term, so model-generated data cannot dominate CARL.
        real_mask = carl_pairs.negative_mask & carl_pairs.valid_mask
        real_loss = _masked_mean(
            jax.nn.softplus(negative_score - real_score),
            real_mask,
        )
        generated_mask = carl_pairs.generated_positive_mask & carl_pairs.negative_mask
        generated_loss = _masked_mean(
            jax.nn.softplus(negative_score - generated_score),
            generated_mask,
        )
        loss = real_loss + config["carl_generated_weight"] * generated_loss
        return loss, (real_loss, generated_loss, real_score, generated_score, negative_score)

    (loss, aux), grad = jax.value_and_grad(carl_loss, has_aux=True)(
        training_state.carl_state.params
    )
    real_loss, generated_loss, real_score, generated_score, negative_score = aux
    training_state = training_state.replace(
        carl_state=training_state.carl_state.apply_gradients(grads=grad)
    )

    return training_state, {
        "carl_loss": loss,
        "carl_real_pair_loss": real_loss,
        "carl_generated_pair_loss": generated_loss,
        "carl_real_score": jnp.mean(real_score),
        "carl_generated_score": _masked_mean(
            generated_score, carl_pairs.generated_positive_mask
        ),
        "carl_negative_score": _masked_mean(negative_score, carl_pairs.negative_mask),
        "carl_generated_accept_fraction": jnp.mean(
            carl_pairs.generated_positive_mask.astype(jnp.float32)
        ),
        "carl_negative_accept_fraction": jnp.mean(
            carl_pairs.negative_mask.astype(jnp.float32)
        ),
        "carl_valid_fraction": jnp.mean(carl_pairs.valid_mask.astype(jnp.float32)),
        "carl_grad_norm": optax.global_norm(grad),
    }