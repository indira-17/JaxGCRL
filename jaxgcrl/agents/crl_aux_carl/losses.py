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


def update_actor_and_alpha(config, networks, transitions, training_state, key):
    """SAC/CRL actor update conditioned on [state, CARL phi(state, goal)].

    The CRL critic stays fixed during this actor update.  The CARL state-goal
    encoder is differentiated jointly with the actor, so it receives the actor
    gradient through the action sampled from pi(a | s, phi(s, g)).  The CARL
    action-sequence encoder remains untouched by this update and is trained
    only by update_carl_aux.
    """

    def actor_loss(actor_params, sg_encoder_params, critic_params, log_alpha, transitions, key):
        obs = transitions.observation
        state = obs[:, : config["state_size"]]
        future_state = transitions.extras["future_state"]
        goal = future_state[:, config["goal_indices"]]

        # The actor's goal input is the CARL state-goal representation.
        sg_repr = networks["sg_encoder"].apply(
            sg_encoder_params,
            jnp.concatenate([state, goal], axis=-1),
        )
        observation = jnp.concatenate([state, sg_repr], axis=-1)

        means, log_stds = networks["actor"].apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= 2 * (jnp.log(2.0) - x_ts - nn.softplus(-2.0 * x_ts))
        log_prob = log_prob.sum(-1)

        sa_encoder_params, g_encoder_params = (
            critic_params["sa_encoder"],
            critic_params["g_encoder"],
        )
        sa_repr = networks["sa_encoder"].apply(
            sa_encoder_params,
            jnp.concatenate([state, action], axis=-1),
        )
        g_repr = networks["g_encoder"].apply(g_encoder_params, goal)
        qf_pi = energy_fn(config["energy_fn"], sa_repr, g_repr)

        actor_loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)
        return actor_loss, log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - config["target_entropy"]))
        return jnp.mean(alpha_loss)

    (actor_loss, log_prob), (actor_grad, sg_grad) = jax.value_and_grad(
        actor_loss,
        argnums=(0, 1),
        has_aux=True,
    )(
        training_state.actor_state.params,
        training_state.carl_state.params["sg_encoder"],
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
        transitions,
        key,
    )
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    # Keep the existing independent CARL optimizer.
    actor_carl_grads = {
        "sg_encoder": sg_grad,
        "a_encoder": jax.tree_util.tree_map(
            jnp.zeros_like,
            training_state.carl_state.params["a_encoder"],
        ),
    }
    new_carl_state = training_state.carl_state.apply_gradients(grads=actor_carl_grads)

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(
        actor_state=new_actor_state,
        alpha_state=new_alpha_state,
        carl_state=new_carl_state,
    )

    metrics = {
        "entropy": -log_prob,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
        "actor_carl_sg_grad_norm": optax.global_norm(sg_grad),
    }

    return training_state, metrics


def update_critic(config, networks, transitions, training_state, key):
    def critic_loss(critic_params, transitions, key):
        sa_encoder_params, g_encoder_params = (
            critic_params["sa_encoder"],
            critic_params["g_encoder"],
        )

        state = transitions.observation[:, : config["state_size"]]
        action = transitions.action

        sa_repr = networks["sa_encoder"].apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
        g_repr = networks["g_encoder"].apply(
            g_encoder_params, transitions.observation[:, config["state_size"] :]
        )

        # InfoNCE
        logits = energy_fn(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
        critic_loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)

        # logsumexp regularisation
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        critic_loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp**2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg)

    (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
        critic_loss, has_aux=True
    )(training_state.critic_state.params, transitions, key)
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    metrics = {
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
        "critic_loss": loss,
    }

    return training_state, metrics


def update_carl_aux(config, networks, transitions, training_state):
    """Updates only the auxiliary CARL encoders."""

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

    (loss, metrics), grad = jax.value_and_grad(carl_loss, has_aux=True)(
        training_state.carl_state.params
    )
    del loss
    carl_state = training_state.carl_state.apply_gradients(grads=grad)
    return training_state.replace(carl_state=carl_state), metrics