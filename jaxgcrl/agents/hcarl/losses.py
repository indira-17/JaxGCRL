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
    """Trains the one-step inverse and state-conditioned action-prior InfoNCE heads."""
    del key

    obs = transitions.observation
    state = obs[:, : config["state_size"]]
    action = transitions.action
    critic_params = training_state.critic_state.params

    def goal_action_loss_fn(rep_params):
        goal_repr = networks["sg_encoder"].apply(
            rep_params["sg_encoder"],
            jnp.concatenate([state, transitions.extras["carl_goal"]], axis=-1),
        )
        action_repr = networks["a_encoder"].apply(
            rep_params["a_encoder"],
            action,
        )
        
        logits = energy_fn(config["energy_fn"], goal_repr[:, None, :], action_repr[None, :, :])
        
        goal_action_loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)
        
        logsumexp = jax.nn.logsumexp(logits, axis=1)
        eye = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.arange(logits.shape[0])
        
        return goal_action_loss, {
            "goal_action_loss": goal_action_loss,
            "goal_action_categorical_accuracy": jnp.mean(correct),
            "goal_action_logits_pos": jnp.sum(logits * eye) / jnp.sum(eye),
            "goal_action_logits_neg": jnp.sum(logits * (1.0 - eye)) / jnp.sum(1.0 - eye),
            "goal_action_logsumexp": jnp.mean(logsumexp),
        }

    def action_prior_loss_fn(rep_params):
        state_repr = networks["s_encoder"].apply(rep_params["s_encoder"], state)
        action_repr = networks["a_encoder"].apply(rep_params["a_encoder"], action)
        
        logits = energy_fn(config["energy_fn"], state_repr[:, None, :], action_repr[None, :, :])
        
        action_prior_loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)
        
        logsumexp = jax.nn.logsumexp(logits, axis=1)
        eye = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.arange(logits.shape[0])
        
        return action_prior_loss, {
            "action_prior_loss": action_prior_loss,
            "action_prior_categorical_accuracy": jnp.mean(correct),
            "action_prior_logits_pos": jnp.sum(logits * eye) / jnp.sum(eye),
            "action_prior_logits_neg": jnp.sum(logits * (1.0 - eye)) / jnp.sum(1.0 - eye),
            "action_prior_logsumexp": jnp.mean(logsumexp),
        }

    def critic_loss_fn(rep_params):
        goal_action_loss, goal_action_metrics = goal_action_loss_fn(rep_params)
        action_prior_loss, action_prior_metrics = action_prior_loss_fn(rep_params)
        critic_loss = goal_action_loss + action_prior_loss
        metrics = {"critic_loss": critic_loss, "action_norm": jnp.mean(jnp.linalg.norm(action, axis=-1)), "goal_action_loss": goal_action_loss, "action_prior_loss": action_prior_loss}
        metrics.update(goal_action_metrics)
        metrics.update(action_prior_metrics)
        return critic_loss, metrics

    rep_params = {
        "sg_encoder": critic_params["sg_encoder"],
        "s_encoder": critic_params["s_encoder"],
        "a_encoder": critic_params["a_encoder"],
    }
    (critic_loss, metrics), rep_grad = jax.value_and_grad(critic_loss_fn, has_aux=True)(rep_params)
    grad = {
        "sg_encoder": rep_grad["sg_encoder"],
        "s_encoder": rep_grad["s_encoder"],
        "a_encoder": rep_grad["a_encoder"],
    }
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)
    return training_state, metrics


def update_actor_and_alpha(config, networks, transitions, training_state, key):
    """Updates the low actor by maximizing the one-step PMI reachability score."""
    del key

    def actor_loss(actor_params, critic_params, transitions):
        obs = transitions.observation
        state = obs[:, : config["state_size"]]
        action = transitions.action

        goal_repr = networks["sg_encoder"].apply(
            critic_params["sg_encoder"],
            jnp.concatenate([state, transitions.extras["low_actor_goal"]], axis=-1),
        )
        state_repr = networks["s_encoder"].apply(
            critic_params["s_encoder"], 
            state,
        )
        
        mean, log_std = networks["actor"].apply(
            actor_params, jnp.concatenate([state, goal_repr],
            axis=-1)
        )
        
        policy_action = nn.tanh(mean)
        policy_action_repr = networks["a_encoder"].apply(critic_params["a_encoder"], policy_action)
        replay_action_repr = networks["a_encoder"].apply(critic_params["a_encoder"], action)
        
        goal_action_score = energy_fn(config["energy_fn"], goal_repr, policy_action_repr)
        action_prior_score = energy_fn(config["energy_fn"], state_repr, policy_action_repr)
        
        goal_action_logits = energy_fn(config["energy_fn"], goal_repr[:, None, :], replay_action_repr[None, :, :])
        action_prior_logits = energy_fn(config["energy_fn"], state_repr[:, None, :], replay_action_repr[None, :, :])
        
        goal_action_log_z = jax.nn.logsumexp(goal_action_logits, axis=1) - jnp.log(jnp.asarray(action.shape[0], dtype=goal_action_logits.dtype))
        action_prior_log_z = jax.nn.logsumexp(action_prior_logits, axis=1) - jnp.log(jnp.asarray(action.shape[0], dtype=action_prior_logits.dtype))
        
        pmi_q = (goal_action_score - goal_action_log_z) - (action_prior_score - action_prior_log_z)
        loss = -jnp.mean(pmi_q)

        std = jnp.exp(log_std)
        clipped_action = jnp.clip(action, -1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh_action = jnp.arctanh(clipped_action)
        log_prob = jax.scipy.stats.norm.logpdf(pre_tanh_action, loc=mean, scale=std)
        log_prob -= 2.0 * (jnp.log(2.0) - pre_tanh_action - nn.softplus(-2.0 * pre_tanh_action))
        log_prob = log_prob.sum(-1)
        actor_sample_noise = jnp.mean(jnp.abs(action - policy_action))
        actor_action_abs_mean = jnp.mean(jnp.abs(policy_action))

        return loss, {
            "actor_loss": loss,
            "pmi_q": jnp.mean(pmi_q),
            "buffer_action_nll": -jnp.mean(log_prob),
            "actor_mse": jnp.mean((policy_action - action) ** 2),
            "actor_std": jnp.mean(std),
            "log_alpha": training_state.alpha_state.params["log_alpha"],
            "actor_sample_noise": actor_sample_noise,
            "actor_action_abs_mean": actor_action_abs_mean,
        }

    (loss, metrics), grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        transitions,
    )
    del loss
    training_state = training_state.replace(actor_state=training_state.actor_state.apply_gradients(grads=grad))
    return training_state, metrics

def update_high_actor(config, networks, transitions, training_state, key):
    """Trains the raw-coordinate high actor toward the sampled k-step waypoint."""
    del key

    def high_actor_loss(high_actor_params, transitions):
        state = transitions.extras["state"]
        final_goal = transitions.extras["high_actor_goal"]
        target_subgoal = transitions.extras["high_actor_target_goal"]

        mean, log_std = networks["high_actor"].apply(
            high_actor_params,
            jnp.concatenate([state, final_goal], axis=-1),
        )
        std = jnp.exp(log_std)
        log_prob = jax.scipy.stats.norm.logpdf(
            target_subgoal,
            loc=mean,
            scale=std,
        ).sum(-1)

        loss = -jnp.mean(log_prob)
        return loss, {
            "high_actor_loss": loss,
            "high_actor_log_prob": jnp.mean(log_prob),
            "high_actor_mse": jnp.mean((mean - target_subgoal) ** 2),
            "high_actor_std": jnp.mean(std),
        }

    (loss, metrics), grad = jax.value_and_grad(high_actor_loss, has_aux=True)(
        training_state.high_actor_state.params,
        transitions,
    )
    del loss
    training_state = training_state.replace(
        high_actor_state=training_state.high_actor_state.apply_gradients(grads=grad)
    )
    return training_state, metrics