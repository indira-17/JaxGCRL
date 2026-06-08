# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Soft Actor-Critic training.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

import functools
import logging
import time
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

import flax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import optax
import wandb
from brax import base, envs
from brax.io import model
from brax.training import gradients, pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.acme.types import NestedArray
from brax.training.agents.sac import losses as sac_losses
from brax.training.replay_buffers_test import jit_wrap
from brax.training.types import Params, Policy, PRNGKey
from flax.struct import dataclass
from flax.training import checkpoints

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import Evaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from . import networks

Metrics = types.Metrics
Env = Union[envs.Env, envs.Wrapper]
State = envs.State


class Transition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    next_observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray


def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
    """Collect data."""
    actions, policy_extras = policy(env_state.obs, key)
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = "i"


# ---------------------------------------------------------------------------
# InfoNCE loss (ported from OGBench)
# ---------------------------------------------------------------------------

def _info_nce_loss(queries, keys, temperature=0.1):
    """Symmetric InfoNCE loss. queries and keys are (batch, rep_dim).

    Both are L2-normalized before computing similarity.
    Uses optax.safe_norm to avoid NaN gradients when norm approaches zero.
    """
    q_norm = optax.safe_norm(queries, min_norm=1e-6, ord=2, axis=-1, keepdims=True)
    k_norm = optax.safe_norm(keys, min_norm=1e-6, ord=2, axis=-1, keepdims=True)
    queries = queries / q_norm
    keys = keys / k_norm
    sim = jnp.matmul(queries, keys.T) / temperature
    diag = jnp.diag(sim)
    loss_fwd = jnp.mean(-diag + jax.nn.logsumexp(sim, axis=-1))
    loss_rev = jnp.mean(-diag + jax.nn.logsumexp(sim, axis=0))
    return loss_fwd + loss_rev


# ---------------------------------------------------------------------------
# Custom losses for goal_rep actor path
# ---------------------------------------------------------------------------

def _make_losses_with_goal_rep(
    sac_network: networks.SACNetworks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
    state_dim: int,
    goal_dim: int,
    use_info_nce: bool,
    nce_temperature: float,
    infonce_weight: float,
):
    """Creates SAC losses where the actor uses goal_rep(state, goal) as input.

    The loss function signatures match brax's gradient_update_fn convention
    (first arg is the params to differentiate).

    For alpha and critic, the second arg is ``actor_params`` (a dict with
    keys "policy", "goal_rep", and optionally "actions_encoder") passed as
    non-differentiable context.
    For actor, the first arg IS actor_params (differentiated).
    """
    target_entropy = -0.5 * action_size
    policy_network = sac_network.policy_network
    q_network = sac_network.q_network
    parametric_action_distribution = sac_network.parametric_action_distribution
    goal_rep_network = sac_network.goal_rep_network
    actions_encoder_network = sac_network.actions_encoder_network

    def _goal_rep_actor_input(goal_rep_params, obs):
        state = obs[..., :state_dim]
        goal = obs[..., state_dim : state_dim + goal_dim]
        phi = goal_rep_network.apply(goal_rep_params, state, goal)
        return jnp.concatenate([state, phi], axis=-1)

    def alpha_loss(
        log_alpha: jnp.ndarray,
        actor_params,
        normalizer_params,
        transitions,
        key: PRNGKey,
    ):
        goal_rep_params = jax.lax.stop_gradient(actor_params["goal_rep"])
        policy_params = jax.lax.stop_gradient(actor_params["policy"])
        actor_input = _goal_rep_actor_input(goal_rep_params, transitions.observation)
        dist_params = policy_network.apply(normalizer_params, policy_params, actor_input)
        action = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        alpha = jnp.exp(log_alpha)
        # Adaptive H_target: add sg(L_geom-NCE) so alpha rises when rep is uncertain
        effective_target = target_entropy
        if use_info_nce:
            ae_params = jax.lax.stop_gradient(actor_params["actions_encoder"])
            infonce = transitions.extras["infonce"]
            h = goal_rep_network.apply(goal_rep_params, infonce["state_t"], infonce["goal_tk"])
            e = actions_encoder_network.apply(ae_params, infonce["actions_list"])
            # Normalized dot-product matrix — same computation as inside _info_nce_loss
            h_n = h / optax.safe_norm(h, min_norm=1e-6, ord=2, axis=-1, keepdims=True)
            e_n = e / optax.safe_norm(e, min_norm=1e-6, ord=2, axis=-1, keepdims=True)
            sim = jnp.matmul(h_n, e_n.T) / nce_temperature  # (B, B)
            diag = jnp.diag(sim)
            # NCE value to learn p(s,g)
            l_nce = (jnp.mean(-diag + jax.nn.logsumexp(sim, axis=-1))
                     + jnp.mean(-diag + jax.nn.logsumexp(sim, axis=0)))
            p = jnp.clip(jax.nn.sigmoid(1.0 - l_nce), 0.03, 0.2)
            # E[K] = (1-p)/p for Geom(p), clipped to [0, |H_target|] so effective_target stays <= 0 and alpha cannot diverge
            bonus = jnp.clip((1.0 - p) / p, 0.0, -target_entropy)
            effective_target = target_entropy + jax.lax.stop_gradient(bonus)
        return jnp.mean(alpha * jax.lax.stop_gradient(-log_prob - effective_target))

    def critic_loss(
        q_params,
        actor_params,
        normalizer_params,
        target_q_params,
        alpha,
        transitions,
        key: PRNGKey,
    ):
        goal_rep_params = jax.lax.stop_gradient(actor_params["goal_rep"])
        policy_params = jax.lax.stop_gradient(actor_params["policy"])

        # Q uses raw [s,g]; policy uses rep input [s,phi] only for action sampling
        q_old_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, transitions.action
        )
        next_actor_input = _goal_rep_actor_input(goal_rep_params, transitions.next_observation)
        next_dist_params = policy_network.apply(normalizer_params, policy_params, next_actor_input)
        next_action = parametric_action_distribution.sample_no_postprocessing(next_dist_params, key)
        next_log_prob = parametric_action_distribution.log_prob(next_dist_params, next_action)
        next_action = parametric_action_distribution.postprocess(next_action)
        next_q = q_network.apply(
            normalizer_params, target_q_params, transitions.next_observation, next_action
        )
        next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
        target_q = jax.lax.stop_gradient(
            transitions.reward * reward_scaling + transitions.discount * discounting * next_v
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)
        truncation = transitions.extras["state_extras"]["truncation"]
        q_error *= jnp.expand_dims(1 - truncation, -1)
        return 0.5 * jnp.mean(jnp.square(q_error))

    def actor_loss(
        actor_params,
        normalizer_params,
        q_params,
        alpha,
        transitions,
        key: PRNGKey,
    ):
        goal_rep_params = actor_params["goal_rep"]  # trainable: finetuned via SAC actor + NCE loss
        policy_params = actor_params["policy"]

        actor_input = _goal_rep_actor_input(goal_rep_params, transitions.observation)
        dist_params = policy_network.apply(normalizer_params, policy_params, actor_input)
        action = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)
        # Q uses raw obs [s,g], not rep space
        q_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, action
        )
        min_q = jnp.min(q_action, axis=-1)
        sac_loss = jnp.mean(alpha * log_prob - min_q)

        total_loss = sac_loss
        if use_info_nce:
            actions_encoder_params = actor_params["actions_encoder"]
            infonce = transitions.extras["infonce"]
            queries = goal_rep_network.apply(
                goal_rep_params, infonce["state_t"], infonce["goal_tk"]
            )
            keys = actions_encoder_network.apply(actions_encoder_params, infonce["actions_list"])
            nce_loss = _info_nce_loss(queries, keys, nce_temperature)
            total_loss = total_loss + infonce_weight * nce_loss
        return total_loss

    return alpha_loss, critic_loss, actor_loss


# ---------------------------------------------------------------------------
# Flatten batch (HER + optional InfoNCE data)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=["config", "env"])
def flatten_batch(config, env, transition: Transition, sample_key: PRNGKey) -> Transition:
    if config.use_her:
        # Find truncation indexes if present
        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
        single_trajectories = jnp.concatenate(
            [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
            axis=0,
        )

        # final_step_mask.shape == (seq_len, seq_len)
        final_step_mask = (
            is_future_mask * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
        )
        final_step_mask = jnp.logical_and(
            final_step_mask,
            transition.extras["state_extras"]["truncation"][None, :],
        )
        non_zero_columns = jnp.nonzero(final_step_mask, size=seq_len)[1]

        # If final state is not present use original goal (i.e. don't change anything)
        new_goals_idx = jnp.where(non_zero_columns == 0, arrangement, non_zero_columns)
        binary_mask = jnp.logical_and(non_zero_columns, non_zero_columns)

        new_goals = (
            binary_mask[:, None] * transition.observation[new_goals_idx][:, env.goal_indices]
            + jnp.logical_not(binary_mask)[:, None]
            * transition.observation[new_goals_idx][:, env.state_dim :]
        )

        # Transform observation
        state = transition.observation[:, : env.state_dim]
        new_obs = jnp.concatenate([state, new_goals], axis=1)

        # Recalculate reward
        dist = jnp.linalg.norm(new_obs[:, env.state_dim :] - new_obs[:, env.goal_indices], axis=1)
        new_reward = jnp.array(dist < env.goal_reach_thresh, dtype=float)

        # Transform next observation
        next_state = transition.next_observation[:, : env.state_dim]
        new_next_obs = jnp.concatenate([next_state, new_goals], axis=1)

        if config.use_info_nce:
            state_dim = env.state_dim
            action_dim = transition.action.shape[-1]
            nce_k = config.nce_k_step
            final_idx = new_goals_idx

            t_k_idx = jnp.minimum(arrangement + nce_k, final_idx)

            state_t = state  # (seq_len, state_dim)
            goal_tk = transition.observation[t_k_idx][:, env.goal_indices]  # (seq_len, goal_dim)

            action_indices = jnp.minimum(
                arrangement[:, None] + jnp.arange(nce_k), final_idx[:, None]
            )
            valid = (arrangement[:, None] + jnp.arange(nce_k)) <= final_idx[:, None]
            actions_at_k = transition.action[action_indices]
            actions_list = jnp.where(
                valid[:, :, None], actions_at_k, 0.0
            ).reshape(seq_len, nce_k * action_dim)

            new_extras = {
                **transition.extras,
                "infonce": {
                    "state_t": state_t,
                    "goal_tk": goal_tk,
                    "actions_list": actions_list,
                },
            }
            return transition._replace(
                observation=jnp.squeeze(new_obs),
                next_observation=jnp.squeeze(new_next_obs),
                reward=jnp.squeeze(new_reward),
                extras=new_extras,
            )

        return transition._replace(
            observation=jnp.squeeze(new_obs),
            next_observation=jnp.squeeze(new_next_obs),
            reward=jnp.squeeze(new_reward),
        )

    return transition


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: running_statistics.RunningStatisticsState
    goal_rep_params: Optional[Params] = None
    actions_encoder_params: Optional[Params] = None


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _load_goal_rep_from_rep_learn_checkpoint(path: str, goal_rep_params, actions_encoder_params=None):
    """Load GoalRep (and optionally ActionsEncoder) params from a rep_learn checkpoint.

    rep_learn saves RepLearnAgent params with keys 'GoalRep_0' and 'ActionsEncoder_0'.
    SAC stores goal_rep_params as the full Flax variable dict {'params': ...}.
    """
    ckpt = checkpoints.restore_checkpoint(ckpt_dir=path, target=None, prefix="rep_params_")
    if ckpt is None:
        raise FileNotFoundError(
            f"No rep_learn checkpoint found at '{path}' with prefix 'rep_params_'. "
            "Run rep_learn training first or check the path."
        )
    loaded_gr = flax.serialization.from_state_dict(goal_rep_params, {'params': ckpt['GoalRep_0']})
    loaded_ae = None
    if actions_encoder_params is not None and 'ActionsEncoder_0' in ckpt:
        loaded_ae = flax.serialization.from_state_dict(
            actions_encoder_params, {'params': ckpt['ActionsEncoder_0']}
        )
    return loaded_gr, loaded_ae


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    sac_network: networks.SACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    state_dim: int = 0,
    goal_dim: int = 0,
    action_size: int = 0,
    nce_k_step: int = 0,
    use_info_nce: bool = False,
    pretrained_goal_rep_path: str = "",
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q, key_gr, key_ae = jax.random.split(key, 4)
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = sac_network.policy_network.init(key_policy)
    q_params = sac_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    goal_rep_params = None
    actions_encoder_params = None

    use_goal_rep = sac_network.goal_rep_network is not None
    if use_goal_rep:
        dummy_state = jnp.zeros((1, state_dim))
        dummy_goal = jnp.zeros((1, goal_dim))
        goal_rep_params = sac_network.goal_rep_network.init(key_gr, dummy_state, dummy_goal)

        if pretrained_goal_rep_path:
            logging.info("Loading pretrained goal_rep from: %s", pretrained_goal_rep_path)
            goal_rep_params, _ = _load_goal_rep_from_rep_learn_checkpoint(
                pretrained_goal_rep_path, goal_rep_params
            )

        actor_params = {"policy": policy_params, "goal_rep": goal_rep_params}
        if use_info_nce and sac_network.actions_encoder_network is not None:
            dummy_actions_flat = jnp.zeros((1, nce_k_step * action_size))
            actions_encoder_params = sac_network.actions_encoder_network.init(key_ae, dummy_actions_flat)
            if pretrained_goal_rep_path:
                _, loaded_ae = _load_goal_rep_from_rep_learn_checkpoint(
                    pretrained_goal_rep_path, goal_rep_params, actions_encoder_params
                )
                if loaded_ae is not None:
                    actions_encoder_params = loaded_ae
            actor_params["actions_encoder"] = actions_encoder_params

        policy_optimizer_state = policy_optimizer.init(actor_params)
    else:
        policy_optimizer_state = policy_optimizer.init(policy_params)

    normalizer_params = running_statistics.init_state(specs.Array((obs_size,), jnp.dtype("float32")))

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        normalizer_params=normalizer_params,
        goal_rep_params=goal_rep_params,
        actions_encoder_params=actions_encoder_params,
    )
    return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])


# ---------------------------------------------------------------------------
# SAC agent
# ---------------------------------------------------------------------------

@dataclass
class SAC:
    """Soft Actor-Critic (SAC) agent."""

    learning_rate: float = 1e-4
    discounting: float = 0.9
    batch_size: int = 256
    normalize_observations: bool = False
    reward_scaling: float = 1.0
    tau: float = 0.005
    min_replay_size: int = 0
    max_replay_size: Optional[int] = 10000
    deterministic_eval: bool = False
    train_step_multiplier: int = 1
    unroll_length: int = 50
    h_dim: int = 256
    n_hidden: int = 2
    use_ln: bool = False
    use_her: bool = False
    # goal_rep as actor input (requires use_her)
    use_goal_rep_actor: bool = False
    use_info_nce: bool = False
    rep_dim: int = 10
    goal_rep_hidden: Tuple[int, ...] = (256, 256)
    nce_k_step: int = 25
    nce_temperature: float = 0.1
    infonce_weight: float = 0.5
    # Pretrained representation (used only for initialisation; rep is always trained end-to-end)
    pretrained_goal_rep_path: str = ""
    # State coverage visualization
    log_state_coverage: bool = False
    state_coverage_xy_dims: Tuple[int, int] = (0, 1)

    def train_fn(
        self,
        config,
        train_env: Env,
        eval_env: Optional[Env] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        process_id = jax.process_index()
        local_devices_to_use = jax.local_device_count()
        if config.max_devices_per_host is not None:
            local_devices_to_use = min(local_devices_to_use, config.max_devices_per_host)
        device_count = local_devices_to_use * jax.process_count()
        logging.info(
            "local_device_count: %s; total_device_count: %s",
            local_devices_to_use,
            device_count,
        )

        if self.use_goal_rep_actor and not self.use_her:
            raise ValueError("use_goal_rep_actor requires use_her=True")
        if self.use_info_nce and not self.use_goal_rep_actor:
            raise ValueError("use_info_nce requires use_goal_rep_actor=True")

        if self.min_replay_size >= config.total_env_steps:
            raise ValueError("No training will happen because min_replay_size >= total_env_steps")

        if self.max_replay_size is None:
            max_replay_size = config.total_env_steps
        else:
            max_replay_size = self.max_replay_size

        env_steps_per_actor_step = config.action_repeat * config.num_envs * self.unroll_length
        num_prefill_actor_steps = self.min_replay_size // self.unroll_length + 1
        logging.info("Num_prefill_actor_steps: %s", num_prefill_actor_steps)
        num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
        assert config.total_env_steps - self.min_replay_size >= 0
        num_evals_after_init = max(config.num_evals - 1, 1)
        num_training_steps_per_epoch = -(
            -(config.total_env_steps - num_prefill_env_steps)
            // (num_evals_after_init * env_steps_per_actor_step)
        )

        assert config.num_envs % device_count == 0
        env = train_env
        wrap_for_training = envs.training.wrap

        rng = jax.random.PRNGKey(config.seed)
        rng, key = jax.random.split(rng)
        v_randomization_fn = None
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(key, config.num_envs // jax.process_count() // local_devices_to_use),
            )
        env = TrajectoryIdWrapper(env)
        env = wrap_for_training(
            env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )
        unwrapped_env = train_env

        obs_size = env.observation_size
        action_size = env.action_size
        state_dim = getattr(env, "state_dim", obs_size)
        goal_dim = len(getattr(env, "goal_indices", []))
        if self.use_goal_rep_actor and goal_dim == 0:
            raise ValueError("use_goal_rep_actor requires env with goal_indices")

        def normalize_fn(x, y):
            return x

        if self.normalize_observations:
            normalize_fn = running_statistics.normalize

        # ---- Build networks ----
        if self.use_goal_rep_actor:
            sac_network = networks.make_sac_networks_with_goal_rep(
                state_dim=state_dim,
                goal_dim=goal_dim,
                action_size=action_size,
                rep_dim=self.rep_dim,
                goal_rep_hidden=self.goal_rep_hidden,
                use_info_nce=self.use_info_nce,
                preprocess_observations_fn=normalize_fn,
                hidden_layer_sizes=[self.h_dim] * self.n_hidden,
                layer_norm=self.use_ln,
            )
            make_policy = networks.make_goal_rep_inference_fn(sac_network, state_dim, goal_dim)
        else:
            sac_network = networks.make_sac_networks(
                observation_size=obs_size,
                action_size=action_size,
                preprocess_observations_fn=normalize_fn,
                layer_norm=self.use_ln,
                hidden_layer_sizes=[self.h_dim] * self.n_hidden,
            )
            make_policy = networks.make_inference_fn(sac_network)

        # ---- Optimizers ----
        alpha_optimizer = optax.adam(learning_rate=3e-4)
        policy_optimizer = optax.adam(learning_rate=self.learning_rate)
        q_optimizer = optax.adam(learning_rate=self.learning_rate)

        # ---- Replay buffer ----
        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))
        dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
            observation=dummy_obs,
            next_observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                },
                "policy_extras": {},
            },
        )
        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=max_replay_size // device_count,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size // device_count,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )

        # ---- Losses and gradient updates ----
        if self.use_goal_rep_actor:
            alpha_loss, critic_loss, actor_loss = _make_losses_with_goal_rep(
                sac_network=sac_network,
                reward_scaling=self.reward_scaling,
                discounting=self.discounting,
                action_size=action_size,
                state_dim=state_dim,
                goal_dim=goal_dim,
                use_info_nce=self.use_info_nce,
                nce_temperature=self.nce_temperature,
                infonce_weight=self.infonce_weight,
            )
        else:
            alpha_loss, critic_loss, actor_loss = sac_losses.make_losses(
                sac_network=sac_network,
                reward_scaling=self.reward_scaling,
                discounting=self.discounting,
                action_size=action_size,
            )

        alpha_update = gradients.gradient_update_fn(
            alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
        critic_update = gradients.gradient_update_fn(
            critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
        actor_update = gradients.gradient_update_fn(
            actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )

        # ---- Helper to pack/unpack actor params for goal_rep path ----
        _use_goal_rep = self.use_goal_rep_actor
        _use_nce = self.use_info_nce

        def _pack_actor_params(ts: TrainingState):
            if _use_goal_rep:
                d = {"policy": ts.policy_params, "goal_rep": ts.goal_rep_params}
                if _use_nce:
                    d["actions_encoder"] = ts.actions_encoder_params
                return d
            return ts.policy_params

        # ---- Update step ----
        def update_step(
            carry: Tuple[TrainingState, PRNGKey], transitions: Transition
        ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
            training_state, key = carry
            key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

            actor_params = _pack_actor_params(training_state)

            alpha_loss_val, alpha_params, alpha_optimizer_state = alpha_update(
                training_state.alpha_params,
                actor_params,
                training_state.normalizer_params,
                transitions,
                key_alpha,
                optimizer_state=training_state.alpha_optimizer_state,
            )
            alpha = jnp.exp(training_state.alpha_params)

            critic_loss_val, q_params, q_optimizer_state = critic_update(
                training_state.q_params,
                actor_params,
                training_state.normalizer_params,
                training_state.target_q_params,
                alpha,
                transitions,
                key_critic,
                optimizer_state=training_state.q_optimizer_state,
            )

            actor_loss_val, new_actor_params, policy_optimizer_state = actor_update(
                actor_params,
                training_state.normalizer_params,
                training_state.q_params,
                alpha,
                transitions,
                key_actor,
                optimizer_state=training_state.policy_optimizer_state,
            )

            # Unpack updated actor params
            if _use_goal_rep:
                new_policy_params = new_actor_params["policy"]
                new_goal_rep_params = new_actor_params["goal_rep"]
                if _use_nce:
                    new_actions_encoder_params = new_actor_params["actions_encoder"]
                else:
                    new_actions_encoder_params = training_state.actions_encoder_params
            else:
                new_policy_params = new_actor_params
                new_goal_rep_params = training_state.goal_rep_params
                new_actions_encoder_params = training_state.actions_encoder_params

            new_target_q_params = jax.tree_util.tree_map(
                lambda x, y: x * (1 - self.tau) + y * self.tau,
                training_state.target_q_params,
                q_params,
            )

            # Policy uses rep input [s,phi]; Q uses raw obs [s,g]
            _obs_state = transitions.observation[..., :state_dim]
            _obs_goal = transitions.observation[..., state_dim : state_dim + goal_dim]
            _obs_phi = sac_network.goal_rep_network.apply(new_goal_rep_params, _obs_state, _obs_goal)
            _obs_rep = jnp.concatenate([_obs_state, _obs_phi], axis=-1)
            # q_rep_actor_mean: Q(raw [s,g], π([s,φ])) — Q value of current policy's actions
            _actor_dist = sac_network.policy_network.apply(
                training_state.normalizer_params, new_actor_params["policy"], _obs_rep
            )
            _actor_actions = sac_network.parametric_action_distribution.mode(_actor_dist)
            rep_actor_q = sac_network.q_network.apply(
                training_state.normalizer_params, q_params, transitions.observation, _actor_actions
            )
            metrics = {
                "critic_loss": critic_loss_val,
                "actor_loss": actor_loss_val,
                "alpha_loss": alpha_loss_val,
                "alpha": jnp.exp(alpha_params),
                "q_rep_actor_mean": jnp.mean(rep_actor_q),
            }
            if _use_nce:
                infonce = transitions.extras["infonce"]
                queries = sac_network.goal_rep_network.apply(
                    new_goal_rep_params, infonce["state_t"], infonce["goal_tk"]
                )
                keys = sac_network.actions_encoder_network.apply(
                    new_actions_encoder_params, infonce["actions_list"]
                )
                metrics["nce_loss"] = _info_nce_loss(
                    queries, keys, self.nce_temperature
                )

            new_training_state = TrainingState(
                policy_optimizer_state=policy_optimizer_state,
                policy_params=new_policy_params,
                q_optimizer_state=q_optimizer_state,
                q_params=q_params,
                target_q_params=new_target_q_params,
                gradient_steps=training_state.gradient_steps + 1,
                env_steps=training_state.env_steps,
                alpha_optimizer_state=alpha_optimizer_state,
                alpha_params=alpha_params,
                normalizer_params=training_state.normalizer_params,
                goal_rep_params=new_goal_rep_params,
                actions_encoder_params=new_actions_encoder_params,
            )
            return (new_training_state, key), metrics

        # ---- Helper to extract policy params for inference ----
        def _get_policy_inference_params(ts: TrainingState):
            if _use_goal_rep:
                return (ts.normalizer_params, ts.policy_params, ts.goal_rep_params)
            return (ts.normalizer_params, ts.policy_params)

        # ---- Experience collection ----
        def get_experience(
            normalizer_params: running_statistics.RunningStatisticsState,
            policy_params,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ):
            policy = make_policy(policy_params)

            @jax.jit
            def f(carry, unused_t):
                env_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                env_state, transition = actor_step(
                    env,
                    env_state,
                    policy,
                    current_key,
                    extra_fields=("truncation", "traj_id"),
                )
                return (env_state, next_key), transition

            (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=self.unroll_length)

            normalizer_params = running_statistics.update(
                normalizer_params,
                jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
                ).observation,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )
            buffer_state = replay_buffer.insert(buffer_state, data)
            return normalizer_params, env_state, buffer_state

        def training_step(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, State, ReplayBufferState, Metrics]:
            experience_key, training_key = jax.random.split(key)
            policy_inf_params = _get_policy_inference_params(training_state)
            normalizer_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params,
                policy_inf_params,
                env_state,
                buffer_state,
                experience_key,
            )
            training_state = training_state.replace(
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            training_state, buffer_state, metrics = train_steps(training_state, buffer_state, training_key)
            return training_state, env_state, buffer_state, metrics

        def prefill_replay_buffer(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
            def f(carry, unused):
                del unused
                training_state, env_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                policy_inf_params = _get_policy_inference_params(training_state)
                new_normalizer_params, env_state, buffer_state = get_experience(
                    training_state.normalizer_params,
                    policy_inf_params,
                    env_state,
                    buffer_state,
                    key,
                )
                new_training_state = training_state.replace(
                    normalizer_params=new_normalizer_params,
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (new_training_state, env_state, buffer_state, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

        def train_steps(
            training_state: TrainingState,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
            experience_key, training_key, sampling_key = jax.random.split(key, 3)
            buffer_state, transitions = replay_buffer.sample(buffer_state)

            batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
            transitions = jax.vmap(flatten_batch, in_axes=(None, None, 0, 0))(
                self, env, transitions, batch_keys
            )

            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
                transitions,
            )
            permutation = jax.random.permutation(experience_key, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                transitions,
            )

            (training_state, _), metrics = jax.lax.scan(
                update_step, (training_state, training_key), transitions
            )
            return training_state, buffer_state, metrics

        def scan_train_steps(n, ts, bs, update_key):
            def body(carry, unsued_t):
                ts, bs, update_key = carry
                new_key, update_key = jax.random.split(update_key)
                ts, bs, metrics = train_steps(ts, bs, update_key)
                return (ts, bs, new_key), metrics

            return jax.lax.scan(body, (ts, bs, update_key), (), length=n)

        def training_epoch(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
            def f(carry, unused_t):
                ts, es, bs, k = carry
                k, new_key, update_key = jax.random.split(k, 3)
                ts, es, bs, metrics = training_step(ts, es, bs, k)
                (ts, bs, update_key), _ = scan_train_steps(self.train_step_multiplier - 1, ts, bs, update_key)
                return (ts, es, bs, new_key), metrics

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            return training_state, env_state, buffer_state, metrics

        training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

        def training_epoch_with_timing(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
            nonlocal training_walltime
            t = time.time()
            (training_state, env_state, buffer_state, metrics) = training_epoch(
                training_state, env_state, buffer_state, key
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            return (
                training_state,
                env_state,
                buffer_state,
                metrics,
            )  # pytype: disable=bad-return-type  # py311-upgrade

        global_key, local_key = jax.random.split(rng)
        local_key = jax.random.fold_in(local_key, process_id)

        # Training state init
        training_state = _init_training_state(
            key=global_key,
            obs_size=obs_size,
            local_devices_to_use=local_devices_to_use,
            sac_network=sac_network,
            alpha_optimizer=alpha_optimizer,
            policy_optimizer=policy_optimizer,
            q_optimizer=q_optimizer,
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_size=action_size,
            nce_k_step=self.nce_k_step,
            use_info_nce=self.use_info_nce,
            pretrained_goal_rep_path=self.pretrained_goal_rep_path,
        )
        del global_key

        local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

        # Env init
        env_keys = jax.random.split(env_key, config.num_envs // jax.process_count())
        env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
        env_state = jax.pmap(env.reset)(env_keys)

        # Replay buffer init
        buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use))

        if not eval_env:
            eval_env = env
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn, rng=jax.random.split(eval_key, config.num_eval_envs)
            )
        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = wrap_for_training(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )

        evaluator = Evaluator(
            eval_env,
            functools.partial(make_policy, deterministic=self.deterministic_eval),
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            key=eval_key,
        )

        # Run initial eval
        metrics = {}
        if process_id == 0 and config.num_evals > 1:
            metrics = evaluator.run_evaluation(
                _unpmap(_get_policy_inference_params(training_state)),
                training_metrics={},
            )
            progress_fn(
                0,
                metrics,
                make_policy,
                _unpmap(_get_policy_inference_params(training_state)),
                unwrapped_env,
            )

        # Create and initialize the replay buffer.
        t = time.time()
        prefill_key, local_key = jax.random.split(local_key)
        prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_keys
        )

        replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
        logging.info("replay size after prefill %s", replay_size)
        assert replay_size >= self.min_replay_size
        training_walltime = time.time() - t

        # State coverage tracking
        # If the env exposes goal_indices we use its first two entries (the agent's x,y in obs)
        _raw_goal_indices = getattr(unwrapped_env, "goal_indices", None)
        if self.log_state_coverage and _raw_goal_indices is not None and len(_raw_goal_indices) >= 2:
            _cov_xy0, _cov_xy1 = int(_raw_goal_indices[0]), int(_raw_goal_indices[1])
            logging.info(
                "State coverage: auto-detected XY dims from env.goal_indices -> (%d, %d)",
                _cov_xy0, _cov_xy1,
            )
        else:
            _cov_xy0, _cov_xy1 = self.state_coverage_xy_dims
            if self.log_state_coverage:
                logging.info(
                    "State coverage: using configured state_coverage_xy_dims -> (%d, %d)",
                    _cov_xy0, _cov_xy1,
                )
        _coverage_history = [] if self.log_state_coverage else None

        current_step = 0
        for eval_epoch_num in range(num_evals_after_init):
            logging.info("step %s", current_step)

            # Optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, buffer_state, training_metrics) = training_epoch_with_timing(
                training_state, env_state, buffer_state, epoch_keys
            )
            current_step = int(_unpmap(training_state.env_steps))

            # accumulate current env positions
            if _coverage_history is not None:
                # env_state.obs shape: (local_devices, num_envs_per_device, obs_size)
                obs_np = np.array(jax.device_get(env_state.obs)).reshape(-1, obs_size)
                _coverage_history.append(obs_np[:, [_cov_xy0, _cov_xy1]])

            # Eval and logging
            if process_id == 0:
                if config.checkpoint_logdir:
                    params = _unpmap(_get_policy_inference_params(training_state))
                    path = f"{config.checkpoint_logdir}_sac_{current_step}.pkl"
                    model.save_params(path, params)

                metrics = evaluator.run_evaluation(
                    _unpmap(_get_policy_inference_params(training_state)),
                    training_metrics,
                )
                do_render = (eval_epoch_num % config.visualization_interval) == 0
                if _coverage_history:
                    all_pos = np.concatenate(_coverage_history, axis=0)
                    bins = 50
                    h, xedges, yedges = np.histogram2d(
                        all_pos[:, 0], all_pos[:, 1], bins=bins
                    )
                    # Coverage entropy: higher = more uniform exploration
                    p = h / h.sum()
                    p_nz = p[p > 0]
                    coverage_entropy = float(-np.sum(p_nz * np.log(p_nz)))
                    occupied_cells = int(np.sum(h > 0))
                    # Log scalar metrics every eval epoch (not just at render intervals)
                    wandb.log({
                        "state_coverage_entropy": coverage_entropy,
                        "state_coverage_cells": occupied_cells,
                    }, step=current_step)
                    if do_render:
                        fig, ax = plt.subplots(figsize=(6, 6))
                        im = ax.imshow(
                            h.T, origin="lower", aspect="auto", cmap="hot",
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                        )
                        plt.colorbar(im, ax=ax, label="visit count")
                        ax.set_title(f"State Coverage (step {current_step}, H={coverage_entropy:.2f}, cells={occupied_cells}/{bins*bins})")
                        ax.set_xlabel(f"dim {_cov_xy0}")
                        ax.set_ylabel(f"dim {_cov_xy1}")
                        wandb.log({"state_coverage": wandb.Image(fig)}, step=current_step)
                        plt.close(fig)
                progress_fn(
                    current_step,
                    metrics,
                    make_policy,
                    _unpmap(_get_policy_inference_params(training_state)),
                    unwrapped_env,
                    do_render,
                )

        total_steps = current_step
        assert total_steps >= config.total_env_steps

        params = _unpmap(_get_policy_inference_params(training_state))

        pmap.assert_is_replicated(training_state)
        logging.info("total steps: %s", total_steps)
        pmap.synchronize_hosts()
        return make_policy, params, metrics
