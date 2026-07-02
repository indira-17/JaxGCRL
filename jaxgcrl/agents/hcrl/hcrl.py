"""Online HCRL implemented as original CRL plus one high actor.

Design goal:
  - flat_policy=True: use the original CRL components and losses.
  - flat_policy=False: add only a high actor that predicts raw subgoals.

Low-level learning is not copied/reimplemented here. The low actor and critic
updates are imported from jaxgcrl.agents.crl.losses through .losses. The CRL
networks Actor, SAEncoder, and Encoder are imported from
jaxgcrl.agents.crl.networks. This avoids accidental drift from CRL.
"""

import logging
import pickle
import random
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from brax import base, envs
from brax.training import types
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.agents.planner import PlannerMode, oracle_subgoal, planner_metrics, validate_planner_config
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

# Reuse original CRL networks. Do not redefine these in HCRL.
from jaxgcrl.agents.crl.networks import Actor, Encoder

from .losses import (
    _normal_sample,
    _tanh_normal_sample,
    update_actor_and_alpha_carl,
    update_carl_aux,
    update_actor_and_alpha,
    update_critic,
    update_high_actor,
    update_high_actor_carl,
)
from .networks import HighActor

Metrics = types.Metrics
Env = Union[envs.Env, envs.Wrapper]
State = envs.State


@dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
    high_actor_state: TrainState
    carl_state: TrainState


class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: Any = ()


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


def _pack_params(training_state: TrainingState):
    return {
        "actor": training_state.actor_state.params,
        "sa_encoder": training_state.critic_state.params["sa_encoder"],
        "g_encoder": training_state.critic_state.params["g_encoder"],
        "high_actor": training_state.high_actor_state.params,
        "sg_encoder": training_state.carl_state.params["sg_encoder"],
        "a_encoder": training_state.carl_state.params["a_encoder"],
    }


@dataclass
class HCRL:
    """Hierarchical CRL agent with one high actor and original CRL low actor + critic."""
    subgoal_steps: int = 25
    rep_dim: int = 10
    high_actor_hidden: Tuple[int, ...] = (512, 512, 512)
    flat_policy: bool = False
    planner_mode: PlannerMode = "none"
    planner_step_size: float = 2.0
    disable_high_actor_update_with_planner: bool = False
    use_carl_actor: bool = False
    carl_lr: float = 3e-4
    carl_repr_dim: int = 64
    carl_actor_encoder_grad: bool = False

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    # gamma
    discount: float = 0.99

    # forward CRL logsumexp penalty
    logsumexp_penalty_coeff: float = 0.1

    train_step_multiplier: int = 1

    disable_entropy_actor: bool = False

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    recent_replay_fraction: float = 0.0
    recent_replay_window: int = 0
    unroll_length: int = 62
    h_dim: int = 256
    n_hidden: int = 2
    skip_connections: int = 4
    use_relu: bool = False

    # phi(s,a) and psi(g) repr dimension
    repr_dim: int = 64

    # layer norm
    use_ln: bool = False

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    # State coverage visualization
    log_state_coverage: bool = False
    state_coverage_xy_dims: Tuple[int, int] = (0, 1)

    def check_config(self, config):
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )
        if not 0.0 <= self.recent_replay_fraction <= 1.0:
            raise ValueError(
                f"recent_replay_fraction must be in [0, 1], got {self.recent_replay_fraction}"
            )
        if self.recent_replay_window < 0:
            raise ValueError(
                f"recent_replay_window must be non-negative, got {self.recent_replay_window}"
            )

    def train_fn(
        self,
        config,
        train_env: Env,
        eval_env: Optional[Env] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[..., None] = lambda *args, **kwargs: None,
    ):
        del randomization_fn
        self.check_config(config)

        logging.info("HCRL flat_policy: %s", self.flat_policy)
        logging.info("HCRL use_carl_actor: %s", self.use_carl_actor)
        logging.info("HCRL planner_mode: %s", self.planner_mode)
        logging.info("HCRL log_state_coverage: %s", self.log_state_coverage)

        unwrapped_env = train_env
        train_env = TrajectoryIdWrapper(train_env)
        train_env = envs.training.wrap(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )
        if eval_env is None:
            eval_env = unwrapped_env
        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = envs.training.wrap(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = int(np.ceil(self.min_replay_size / self.unroll_length))
        num_training_steps_per_epoch = int(
            np.ceil(
                (config.total_env_steps - num_prefill_env_steps)
                / (config.num_evals * env_steps_per_actor_step)
            )
        )
        assert num_training_steps_per_epoch > 0, "total_env_steps too small for this setup"

        logging.info("num_prefill_actor_steps: %d", num_prefill_actor_steps)
        logging.info("num_training_steps_per_epoch: %d", num_training_steps_per_epoch)

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key = jax.random.split(key, 4)

        # Keep CRL component keys together. The high key is split after those so
        # adding hierarchy perturbs CRL initialization as little as possible.
        key, sa_key, g_key, actor_key = jax.random.split(key, 4)
        key, high_key, sg_key, a_key = jax.random.split(key, 4)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size
        validate_planner_config(
            self.planner_mode,
            goal_indices=tuple(np.asarray(train_env.goal_indices)),
            goal_size=goal_size,
        )
        target_entropy = -0.5 * action_size

        # State coverage tracking.
        _raw_goal_indices = getattr(unwrapped_env, "goal_indices", None)
        if self.log_state_coverage and _raw_goal_indices is not None and len(_raw_goal_indices) >= 2:
            _cov_xy0, _cov_xy1 = int(_raw_goal_indices[0]), int(_raw_goal_indices[1])
            logging.info(
                "State coverage: auto-detected XY dims from env.goal_indices -> (%d, %d)",
                _cov_xy0,
                _cov_xy1,
            )
        else:
            _cov_xy0, _cov_xy1 = self.state_coverage_xy_dims
            if self.log_state_coverage:
                logging.info(
                    "State coverage: using configured state_coverage_xy_dims -> (%d, %d)",
                    _cov_xy0,
                    _cov_xy1,
                )
        _coverage_history = [] if self.log_state_coverage else None

        # Original CRL networks plus high actor.
        sa_encoder_module = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        g_encoder_module = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        actor_module = Actor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        actor_input_size = state_size + (self.carl_repr_dim if self.use_carl_actor else goal_size)
        high_actor_output_size = self.carl_repr_dim if self.use_carl_actor else goal_size
        high_actor_module = Actor(
            action_size=high_actor_output_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        sg_encoder_module = Encoder(
            repr_dim=self.carl_repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        a_encoder_module = Encoder(
            repr_dim=self.carl_repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        networks = {
            "sa_encoder": sa_encoder_module,
            "g_encoder": g_encoder_module,
            "actor": actor_module,
            "high_actor": high_actor_module,
            "sg_encoder": sg_encoder_module,
            "a_encoder": a_encoder_module,
        }

        dummy_state = jnp.ones((1, state_size))
        dummy_goal = jnp.ones((1, goal_size))
        dummy_action = jnp.ones((1, action_size))
        dummy_actor_obs = jnp.ones((1, actor_input_size))
        dummy_sa = jnp.concatenate([dummy_state, dummy_action], axis=-1)
        dummy_high_obs = jnp.concatenate([dummy_state, dummy_goal], axis=-1)
        dummy_sg_obs = jnp.concatenate([dummy_state, dummy_goal], axis=-1)
        dummy_action_sequence = jnp.ones((1, self.subgoal_steps * action_size))

        sa_params = sa_encoder_module.init(sa_key, dummy_sa)
        g_params = g_encoder_module.init(g_key, dummy_goal)
        actor_params = actor_module.init(actor_key, dummy_actor_obs)
        high_params = high_actor_module.init(high_key, dummy_high_obs)
        sg_params = sg_encoder_module.init(sg_key, dummy_sg_obs)
        a_params = a_encoder_module.init(a_key, dummy_action_sequence)

        critic_state = TrainState.create(
            apply_fn=None,
            params={"sa_encoder": sa_params, "g_encoder": g_params},
            tx=optax.adam(learning_rate=self.critic_lr),
        )
        actor_state = TrainState.create(
            apply_fn=actor_module.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=self.policy_lr),
        )
        high_actor_state = TrainState.create(
            apply_fn=high_actor_module.apply,
            params=high_params,
            tx=optax.adam(learning_rate=self.policy_lr),
        )
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": jnp.array(0.0)},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )
        carl_state = TrainState.create(
            apply_fn=None,
            params={"sg_encoder": sg_params, "a_encoder": a_params},
            tx=optax.adam(learning_rate=self.carl_lr),
        )
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
            high_actor_state=high_actor_state,
            carl_state=carl_state,
        )

        dummy_transition = Transition(
            observation=jnp.zeros((obs_size,)),
            action=jnp.zeros((action_size,)),
            reward=0.0,
            discount=0.0,
            extras={"state_extras": {"truncation": 0.0, "traj_id": 0.0}},
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)
        _use_recent_replay = self.recent_replay_fraction > 0.0 and self.recent_replay_window > 0
        _recent_replay_fraction = float(self.recent_replay_fraction)
        _recent_replay_window = int(self.recent_replay_window)

        def sample_replay_buffer(buffer_state):
            if not _use_recent_replay:
                return replay_buffer.sample(buffer_state)

            key, env_key, full_key, recent_key, mix_key = jax.random.split(buffer_state.key, 5)
            envs_idxs = jax.random.choice(
                env_key,
                jnp.arange(replay_buffer.num_envs),
                shape=(replay_buffer.num_envs,),
                replace=False,
            )
            max_start = buffer_state.insert_position - replay_buffer.episode_length
            min_start = buffer_state.sample_position
            recent_min_start = jnp.maximum(min_start, max_start - _recent_replay_window)
            full_starts = jax.random.randint(
                full_key,
                shape=(replay_buffer.num_envs,),
                minval=min_start,
                maxval=max_start,
            )
            recent_starts = jax.random.randint(
                recent_key,
                shape=(replay_buffer.num_envs,),
                minval=recent_min_start,
                maxval=max_start,
            )
            use_recent = (
                jax.random.uniform(mix_key, shape=(replay_buffer.num_envs,))
                < _recent_replay_fraction
            )
            start_values = jnp.where(use_recent, recent_starts, full_starts)
            matrix = start_values[:, jnp.newaxis] + jnp.arange(replay_buffer.episode_length)

            def create_batch(arr_2d, indices):
                return jnp.take(arr_2d, indices, axis=0, mode="wrap")

            batch = jax.vmap(create_batch, in_axes=(1, 0))(
                buffer_state.data[:, envs_idxs, :],
                matrix,
            )
            transitions = replay_buffer._unflatten_fn(batch)
            return buffer_state.replace(key=key), transitions

        crl_config = dict(
            discount=self.discount,
            state_size=state_size,
            goal_indices=tuple(np.asarray(train_env.goal_indices)),
            target_entropy=target_entropy,
            contrastive_loss_fn=self.contrastive_loss_fn,
            energy_fn=self.energy_fn,
            logsumexp_penalty_coeff=self.logsumexp_penalty_coeff,
            carl_actor_encoder_grad=self.carl_actor_encoder_grad,
        )

        _discount = float(self.discount)
        _state_size = int(state_size)
        _goal_indices_arr = jnp.array(train_env.goal_indices)
        _goal_indices_tuple = tuple(int(x) for x in np.asarray(train_env.goal_indices))
        _subgoal_steps = int(self.subgoal_steps)
        _flat_policy = bool(self.flat_policy)
        _use_planner = self.planner_mode != "none"

        def flatten_batch_hcrl(transition, sample_key):
            """Original CRL future-goal relabeling plus optional hierarchy.

            flat_policy=True:
                observation = concat(state, CRL sampled future goal)
                extras["future_state"] = CRL sampled future state

            flat_policy=False:
                high goal = CRL sampled future goal
                low goal = k-step waypoint toward high goal
                observation = concat(state, low goal)
                extras["future_state"] = waypoint future state
            """
            seq_len = transition.observation.shape[0]
            arrangement = jnp.arange(seq_len)
            traj_ids = transition.extras["state_extras"]["traj_id"]

            is_future_mask = jnp.array(
                arrangement[:, None] < arrangement[None], dtype=jnp.float32
            )
            discount = _discount ** jnp.array(
                arrangement[None] - arrangement[:, None], dtype=jnp.float32
            )
            probs = is_future_mask * discount

            single_trajectories = jnp.concatenate(
                [traj_ids[:, jnp.newaxis].T] * seq_len,
                axis=0,
            )
            probs = probs * jnp.equal(single_trajectories, single_trajectories.T)
            probs = probs + jnp.eye(seq_len) * 1e-5

            # Exact CRL sampled future index.
            future_goal_idx = jax.random.categorical(sample_key, jnp.log(probs))

            # Hierarchical waypoint toward the sampled future goal.
            waypoint_idx = jnp.minimum(arrangement + _subgoal_steps, future_goal_idx)
            low_goal_idx = jnp.where(_flat_policy, future_goal_idx, waypoint_idx)

            state = transition.observation[:-1, :_state_size]
            action = transition.action[:-1]
            reward = transition.reward[:-1]
            discount_t = transition.discount[:-1]

            # This is the future state used by original CRL actor loss.
            low_future_state_full = jnp.take(transition.observation, low_goal_idx[:-1], axis=0)
            low_goal = low_future_state_full[:, _goal_indices_arr]
            low_future_state = low_future_state_full[:, :_state_size]

            crl_obs = jnp.concatenate([state, low_goal], axis=-1)

            # Extra high-actor targets. These are ignored in flat mode.
            high_goal_full = jnp.take(transition.observation, future_goal_idx[:-1], axis=0)
            high_target_full = jnp.take(transition.observation, waypoint_idx[:-1], axis=0)
            current_idx = arrangement[:-1]
            final_idx = jnp.max(
                jnp.where(
                    jnp.equal(single_trajectories, single_trajectories.T),
                    arrangement[None, :],
                    -1,
                ),
                axis=1,
            )
            action_offsets = jnp.arange(_subgoal_steps)
            action_idx = jnp.minimum(
                current_idx[:, None] + action_offsets[None, :],
                jnp.maximum(final_idx[:-1, None] - 1, current_idx[:, None]),
            )
            action_idx = jnp.minimum(action_idx, seq_len - 2)
            action_sequence = jnp.take(transition.action, action_idx, axis=0)
            action_sequence = jnp.reshape(action_sequence, (action_sequence.shape[0], -1))

            return Transition(
                observation=crl_obs,
                action=action,
                reward=reward,
                discount=discount_t,
                extras={
                    # Required by original CRL actor loss.
                    "future_state": low_future_state,
                    # Only used by HCRL high actor.
                    "state": state,
                    "carl_goal": low_goal,
                    "action_sequence": action_sequence,
                    "high_actor_goal": high_goal_full[:, _goal_indices_arr],
                    "high_actor_target_goal": high_target_full[:, _goal_indices_arr],
                    "high_actor_target_state": high_target_full[:, :_state_size],
                },
            )

        def _get_action(params, state, goal, key, deterministic):
            if _flat_policy:
                raw_actor_goal = goal
                if self.use_carl_actor:
                    actor_goal = sg_encoder_module.apply(
                        params["sg_encoder"],
                        jnp.concatenate([state, raw_actor_goal], axis=-1),
                    )
                else:
                    actor_goal = raw_actor_goal
            elif _use_planner:
                raw_actor_goal = oracle_subgoal(
                    state,
                    goal,
                    _goal_indices_tuple,
                    self.planner_mode,
                    self.planner_step_size,
                )
                if self.use_carl_actor:
                    actor_goal = sg_encoder_module.apply(
                        params["sg_encoder"],
                        jnp.concatenate([state, raw_actor_goal], axis=-1),
                    )
                else:
                    actor_goal = raw_actor_goal
            else:
                high_key, key = jax.random.split(key)
                high_obs = jnp.concatenate([state, goal], axis=-1)
                subgoal_mean, subgoal_log_std = high_actor_module.apply(
                    params["high_actor"], high_obs
                )
                if deterministic:
                    actor_goal = subgoal_mean
                else:
                    actor_goal, _ = _normal_sample(subgoal_mean, subgoal_log_std, high_key)

            actor_obs = jnp.concatenate([state, actor_goal], axis=-1)
            mean, log_std = actor_module.apply(params["actor"], actor_obs)
            if deterministic:
                action = nn.tanh(mean)
            else:
                action, _ = _tanh_normal_sample(mean, log_std, key)
            return action

        def actor_step(training_state, env, env_state, key, extra_fields):
            obs = env_state.obs
            state = obs[:, :state_size]
            goal = obs[:, state_size:]
            actions = _get_action(_pack_params(training_state), state, goal, key, deterministic=False)
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            obs = env_state.obs
            state = obs[:, :state_size]
            goal = obs[:, state_size:]
            actions = _get_action(
                _pack_params(training_state),
                state,
                goal,
                jax.random.PRNGKey(0),
                deterministic=True,
            )
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        @jax.jit
        def get_experience(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused_t):
                env_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                env_state, transition = actor_step(
                    training_state,
                    train_env,
                    env_state,
                    current_key,
                    extra_fields=("truncation", "traj_id"),
                )
                return (env_state, next_key), transition

            (env_state, _), data = jax.lax.scan(
                f, (env_state, key), (), length=self.unroll_length
            )
            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, buffer_state

        def prefill_replay_buffer(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                env_state, buffer_state = get_experience(
                    training_state, env_state, buffer_state, key
                )
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (training_state, env_state, buffer_state, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        @jax.jit
        def update_networks(carry, batch):
            training_state, key = carry
            key, critic_key, actor_key, high_actor_key = jax.random.split(key, 4)

            training_state, critic_metrics = update_critic(
                crl_config, networks, batch, training_state, critic_key
            )
            if self.use_carl_actor:
                training_state, carl_metrics = update_carl_aux(
                    crl_config, networks, batch, training_state
                )
                training_state, actor_metrics = update_actor_and_alpha_carl(
                    crl_config, networks, batch, training_state, actor_key
                )
            else:
                carl_metrics = {}
                training_state, actor_metrics = update_actor_and_alpha(
                    crl_config, networks, batch, training_state, actor_key
                )

            if _flat_policy or (_use_planner and self.disable_high_actor_update_with_planner):
                high_actor_metrics = {
                    "high_actor_loss": jnp.array(0.0),
                    "high_actor_log_prob": jnp.array(0.0),
                    "high_actor_mse": jnp.array(0.0),
                    "high_actor_std": jnp.array(0.0),
                }
            else:
                if self.use_carl_actor:
                    training_state, high_actor_metrics = update_high_actor_carl(
                        crl_config, networks, batch, training_state, high_actor_key
                    )
                else:
                    training_state, high_actor_metrics = update_high_actor(
                        crl_config, networks, batch, training_state, high_actor_key
                    )
            if _use_planner:
                state = batch.extras["state"]
                final_goal = batch.extras["high_actor_goal"]
                subgoal = oracle_subgoal(
                    state,
                    final_goal,
                    _goal_indices_tuple,
                    self.planner_mode,
                    self.planner_step_size,
                )
                high_actor_metrics.update(
                    planner_metrics(
                        state,
                        final_goal,
                        subgoal,
                        _goal_indices_tuple,
                        self.planner_step_size,
                    )
                )

            training_state = training_state.replace(
                gradient_steps=training_state.gradient_steps + 1
            )
            metrics = {}
            metrics.update(critic_metrics)
            metrics.update(carl_metrics)
            metrics.update(actor_metrics)
            metrics.update(high_actor_metrics)
            metrics["gradient_steps"] = training_state.gradient_steps
            return (training_state, key), metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            experience_key, permute_key, sampling_key, training_key = jax.random.split(key, 4)
            env_state, buffer_state = get_experience(
                training_state, env_state, buffer_state, experience_key
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            buffer_state, transitions = sample_replay_buffer(buffer_state)
            batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
            batches = jax.vmap(flatten_batch_hcrl)(transitions, batch_keys)
            batches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), batches
            )
            permutation = jax.random.permutation(permute_key, batches.observation.shape[0])
            batches = jax.tree_util.tree_map(lambda x: x[permutation], batches)
            num_updates = batches.observation.shape[0] // self.batch_size
            batches = jax.tree_util.tree_map(lambda x: x[: num_updates * self.batch_size], batches)
            batches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (num_updates, self.batch_size) + x.shape[1:]),
                batches,
            )
            (training_state, _), metrics = jax.lax.scan(
                update_networks, (training_state, training_key), batches
            )
            return (training_state, env_state, buffer_state), metrics

        @jax.jit
        def training_epoch(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused_t):
                ts, es, bs, k = carry
                k, train_key = jax.random.split(k, 2)
                (ts, es, bs), step_metrics = training_step(ts, es, bs, train_key)
                return (ts, es, bs, k), step_metrics

            (training_state, env_state, buffer_state, key), epoch_metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )
            epoch_metrics = jax.tree_util.tree_map(lambda x: jnp.asarray(x).mean(), epoch_metrics)
            epoch_metrics = dict(epoch_metrics)
            epoch_metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            epoch_metrics["recent_replay_fraction"] = jnp.asarray(
                _recent_replay_fraction if _use_recent_replay else 0.0
            )
            epoch_metrics["recent_replay_window"] = jnp.asarray(
                _recent_replay_window if _use_recent_replay else 0
            )
            return training_state, env_state, buffer_state, epoch_metrics

        key, prefill_key = jax.random.split(key, 2)
        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        evaluator = ActorEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        def _make_policy(param):
            if _flat_policy:
                if not self.use_carl_actor:
                    # Match the original CRL make_policy style as closely as possible.
                    return lambda obs, rng: actor_module.apply(param["actor"], obs)

                def _flat_carl_policy(obs, rng):
                    s = obs[:, :state_size]
                    g = obs[:, state_size:]
                    z = sg_encoder_module.apply(
                        param["sg_encoder"],
                        jnp.concatenate([s, g], axis=-1),
                    )
                    low_obs = jnp.concatenate([s, z], axis=-1)
                    return actor_module.apply(param["actor"], low_obs)

                return _flat_carl_policy

            def _policy(obs, rng):
                s = obs[:, :state_size]
                g = obs[:, state_size:]
                if _use_planner:
                    raw_subgoal_mean = oracle_subgoal(
                        s,
                        g,
                        _goal_indices_tuple,
                        self.planner_mode,
                        self.planner_step_size,
                    )
                    if self.use_carl_actor:
                        subgoal_mean = sg_encoder_module.apply(
                            param["sg_encoder"],
                            jnp.concatenate([s, raw_subgoal_mean], axis=-1),
                        )
                    else:
                        subgoal_mean = raw_subgoal_mean
                else:
                    high_obs = jnp.concatenate([s, g], axis=-1)
                    subgoal_mean, _ = high_actor_module.apply(param["high_actor"], high_obs)
                low_obs = jnp.concatenate([s, subgoal_mean], axis=-1)
                return actor_module.apply(param["actor"], low_obs)

            return _policy

        training_walltime = 0.0
        logging.info("starting HCRL = CRL + high actor training....")

        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)
            training_state, env_state, buffer_state, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key
            )

            if _coverage_history is not None:
                obs_np = np.array(jax.device_get(env_state.obs)).reshape(-1, obs_size)
                _coverage_history.append(obs_np[:, [_cov_xy0, _cov_xy1]])

            metrics = jax.tree_util.tree_map(lambda x: jnp.asarray(x).mean(), metrics)
            metrics = jax.tree_util.tree_map(lambda x: float(x.block_until_ready()), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            current_step = int(training_state.env_steps.item())

            metrics = {
                "training/sps": float(sps),
                "training/walltime": float(training_walltime),
                "training/envsteps": current_step,
                **{f"training/{name}": value for name, value in metrics.items()},
            }

            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            do_render = ne % config.visualization_interval == 0
            make_policy = _make_policy

            if _coverage_history:
                all_pos = np.concatenate(_coverage_history, axis=0)
                bins = 50
                h, xedges, yedges = np.histogram2d(
                    all_pos[:, 0], all_pos[:, 1], bins=bins
                )
                denom = max(float(h.sum()), 1.0)
                p = h / denom
                p_nz = p[p > 0]
                coverage_entropy = float(-np.sum(p_nz * np.log(p_nz)))
                occupied_cells = int(np.sum(h > 0))
                metrics["state_coverage_entropy"] = coverage_entropy
                metrics["state_coverage_cells"] = occupied_cells

                if do_render:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(
                        h.T,
                        origin="lower",
                        aspect="auto",
                        cmap="hot",
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    )
                    plt.colorbar(im, ax=ax, label="visit count")
                    ax.set_title(
                        f"State Coverage (step {current_step}, H={coverage_entropy:.2f}, "
                        f"cells={occupied_cells}/{bins * bins})"
                    )
                    ax.set_xlabel(f"dim {_cov_xy0}")
                    ax.set_ylabel(f"dim {_cov_xy1}")
                    wandb.log({"state_coverage": wandb.Image(fig)}, step=current_step)
                    plt.close(fig)

            params = _pack_params(training_state)
            progress_fn(
                current_step,
                metrics,
                make_policy,
                params,
                unwrapped_env,
                do_render=do_render,
            )

            if config.checkpoint_logdir:
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

        total_steps = current_step
        assert total_steps >= config.total_env_steps
        logging.info("total steps: %s", total_steps)
        return make_policy, _pack_params(training_state), metrics
