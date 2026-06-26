"""Online HCARL: CARL representation + HIQL-style value/AWR hierarchy.

Design goal:
  - CARL learns phi(s, g) using state-goal/action-sequence contrastive loss.
  - Value learns V(s, phi(s, g)) with an HIQL-style expectile TD loss.
  - Low actor is AWR/NLL on dataset actions conditioned on phi(s, local_goal).
  - High actor is AWR/NLL on latent subgoals phi(s, k-step_subgoal).
"""

import logging
import pickle
import random
import time
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

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
from flax.training import checkpoints
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

# Reuse original CRL networks. Do not redefine these in HCARL.
from jaxgcrl.agents.crl.networks import Actor, Encoder

from .losses import (
    _normal_sample,
    _tanh_normal_sample,
    update_actor_and_alpha,
    update_critic,
)

Metrics = types.Metrics
Env = Union[envs.Env, envs.Wrapper]
State = envs.State


@dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    target_critic_params: Any
    alpha_state: TrainState
    high_actor_state: TrainState


class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: Any = ()


class Value(nn.Module):
    layer_sizes: Tuple[int, ...] = (512, 512, 512)
    use_ln: bool = False

    @nn.compact
    def __call__(self, state, z):
        x = jnp.concatenate([state, z], axis=-1)
        for hidden_size in self.layer_sizes:
            x = nn.Dense(hidden_size)(x)
            if self.use_ln:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
        return nn.Dense(1)(x)


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def load_carl_params(path: str):
    """Loads separate HCARL encoder parameter trees from a CARL checkpoint."""
    params = checkpoints.restore_checkpoint(path, target=None)
    if "params" in params and "sg_encoder" in params["params"]:
        params = params["params"]

    sg_params = params["sg_encoder"]
    a_params = params["a_encoder"]
    if "params" not in sg_params:
        sg_params = {"params": sg_params}
    if "params" not in a_params:
        a_params = {"params": a_params}
    return sg_params, a_params


def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


def _pack_params(training_state: TrainingState):
    return {
        "actor": training_state.actor_state.params,
        "sg_encoder": training_state.critic_state.params["sg_encoder"],
        "a_encoder": training_state.critic_state.params["a_encoder"],
        "high_actor": training_state.high_actor_state.params,
    }


@dataclass
class HCARL:
    subgoal_steps: int = 25
    high_actor_hidden: Tuple[int, ...] = (512, 512, 512)
    value_hidden: Tuple[int, ...] = (512, 512, 512)
    flat_policy: bool = False

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    discount: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    train_step_multiplier: int = 1
    disable_entropy_actor: bool = False

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62
    h_dim: int = 256
    n_hidden: int = 2
    skip_connections: int = 4
    use_relu: bool = False
    use_ln: bool = False
    repr_dim: int = 64

    is_pretrain_carl_high: bool = False
    pretrained_carl_path: Optional[str] = "/scratch/cluster/idutta/JaxGCRL/jaxgcrl/checkpoints/antmaze-medium-navigate-v0/rep_params_89400/checkpoint"
    pretrained_high_actor_path: Optional[str] = "/scratch/cluster/idutta/JaxGCRL/jaxgcrl/checkpoints/high_actor/hcrl_high_actor_step_20744704.pkl"

    contrastive_loss_fn: str = "fwd_infonce"
    energy_fn: str = "norm"
    target_entropy_scale: float = 0.5

    value_goal_eps: float = 0.5
    value_loss_coeff: float = 1.0
    expectile: float = 0.7
    target_update_rate: float = 0.005
    actor_beta: float = 1.0
    actor_max_weight: float = 20.0
    low_actor_beta: float = 1.0
    low_actor_max_weight: float = 20.0
    high_actor_beta: float = 1.0
    high_actor_max_weight: float = 20.0
    stop_value_encoder_grad: bool = True

    # HIQL/GCSDataset-style goal sampling.
    p_randomgoal: float = 0.3
    p_trajgoal: float = 0.5
    p_currgoal: float = 0.2
    geom_sample: int = 0
    reward_scale: float = 1.0
    reward_shift: float = 0.0
    terminal: bool = False
    high_p_randomgoal: float = 0.0

    log_state_coverage: bool = False
    state_coverage_xy_dims: Tuple[int, int] = (0, 1)

    def check_config(self, config):
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
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

        logging.info("HCARL flat_policy: %s", self.flat_policy)
        logging.info("HCARL log_state_coverage: %s", self.log_state_coverage)

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
        num_training_steps_per_epoch = (config.total_env_steps - num_prefill_env_steps) // (
            config.num_evals * env_steps_per_actor_step
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
        key, sg_key, a_key, actor_key, high_key, value_key = jax.random.split(key, 6)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size
        target_entropy = -self.target_entropy_scale * action_size

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

        # CARL representation networks plus low/high actors and value.
        sg_encoder_module = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        a_encoder_module = Encoder(
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
        # Paper-style high actor outputs a latent CARL subgoal z, not raw goal coordinates.
        high_actor_module = Actor(
            action_size=goal_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        value_module = Value(
            layer_sizes=self.value_hidden,
            use_ln=self.use_ln,
        )
        networks = {
            "sg_encoder": sg_encoder_module,
            "a_encoder": a_encoder_module,
            "value_module": value_module,
            "actor": actor_module,
            "high_actor": high_actor_module,
        }

        dummy_state = jnp.ones((1, state_size))
        dummy_goal = jnp.ones((1, goal_size))
        # CARL action encoder receives the fixed-length action sequence (a_t, ..., a_{t+k-1}), flattened into one vector.
        dummy_action = jnp.ones((1, action_size * self.subgoal_steps))
        dummy_obs = jnp.ones((1, obs_size))
        dummy_sg = jnp.concatenate([dummy_state, dummy_goal], axis=-1)
        dummy_rep = jnp.ones((1, self.repr_dim))

        sg_params = sg_encoder_module.init(sg_key, dummy_sg)
        # Action encoder is initialized on flattened k-step action sequences.
        a_params = a_encoder_module.init(a_key, dummy_action)
        # Low actor is conditioned on [state, phi(state, goal/subgoal)].
        dummy_actor_obs = jnp.ones((1, state_size + self.repr_dim))
        actor_params = actor_module.init(actor_key, dummy_actor_obs)
        high_params = high_actor_module.init(high_key, dummy_obs)
        value_params = value_module.init(value_key, dummy_state, dummy_rep)

        if self.is_pretrain_carl_high:
            if not self.pretrained_carl_path:
                raise ValueError("pretrained_carl_path is required when is_pretrain_carl_high=True")
            if not self.pretrained_high_actor_path:
                raise ValueError("pretrained_high_actor_path is required when is_pretrain_carl_high=True")

            sg_params, a_params = load_carl_params(self.pretrained_carl_path)
            high_params = load_params(self.pretrained_high_actor_path)
            if "high_actor" in high_params:
                high_params = high_params["high_actor"]
            logging.info("Loaded frozen CARL checkpoint from %s", self.pretrained_carl_path)
            logging.info("Loaded frozen high actor checkpoint from %s", self.pretrained_high_actor_path)

        critic_state = TrainState.create(
            apply_fn=None,
            params={
                "sg_encoder": sg_params,
                "a_encoder": a_params,
                "value1": value_params,
            },
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
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            target_critic_params=critic_state.params,
            alpha_state=alpha_state,
            high_actor_state=high_actor_state,
        )

        dummy_transition = Transition(
            observation=jnp.zeros((obs_size,)),
            action=jnp.zeros((action_size,)),
            reward=0.0,
            discount=0.0,
            extras={"state_extras": {"truncation": 0.0, "traj_id": 0.0}, 'low_actor_goals': jnp.zeros((goal_size,))},
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

        crl_config = dict(
            discount=self.discount,
            state_size=state_size,
            goal_indices=tuple(np.asarray(train_env.goal_indices)),
            target_entropy=target_entropy,
            contrastive_loss_fn=self.contrastive_loss_fn,
            energy_fn=self.energy_fn,
            logsumexp_penalty_coeff=self.logsumexp_penalty_coeff,
            value_goal_eps=self.value_goal_eps,
            value_loss_coeff=self.value_loss_coeff,
            expectile=self.expectile,
            target_update_rate=self.target_update_rate,
            actor_beta=self.actor_beta,
            actor_max_weight=self.actor_max_weight,
            low_actor_beta=self.low_actor_beta,
            low_actor_max_weight=self.low_actor_max_weight,
            high_actor_beta=self.high_actor_beta,
            high_actor_max_weight=self.high_actor_max_weight,
            stop_value_encoder_grad=self.stop_value_encoder_grad,
            p_randomgoal=self.p_randomgoal,
            p_trajgoal=self.p_trajgoal,
            p_currgoal=self.p_currgoal,
            geom_sample=self.geom_sample,
            reward_scale=self.reward_scale,
            reward_shift=self.reward_shift,
            terminal=self.terminal,
            high_p_randomgoal=self.high_p_randomgoal,
            flat_policy=self.flat_policy,
        )

        _discount = float(self.discount)
        _state_size = int(state_size)
        _goal_indices_arr = jnp.array(train_env.goal_indices)
        _subgoal_steps = int(self.subgoal_steps)
        _flat_policy = bool(self.flat_policy)

        def flatten_batch_hcrl(transition, sample_key):
            """HIQL GCSDataset-style sampling, kept close to the original code.

            Returns the same four objects as GCSDataset:
                goals        -> value / flat-policy goals
                low_goals    -> k-step goals for the low policy
                high_goals   -> high-level policy goals
                high_targets -> high-level waypoint targets
            """
            seq_len = transition.observation.shape[0]
            indx = jnp.arange(seq_len)
            traj_ids = transition.extras["state_extras"]["traj_id"]

            # Equivalent of terminal_locs[searchsorted(terminal_locs, indx)] inside this sampled trajectory chunk.
            same_traj = jnp.equal(traj_ids[:, None], traj_ids[None, :])
            future_or_self = indx[None, :] >= indx[:, None]
            final_indx = jnp.max(
                jnp.where(same_traj & future_or_self, indx[None, :], indx[:, None]),
                axis=1,
            )

            def sample_goals(key, p_randomgoal, p_trajgoal, p_currgoal):
                random_key, traj_key, traj_pick_key, curr_key, geom_key = jax.random.split(key, 5)
                goal_indx = jax.random.randint(random_key, shape=(seq_len,), minval=0, maxval=seq_len)

                if bool(self.geom_sample):
                    us = jax.random.uniform(geom_key, shape=(seq_len,))
                    offset = jnp.ceil(
                        jnp.log(1.0 - us + 1e-8) / jnp.log(jnp.minimum(_discount, 0.999999))
                    ).astype(jnp.int32)
                    middle_goal_indx = jnp.minimum(indx + offset, final_indx)
                else:
                    distance = jax.random.uniform(traj_key, shape=(seq_len,))
                    middle_goal_indx = jnp.round(
                        jnp.minimum(indx + 1, final_indx).astype(jnp.float32) * distance
                        + final_indx.astype(jnp.float32) * (1.0 - distance)
                    ).astype(jnp.int32)

                traj_prob = p_trajgoal / jnp.maximum(1.0 - p_currgoal, 1e-6)
                goal_indx = jnp.where(
                    jax.random.uniform(traj_pick_key, shape=(seq_len,)) < traj_prob,
                    middle_goal_indx,
                    goal_indx,
                )
                goal_indx = jnp.where(
                    jax.random.uniform(curr_key, shape=(seq_len,)) < p_currgoal,
                    indx,
                    goal_indx,
                )
                return goal_indx

            goal_key, high_traj_key, high_random_key, high_pick_key = jax.random.split(sample_key, 4)

            # Same as GCDataset.sample(): sample relabelled goals and recompute reward/mask.
            # Not used for now
            goal_indx = sample_goals(
                goal_key,
                float(self.p_randomgoal),
                float(self.p_trajgoal),
                float(self.p_currgoal),
            )
            success = (goal_indx == indx).astype(jnp.float32)
            reward = success * float(self.reward_scale) + float(self.reward_shift)
            mask = jnp.where(bool(self.terminal), 1.0 - success, jnp.ones_like(success))
            goals = jnp.take(transition.observation, goal_indx, axis=0)[:, _goal_indices_arr]

            # Same as GCSDataset.sample(): low_goals = s_{t+k}.
            way_indx = jnp.minimum(indx + _subgoal_steps, final_indx)
            way_obs = jnp.take(transition.observation, way_indx, axis=0)
            low_goals = way_obs[:, _goal_indices_arr]

            # Not used for pretrained high actor
            # Same as GCSDataset.sample(): high_goals and high_targets.
            high_traj_goal_indx = sample_goals(high_traj_key, 0.0, 1.0, 0.0)
            high_traj_target_indx = jnp.minimum(indx + _subgoal_steps, high_traj_goal_indx)

            high_random_goal_indx = jax.random.randint(
                high_random_key,
                shape=(seq_len,),
                minval=0,
                maxval=seq_len,
            )
            high_random_target_indx = way_indx

            pick_random = jax.random.uniform(high_pick_key, shape=(seq_len,)) < float(self.high_p_randomgoal)
            high_goal_indx = jnp.where(pick_random, high_random_goal_indx, high_traj_goal_indx)
            high_target_indx = jnp.where(pick_random, high_random_target_indx, high_traj_target_indx)

            high_goals = jnp.take(transition.observation, high_goal_indx, axis=0)[:, _goal_indices_arr]
            high_targets_full = jnp.take(transition.observation, high_target_indx, axis=0)
            high_targets = high_targets_full[:, _goal_indices_arr]

            state = transition.observation[:-1, :_state_size]
            next_state = transition.observation[1:, :_state_size]

            # Low actor is still trained on the first primitive action a_t.
            action = transition.action[:-1]

            action_offsets = jnp.arange(_subgoal_steps)
            last_action_indx = jnp.maximum(indx, final_indx - 1)
            action_seq_indx = jnp.minimum(
                indx[:, None] + action_offsets[None, :],
                last_action_indx[:, None],
            )
            action_seq = jnp.take(transition.action, action_seq_indx, axis=0)
            action_seq = jnp.reshape(action_seq, (seq_len, _subgoal_steps * action_size))
            stored_low_actor_goals = transition.extras["low_actor_goals"]

            if _flat_policy:
                train_goal = goals[:-1]
                train_reward = reward[:-1]
                train_mask = mask[:-1]
                train_success = success[:-1]
            else:
                # Raw high-actor subgoal that generated action a_t.
                train_goal = stored_low_actor_goals[:-1]  # [seq_len - 1, goal_size]

                next_goal_state = next_state[:, _goal_indices_arr]
                train_success = (jnp.linalg.norm(next_goal_state - train_goal, axis=-1) < float(self.value_goal_eps)).astype(jnp.float32)
                train_reward = (train_success * float(self.reward_scale) + float(self.reward_shift))
                train_mask = jnp.where(bool(self.terminal), 1.0 - train_success, jnp.ones_like(train_success))

            return Transition(
                observation=jnp.concatenate([state, train_goal], axis=-1),
                action=action,
                reward=train_reward,
                discount=train_mask * transition.discount[:-1],
                extras={
                    "future_state": way_obs[:-1, :_state_size],
                    "next_state": next_state,
                    "state": state,
                    "value_goal": train_goal,
                    "low_actor_goal": train_goal,
                    "next_low_actor_goal": train_goal,
                    "action_sequence": action_seq[:-1],
                    "high_actor_goal": high_goals[:-1],
                    "high_actor_target_goal": high_targets[:-1],
                    "high_actor_target_state": high_targets_full[:-1, :_state_size],
                    "hiql_value_goal_success": train_success,
                    "hiql_pick_random_high_goal": pick_random[:-1].astype(jnp.float32),
                },
            )

        def _get_action(params, state, goal, key, deterministic):
            if _flat_policy:
                # Flat mode still uses the CARL representation as the actor goal.
                raw_actor_goal = sg_encoder_module.apply(
                    params["sg_encoder"],
                    jnp.concatenate([state, goal], axis=-1),
                )
            else:
                high_key, key = jax.random.split(key)
                high_obs = jnp.concatenate([state, goal], axis=-1)
                subgoal_mean, subgoal_log_std = high_actor_module.apply(
                    params["high_actor"], high_obs
                )
                if deterministic:
                    raw_actor_goal = subgoal_mean
                else:
                    raw_actor_goal, _ = _normal_sample(subgoal_mean, subgoal_log_std, high_key)

            actor_goal = sg_encoder_module.apply(
                params["sg_encoder"],
                jnp.concatenate([state, raw_actor_goal], axis=-1),
            )
            actor_obs = jnp.concatenate([state, actor_goal], axis=-1)
            mean, log_std = actor_module.apply(params["actor"], actor_obs)
            if deterministic:
                action = nn.tanh(mean)
            else:
                action, _ = _tanh_normal_sample(mean, log_std, key)
            return action, raw_actor_goal

        def actor_step(training_state, env, env_state, key, extra_fields):
            obs = env_state.obs
            state = obs[:, :state_size]
            goal = obs[:, state_size:]
            actions, low_actor_goals = _get_action(_pack_params(training_state), state, goal, key, deterministic=False)
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras, "low_actor_goals": low_actor_goals},
            )

        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            obs = env_state.obs
            state = obs[:, :state_size]
            goal = obs[:, state_size:]
            actions, low_actor_goals = _get_action(
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
                extras={"state_extras": state_extras, "low_actor_goals": low_actor_goals},
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
            training_state, actor_metrics = update_actor_and_alpha(
                crl_config, networks, batch, training_state, actor_key
            )

            # The high actor is loaded once and kept frozen.
            high_actor_metrics = {
                "high_actor_loss": jnp.array(0.0),
                "high_actor_log_prob": jnp.array(0.0),
                "high_actor_mse": jnp.array(0.0),
                "high_actor_std": jnp.array(0.0),
                "high_actor_weight": jnp.array(0.0),
                "high_actor_adv": jnp.array(0.0),
                "high_actor_v_curr": jnp.array(0.0),
                "high_actor_v_next": jnp.array(0.0),
                "high_log_std_mean": jnp.array(0.0),
                "high_latent_noise_mean": jnp.array(0.0),
            }

            training_state = training_state.replace(
                gradient_steps=training_state.gradient_steps + 1
            )
            metrics = {}
            metrics.update(critic_metrics)
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

            buffer_state, transitions = replay_buffer.sample(buffer_state)
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
            def _policy(obs, rng):
                s = obs[:, :state_size]
                g = obs[:, state_size:]
                if _flat_policy:
                    actor_goal = sg_encoder_module.apply(
                        param["sg_encoder"],
                        jnp.concatenate([s, g], axis=-1),
                    )
                else:
                    high_obs = jnp.concatenate([s, g], axis=-1)
                    actor_goal, _ = high_actor_module.apply(param["high_actor"], high_obs)
                    actor_goal = sg_encoder_module.apply(
                        param["sg_encoder"],
                        jnp.concatenate([s, actor_goal], axis=-1),
                    )
                low_obs = jnp.concatenate([s, actor_goal], axis=-1)
                return actor_module.apply(param["actor"], low_obs)

            return _policy

        training_walltime = 0.0
        logging.info("starting HCARL = CARL representation + value/AWR training....")

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