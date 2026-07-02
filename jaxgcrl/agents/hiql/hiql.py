"""Online HIQL training loop.

Follows the CRL agent pattern: collects experience into a replay buffer,
samples trajectory segments, relabels goals inside a jitted flatten_batch,
and updates all HIQL networks with a single gradient step.
"""

import logging
import pickle
import random
import time
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.agents.planner import (
    PlannerMode,
    oracle_subgoal,
    planner_metrics,
    validate_planner_config,
)
from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from .losses import update_hiql
from .networks import GCActor, GoalRep, Value

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class TrainingState:
    """Contains training state for the learner."""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    hiql_state: TrainState
    target_params: Any


class Transition(NamedTuple):
    """Container for a transition."""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


# ---------------------------------------------------------------------------
# HIQL agent dataclass
# ---------------------------------------------------------------------------

@dataclass
class HIQL:
    """Online Hierarchical Implicit Q-Learning (HIQL) agent."""

    lr: float = 3e-4
    batch_size: int = 1024

    discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.7

    low_alpha: float = 3.0
    high_alpha: float = 3.0
    subgoal_steps: int = 25
    rep_dim: int = 10
    low_actor_rep_grad: bool = False
    const_std: bool = True
    flat_policy: bool = False
    planner_mode: PlannerMode = "none"
    planner_step_size: float = 2.0
    disable_high_actor_update_with_planner: bool = False

    value_p_curgoal: float = 0.2
    value_p_trajgoal: float = 0.5
    value_geom_sample: bool = True

    layer_norm: bool = True
    value_hidden: Tuple[int, ...] = (512, 512, 512)
    actor_hidden: Tuple[int, ...] = (512, 512, 512)

    train_step_multiplier: int = 1
    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62

    def train_fn(
        self,
        config,
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn=None,
        progress_fn: Callable[..., None] = lambda *args, **kwargs: None,
    ):
        unwrapped_env = train_env
        train_env = TrajectoryIdWrapper(train_env)
        train_env = envs.training.wrap(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )
        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = envs.training.wrap(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = np.ceil(self.min_replay_size / self.unroll_length)
        num_training_steps_per_epoch = int(
            np.ceil(
                (config.total_env_steps - num_prefill_env_steps)
                / (config.num_evals * env_steps_per_actor_step)
            )
        )

        assert num_training_steps_per_epoch > 0

        logging.info("num_prefill_actor_steps: %d", num_prefill_actor_steps)
        logging.info("num_training_steps_per_epoch: %d", num_training_steps_per_epoch)

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key = jax.random.split(key, 4)
        key, goal_rep_key, value1_key, value2_key, low_actor_key, high_actor_key = jax.random.split(key, 6)

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

        # ===== Network definitions =====
        goal_rep_module = GoalRep(
            layer_sizes=list(self.value_hidden),
            rep_dim=self.rep_dim,
            layer_norm=self.layer_norm,
        )
        value_module = Value(
            layer_sizes=list(self.value_hidden),
            layer_norm=self.layer_norm,
        )
        low_actor_module = GCActor(
            action_dim=action_size,
            layer_sizes=list(self.actor_hidden),
            const_std=self.const_std,
        )
        high_actor_module = GCActor(
            action_dim=self.rep_dim,
            layer_sizes=list(self.actor_hidden),
            const_std=self.const_std,
        )

        networks = {
            "goal_rep": goal_rep_module,
            "value": value_module,
            "low_actor": low_actor_module,
            "high_actor": high_actor_module,
        }

        # ===== Init params =====
        dummy_state = jnp.ones((1, state_size))
        dummy_goal = jnp.ones((1, goal_size))
        dummy_rep = jnp.ones((1, self.rep_dim))

        goal_rep_params = goal_rep_module.init(goal_rep_key, dummy_state, dummy_goal)
        value1_params = value_module.init(value1_key, dummy_state, dummy_rep)
        value2_params = value_module.init(value2_key, dummy_state, dummy_rep)
        low_actor_params = low_actor_module.init(low_actor_key, dummy_state, dummy_rep)
        high_actor_params = high_actor_module.init(high_actor_key, dummy_state, dummy_goal)

        all_params = {
            "goal_rep": goal_rep_params,
            "value1": value1_params,
            "value2": value2_params,
            "low_actor": low_actor_params,
            "high_actor": high_actor_params,
        }

        hiql_state = TrainState.create(
            apply_fn=None,
            params=all_params,
            tx=optax.adam(learning_rate=self.lr),
        )

        target_params = {
            "goal_rep": goal_rep_params,
            "value1": value1_params,
            "value2": value2_params,
        }

        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            hiql_state=hiql_state,
            target_params=target_params,
        )

        # ===== Replay buffer =====
        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))
        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                }
            },
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

        # ===== Config dict for losses =====
        planner_disables_high_actor = (
            self.planner_mode != "none" and self.disable_high_actor_update_with_planner
        )
        hiql_config = dict(
            discount=self.discount,
            tau=self.tau,
            expectile=self.expectile,
            low_alpha=self.low_alpha,
            high_alpha=self.high_alpha,
            low_actor_rep_grad=self.low_actor_rep_grad,
            goal_indices=tuple(train_env.goal_indices),
            flat_policy=self.flat_policy or planner_disables_high_actor,
        )

        # Closed-over constants for goal relabeling
        _discount = float(self.discount)
        _state_size = int(state_size)
        _goal_indices_arr = jnp.array(train_env.goal_indices)
        _subgoal_steps = int(self.subgoal_steps)
        _value_p_curgoal = float(self.value_p_curgoal)
        _value_p_trajgoal = float(self.value_p_trajgoal)
        _value_geom_sample = bool(self.value_geom_sample)
        _flat_policy = bool(self.flat_policy)
        _goal_indices_tuple = tuple(int(x) for x in np.asarray(train_env.goal_indices))
        _use_planner = self.planner_mode != "none"

        def flatten_batch_hiql(transition, sample_key):
            """Relabel a trajectory segment into an HIQL training batch.
            """
            seq_len = transition.observation.shape[0]
            arrangement = jnp.arange(seq_len)

            traj_ids = transition.extras["state_extras"]["traj_id"]
            same_traj = jnp.equal(traj_ids[:, None], traj_ids[None, :])

            big_neg = jnp.where(same_traj, arrangement[None, :], -1)
            final_idx = jnp.max(big_neg, axis=1)

            k1, k2, k3, k4, k5 = jax.random.split(sample_key, 5)

            # Geometric future goals
            geom_offsets = jax.random.geometric(k1, p=1 - _discount, shape=(seq_len,))
            geom_goal_idx = jnp.minimum(arrangement + geom_offsets, final_idx)

            # Uniform future goals
            distances = jax.random.uniform(k2, shape=(seq_len,))
            uniform_goal_idx = jnp.round(
                jnp.minimum(arrangement + 1, final_idx) * distances + final_idx * (1 - distances)
            ).astype(jnp.int32)

            traj_goal_idx = jnp.where(_value_geom_sample, geom_goal_idx, uniform_goal_idx)

            random_goal_idx = jax.random.randint(k3, shape=(seq_len,), minval=0, maxval=seq_len)

            choice = jax.random.uniform(k4, shape=(seq_len,))
            value_goal_idx = jnp.where(
                choice < _value_p_curgoal,
                arrangement,
                jnp.where(
                    choice < _value_p_curgoal + _value_p_trajgoal,
                    traj_goal_idx,
                    random_goal_idx,
                ),
            )

            value_goals = transition.observation[value_goal_idx]
            successes = (arrangement == value_goal_idx).astype(jnp.float32)
            rewards = successes - 1.0
            masks = 1.0 - successes

            # Low-level actor goals (flat: reuse value goals; hierarchical: subgoal_steps ahead)
            low_goal_idx = jnp.where(
                _flat_policy,
                value_goal_idx,
                jnp.minimum(arrangement + _subgoal_steps, final_idx),
            )
            low_actor_goals = transition.observation[low_goal_idx]
            low_successes = jnp.where(
                _flat_policy,
                successes,
                (arrangement == low_goal_idx).astype(jnp.float32),
            )
            low_rewards = low_successes - 1.0
            low_masks = 1.0 - low_successes

            # High-level actor goals & targets
            high_geom_offsets = jax.random.geometric(k5, p=1 - _discount, shape=(seq_len,))
            high_traj_goal_idx = jnp.minimum(arrangement + high_geom_offsets, final_idx)
            high_traj_target_idx = jnp.minimum(arrangement + _subgoal_steps, high_traj_goal_idx)

            k6, k7 = jax.random.split(k5)
            high_random_goal_idx = jax.random.randint(k6, shape=(seq_len,), minval=0, maxval=seq_len)
            high_random_target_idx = jnp.minimum(arrangement + _subgoal_steps, final_idx)

            pick_random = jax.random.uniform(k7, shape=(seq_len,)) < (1.0 - _value_p_curgoal - _value_p_trajgoal)
            high_goal_idx = jnp.where(pick_random, high_random_goal_idx, high_traj_goal_idx)
            high_target_idx = jnp.where(pick_random, high_random_target_idx, high_traj_target_idx)

            high_actor_goals = transition.observation[high_goal_idx]
            high_actor_targets = transition.observation[high_target_idx]

            observations = transition.observation[:-1, :_state_size]
            next_observations = transition.observation[1:, :_state_size]
            actions = transition.action[:-1]

            # Goals use goal_indices; high_actor_targets stays state_size (used as obs for Value)
            value_goals = value_goals[:-1][:, _goal_indices_arr]
            rewards = rewards[:-1]
            masks = masks[:-1]
            low_actor_goals = low_actor_goals[:-1][:, _goal_indices_arr]
            low_rewards = low_rewards[:-1]
            low_masks = low_masks[:-1]
            high_actor_goals = high_actor_goals[:-1][:, _goal_indices_arr]
            high_actor_targets = high_actor_targets[:-1, :_state_size]

            return {
                "observations": observations,
                "next_observations": next_observations,
                "actions": actions,
                "value_goals": value_goals,
                "rewards": rewards,
                "masks": masks,
                "low_actor_goals": low_actor_goals,
                "low_rewards": low_rewards,
                "low_masks": low_masks,
                "high_actor_goals": high_actor_goals,
                "high_actor_targets": high_actor_targets,
            }

        # ===== Actor step (for data collection) =====
        _flat = self.flat_policy

        def _get_action(params, state, goal, key, deterministic):
            """Shared action selection for both stochastic and deterministic modes."""
            if _flat:
                phi = goal_rep_module.apply(params["goal_rep"], state, goal)
            elif _use_planner:
                raw_subgoal = oracle_subgoal(
                    state,
                    goal,
                    _goal_indices_tuple,
                    self.planner_mode,
                    self.planner_step_size,
                )
                phi = goal_rep_module.apply(params["goal_rep"], state, raw_subgoal)
            else:
                high_key, key = jax.random.split(key)
                high_dist = high_actor_module.apply(params["high_actor"], state, goal)
                phi = high_dist.mode() if deterministic else high_dist.sample(seed=high_key)

            low_dist = low_actor_module.apply(params["low_actor"], state, phi)
            actions = low_dist.mode() if deterministic else low_dist.sample(seed=key)
            return jnp.clip(actions, -1, 1)

        def actor_step(training_state, env, env_state, key, extra_fields):
            obs = env_state.obs
            state = obs[:, :state_size]
            goal = obs[:, state_size:]
            params = training_state.hiql_state.params
            actions = _get_action(params, state, goal, key, deterministic=False)

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
            params = training_state.hiql_state.params
            actions = _get_action(params, state, goal, jax.random.PRNGKey(0), deterministic=True)

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}

            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        # ===== Experience collection =====
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

            (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=self.unroll_length)
            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, buffer_state

        def prefill_replay_buffer(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                env_state, buffer_state = get_experience(
                    training_state,
                    env_state,
                    buffer_state,
                    key,
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

        # ===== Training step =====
        @jax.jit
        def update_networks(carry, batch):
            training_state, key = carry
            key, update_key = jax.random.split(key)
            training_state, metrics = update_hiql(
                hiql_config, networks, batch, training_state, update_key
            )
            if _use_planner:
                planner_subgoal = oracle_subgoal(
                    batch["observations"],
                    batch["high_actor_goals"],
                    _goal_indices_tuple,
                    self.planner_mode,
                    self.planner_step_size,
                )
                metrics.update(
                    planner_metrics(
                        batch["observations"],
                        batch["high_actor_goals"],
                        planner_subgoal,
                        _goal_indices_tuple,
                        self.planner_step_size,
                    )
                )
            return (training_state, key), metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            experience_key, permute_key, sampling_key, training_key = jax.random.split(key, 4)

            env_state, buffer_state = get_experience(
                training_state,
                env_state,
                buffer_state,
                experience_key,
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            buffer_state, transitions = replay_buffer.sample(buffer_state)

            batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
            batches = jax.vmap(flatten_batch_hiql)(transitions, batch_keys)

            batches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), batches
            )

            permutation = jax.random.permutation(permute_key, batches["observations"].shape[0])
            batches = jax.tree_util.tree_map(lambda x: x[permutation], batches)
            batches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
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
                (ts, es, bs), metrics = training_step(ts, es, bs, train_key)
                return (ts, es, bs, k), metrics

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )

            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            return training_state, env_state, buffer_state, metrics

        # ===== Prefill =====
        key, prefill_key = jax.random.split(key, 2)
        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        # ===== Evaluator =====
        evaluator = ActorEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        # ===== Main loop =====
        training_walltime = 0
        logging.info("starting training....")
        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)

            training_state, env_state, buffer_state, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time

            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            current_step = int(training_state.env_steps.item())

            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            do_render = ne % config.visualization_interval == 0
            def _make_policy(param):
                def _policy(obs, rng):
                    s = obs[:, :state_size]
                    g = obs[:, state_size:]
                    if _flat:
                        phi = goal_rep_module.apply(param["goal_rep"], s, g)
                    elif _use_planner:
                        raw_subgoal = oracle_subgoal(
                            s,
                            g,
                            _goal_indices_tuple,
                            self.planner_mode,
                            self.planner_step_size,
                        )
                        phi = goal_rep_module.apply(param["goal_rep"], s, raw_subgoal)
                    else:
                        phi = high_actor_module.apply(param["high_actor"], s, g).mode()
                    action = low_actor_module.apply(param["low_actor"], s, phi).mode()
                    action = jnp.clip(action, -1, 1)
                    return action, {}
                return _policy
            make_policy = _make_policy

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.hiql_state.params,
                unwrapped_env,
                do_render=do_render,
            )

            if config.checkpoint_logdir:
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, training_state.hiql_state.params)

        total_steps = current_step
        assert total_steps >= config.total_env_steps

        logging.info("total steps: %s", total_steps)

        return make_policy, training_state.hiql_state.params, metrics
