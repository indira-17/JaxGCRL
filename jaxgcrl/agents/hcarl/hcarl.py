"""Online HCARL: CARL representation + HIQL-style value/AWR hierarchy.

Design goal:
  - CARL learns phi(s, g) using state-goal/action-sequence contrastive loss.
  - Low value learns V(s, g_raw) with an HIQL-style expectile TD loss.
  - Low actor is AWR/NLL on dataset actions conditioned on phi(s, local_goal).
  - High actor is AWR/NLL on latent subgoals phi(s, k-step_subgoal).
"""

import logging
import pickle
import random
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Union

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

# Reuse original CRL networks. Do not redefine these in HCARL.
from jaxgcrl.agents.crl.networks import Actor, Encoder

from .losses import (
    _normal_sample,
    _tanh_normal_sample,
    update_actor_and_alpha,
    update_critic,
    update_high_actor,
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
    sg_actor_opt_state: Any


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


def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


def make_replay_snapshot_metadata(
    *,
    env_name: str,
    obs_size: int,
    action_size: int,
    num_envs: int,
    episode_length: int,
    subgoal_steps: int,
    goal_indices: Tuple[int, ...],
    snapshot_step: int,
) -> Dict[str, Any]:
    return {
        "env_name": env_name,
        "obs_size": int(obs_size),
        "action_size": int(action_size),
        "num_envs": int(num_envs),
        "episode_length": int(episode_length),
        "subgoal_steps": int(subgoal_steps),
        "goal_indices": tuple(int(x) for x in goal_indices),
        "snapshot_step": int(snapshot_step),
    }


def validate_replay_snapshot_metadata(
    metadata: Dict[str, Any],
    *,
    env_name: str,
    obs_size: int,
    action_size: int,
    num_envs: int,
    episode_length: int,
    subgoal_steps: int,
    goal_indices: Tuple[int, ...],
) -> None:
    expected = make_replay_snapshot_metadata(
        env_name=env_name,
        obs_size=obs_size,
        action_size=action_size,
        num_envs=num_envs,
        episode_length=episode_length,
        subgoal_steps=subgoal_steps,
        goal_indices=goal_indices,
        snapshot_step=int(metadata.get("snapshot_step", 0)),
    )
    for key, expected_value in expected.items():
        if key == "snapshot_step":
            continue
        actual_value = metadata.get(key)
        if key == "goal_indices" and actual_value is not None:
            actual_value = tuple(int(x) for x in actual_value)
        if actual_value != expected_value:
            raise ValueError(
                f"Replay snapshot metadata mismatch for {key}: "
                f"expected {expected_value}, got {actual_value}"
            )


def save_replay_snapshot(path: str, buffer_state: Any, metadata: Dict[str, Any]) -> None:
    path = epath.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fout:
        fout.write(pickle.dumps({"buffer_state": buffer_state, "metadata": metadata}))


def load_replay_snapshot(path: str) -> Tuple[Any, Dict[str, Any]]:
    with epath.Path(path).open("rb") as fin:
        snapshot = pickle.loads(fin.read())
    if "buffer_state" not in snapshot or "metadata" not in snapshot:
        raise ValueError(f"Invalid replay snapshot at {path}: expected buffer_state and metadata")
    return snapshot["buffer_state"], snapshot["metadata"]


def _pack_params(training_state: TrainingState):
    return {
        "actor": training_state.actor_state.params,
        "sg_encoder": training_state.critic_state.params["sg_encoder"],
        "a_encoder": training_state.critic_state.params["a_encoder"],
        "high_actor": training_state.high_actor_state.params,
    }


def _joint_pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Projects a set of vectors to two dimensions using a shared PCA basis."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Expected a non-empty [num_vectors, latent_dim] array.")

    centered = vectors - vectors.mean(axis=0, keepdims=True)
    num_components = min(2, centered.shape[0], centered.shape[1])
    if num_components == 0:
        return np.zeros((centered.shape[0], 2), dtype=np.float64)

    _, _, right_singular_vectors = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ right_singular_vectors[:num_components].T
    if num_components == 1:
        projection = np.pad(projection, ((0, 0), (0, 1)))
    return projection


def make_carl_representation_figure(
    state_goal_repr: np.ndarray,
    action_repr: np.ndarray,
    goal_delta: np.ndarray,
    num_links: int,
):
    """Plots matched CARL state-goal and action-sequence embeddings in one PCA space."""
    state_goal_repr = np.asarray(state_goal_repr)
    action_repr = np.asarray(action_repr)
    goal_delta = np.asarray(goal_delta)

    joint_projection = _joint_pca_2d(np.concatenate([state_goal_repr, action_repr], axis=0))
    num_pairs = state_goal_repr.shape[0]
    state_goal_2d = joint_projection[:num_pairs]
    action_2d = joint_projection[num_pairs:]

    if goal_delta.shape[-1] >= 2:
        color_values = np.arctan2(goal_delta[:, 1], goal_delta[:, 0])
        color_label = "local-goal direction (radians)"
        cmap = "twilight"
    else:
        color_values = np.linalg.norm(goal_delta, axis=-1)
        color_label = "local-goal displacement"
        cmap = "viridis"

    color_min = float(np.nanmin(color_values))
    color_max = float(np.nanmax(color_values))
    if np.isclose(color_min, color_max):
        color_max = color_min + 1.0

    positive_distance = np.linalg.norm(state_goal_repr - action_repr, axis=-1).mean()
    shuffled_distance = np.linalg.norm(state_goal_repr - np.roll(action_repr, shift=1, axis=0), axis=-1).mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    state_scatter = ax.scatter(
        state_goal_2d[:, 0],
        state_goal_2d[:, 1],
        c=color_values,
        cmap=cmap,
        vmin=color_min,
        vmax=color_max,
        marker="o",
        s=18,
        alpha=0.65,
        label=r"$\phi(s_t, s_{t+K})$",
    )
    ax.scatter(
        action_2d[:, 0],
        action_2d[:, 1],
        c=color_values,
        cmap=cmap,
        vmin=color_min,
        vmax=color_max,
        marker="x",
        s=24,
        alpha=0.65,
        label=r"$e(a_{t:t+K-1})$",
    )

    link_count = min(max(int(num_links), 0), num_pairs)
    if link_count > 0:
        link_indices = np.linspace(0, num_pairs - 1, link_count, dtype=np.int32)
        for index in link_indices:
            ax.plot(
                [state_goal_2d[index, 0], action_2d[index, 0]],
                [state_goal_2d[index, 1], action_2d[index, 1]],
                linewidth=0.5,
                alpha=0.18,
            )

    colorbar = fig.colorbar(state_scatter, ax=ax, pad=0.02)
    colorbar.set_label(color_label)
    ax.set_title(
        "CARL representation space (joint PCA)\n"
        f"positive L2 = {positive_distance:.3f}, "
        f"shuffled L2 = {shuffled_distance:.3f}"
    )
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


@dataclass
class HCARL:
    subgoal_steps: int = 25
    high_actor_hidden: Tuple[int, ...] = (512, 512, 512)
    value_hidden: Tuple[int, ...] = (512, 512, 512)
    flat_policy: bool = False
    use_split_values: bool = True
    planner_mode: PlannerMode = "none"
    planner_step_size: float = 2.0
    disable_high_actor_update_with_planner: bool = False

    replay_snapshot_path: Optional[str] = None
    save_replay_snapshot_path: Optional[str] = None
    random_snapshot_steps: int = 0
    offline_update_steps: int = 0

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

    # Replay-based CARL representation diagnostic. It is evaluated only at
    # visualization intervals and does not affect the replay sampler state.
    log_representation_space: bool = False
    representation_viz_max_points: int = 2048
    representation_viz_num_links: int = 100

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
        logging.info("HCARL log_representation_space: %s", self.log_representation_space)
        logging.info("HCARL use_split_values: %s", self.use_split_values)
        logging.info("HCARL planner_mode: %s", self.planner_mode)

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
        num_prefill_actor_steps = int(np.ceil(self.min_replay_size / self.unroll_length))
        random_snapshot_actor_steps = int(
            np.ceil(max(0, self.random_snapshot_steps) / env_steps_per_actor_step)
        )
        initial_actor_steps = (
            random_snapshot_actor_steps if random_snapshot_actor_steps > 0
            else 0 if self.replay_snapshot_path is not None
            else num_prefill_actor_steps
        )
        initial_env_steps = initial_actor_steps * env_steps_per_actor_step
        remaining_env_steps = max(1, config.total_env_steps - initial_env_steps)
        num_training_steps_per_epoch = int(
            np.ceil(remaining_env_steps / (config.num_evals * env_steps_per_actor_step))
        )
        assert num_training_steps_per_epoch > 0, "total_env_steps too small for this setup"

        logging.info("num_prefill_actor_steps: %d", num_prefill_actor_steps)
        logging.info("random_snapshot_actor_steps: %d", random_snapshot_actor_steps)
        logging.info("num_training_steps_per_epoch: %d", num_training_steps_per_epoch)

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key = jax.random.split(key, 4)

        # Keep CRL component keys together. The high key is split after those so
        # adding hierarchy perturbs CRL initialization as little as possible.
        key, sg_key, a_key, actor_key, high_key, value_low_key, value_high_key = jax.random.split(key, 7)

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
            action_size=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        value_low_module = Value(
            layer_sizes=self.value_hidden,
            use_ln=self.use_ln,
        )
        value_high_module = Value(
            layer_sizes=self.value_hidden,
            use_ln=self.use_ln,
        )
        networks = {
            "sg_encoder": sg_encoder_module,
            "a_encoder": a_encoder_module,
            "value_low_module": value_low_module,
            "value_high_module": value_high_module,
            "actor": actor_module,
            "high_actor": high_actor_module,
        }

        dummy_state = jnp.ones((1, state_size))
        dummy_goal = jnp.ones((1, goal_size))
        # CARL action encoder receives the fixed-length action sequence
        # (a_t, ..., a_{t+k-1}), flattened into one vector.
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
        # Low value is conditioned on raw local/planner subgoal coordinates, not phi(s, g).
        value_low_params = value_low_module.init(value_low_key, dummy_state, dummy_goal)
        value_high_params = value_high_module.init(value_high_key, dummy_state, dummy_goal)

        critic_state = TrainState.create(
            apply_fn=None,
            params={
                "sg_encoder": sg_params,
                "a_encoder": a_params,
                "value_low": value_low_params,
                "value_high": value_high_params,
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
        # CARL's state-goal encoder receives a second, dedicated optimizer
        # stream from the low/high AWR actor losses.  This avoids applying
        # zero actor gradients through the full critic Adam state.
        sg_actor_tx = optax.adam(learning_rate=self.critic_lr)
        sg_actor_opt_state = sg_actor_tx.init(critic_state.params["sg_encoder"])
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            target_critic_params=critic_state.params,
            alpha_state=alpha_state,
            high_actor_state=high_actor_state,
            sg_actor_opt_state=sg_actor_opt_state,
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
            use_split_values=self.use_split_values,
            p_randomgoal=self.p_randomgoal,
            p_trajgoal=self.p_trajgoal,
            p_currgoal=self.p_currgoal,
            geom_sample=self.geom_sample,
            reward_scale=self.reward_scale,
            reward_shift=self.reward_shift,
            terminal=self.terminal,
            high_p_randomgoal=self.high_p_randomgoal,
        )

        _discount = float(self.discount)
        _state_size = int(state_size)
        _goal_indices_arr = jnp.array(train_env.goal_indices)
        _goal_indices_tuple = tuple(int(x) for x in np.asarray(train_env.goal_indices))
        _subgoal_steps = int(self.subgoal_steps)
        _flat_policy = bool(self.flat_policy)
        _use_planner = self.planner_mode != "none"
        # In flat or planner mode there is no learned high policy, so the
        # high value branch has no downstream use and is disabled as well.
        crl_config["train_high_value"] = not (_flat_policy or _use_planner)
        _env_name = str(getattr(config, "env", "unknown"))

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

            # Equivalent of terminal_locs[searchsorted(...)] inside this
            # sampled trajectory chunk.
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

            goal_key, high_value_key, high_traj_key, high_random_key, high_pick_key = jax.random.split(
                sample_key, 5
            )

            # Same as GCDataset.sample(): sample relabelled goals and recompute reward/mask.
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

            high_value_goal_indx = sample_goals(
                high_value_key,
                float(self.p_randomgoal),
                float(self.p_trajgoal),
                float(self.p_currgoal),
            )
            high_value_success = (high_value_goal_indx == indx).astype(jnp.float32)
            high_value_reward = (
                high_value_success * float(self.reward_scale) + float(self.reward_shift)
            )
            high_value_mask = jnp.where(
                bool(self.terminal), 1.0 - high_value_success, jnp.ones_like(high_value_success)
            )
            high_value_goals = jnp.take(
                transition.observation, high_value_goal_indx, axis=0
            )[:, _goal_indices_arr]

            # Same as GCSDataset.sample(): low_goals = s_{t+k}.
            way_indx = jnp.minimum(indx + _subgoal_steps, final_indx)
            way_obs = jnp.take(transition.observation, way_indx, axis=0)
            low_goals = way_obs[:, _goal_indices_arr]

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

            low_success = (way_indx == indx).astype(jnp.float32)
            low_reward = low_success * float(self.reward_scale) + float(self.reward_shift)
            low_mask = jnp.where(bool(self.terminal), 1.0 - low_success, jnp.ones_like(low_success))

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

            actor_goals = jnp.where(_flat_policy, goals, low_goals)

            return Transition(
                observation=jnp.concatenate([state, actor_goals[:-1]], axis=-1),
                action=action,
                reward=reward[:-1],
                discount=mask[:-1] * transition.discount[:-1],
                extras={
                    "future_state": way_obs[:-1, :_state_size],
                    "next_state": next_state,
                    "state": state,
                    "value_goal": goals[:-1],
                    "low_value_goal": low_goals[:-1],
                    "low_value_reward": low_reward[:-1],
                    "low_value_discount": low_mask[:-1] * transition.discount[:-1],
                    "low_actor_goal": low_goals[:-1],
                    "low_actor_state": way_obs[:-1, :_state_size],
                    "action_sequence": action_seq[:-1],
                    "high_actor_goal": high_goals[:-1],
                    "high_value_goal": high_value_goals[:-1],
                    "high_value_reward": high_value_reward[:-1],
                    "high_value_discount": high_value_mask[:-1] * transition.discount[:-1],
                    "high_actor_target_goal": high_targets[:-1],
                    "high_actor_target_state": high_targets_full[:-1, :_state_size],
                    "hiql_value_goal_success": success[:-1],
                    "low_value_goal_success": low_success[:-1],
                    "high_value_goal_success": high_value_success[:-1],
                    "hiql_pick_random_high_goal": pick_random[:-1].astype(jnp.float32),
                },
            )

        def _get_action(params, state, goal, key, deterministic):
            if _flat_policy:
                # Flat mode still uses the CARL representation as the actor goal.
                actor_goal = sg_encoder_module.apply(
                    params["sg_encoder"],
                    jnp.concatenate([state, goal], axis=-1),
                )
            elif _use_planner:
                raw_subgoal = oracle_subgoal(
                    state,
                    goal,
                    _goal_indices_tuple,
                    self.planner_mode,
                    self.planner_step_size,
                )
                actor_goal = sg_encoder_module.apply(
                    params["sg_encoder"],
                    jnp.concatenate([state, raw_subgoal], axis=-1),
                )
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

        @jax.jit
        def get_random_experience(env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused_t):
                env_state, current_key = carry
                action_key, next_key = jax.random.split(current_key)
                actions = jax.random.uniform(
                    action_key,
                    shape=(config.num_envs, action_size),
                    minval=-1.0,
                    maxval=1.0,
                )
                nstate = train_env.step(env_state, actions)
                state_extras = {x: nstate.info[x] for x in ("truncation", "traj_id")}
                transition = Transition(
                    observation=env_state.obs,
                    action=actions,
                    reward=nstate.reward,
                    discount=1 - nstate.done,
                    extras={"state_extras": state_extras},
                )
                return (nstate, next_key), transition

            (env_state, _), data = jax.lax.scan(
                f, (env_state, key), (), length=self.unroll_length
            )
            buffer_state = replay_buffer.insert(buffer_state, data)
            action_abs_mean = jnp.mean(jnp.abs(data.action))
            return env_state, buffer_state, action_abs_mean

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

        def collect_random_snapshot(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, buffer_state, key = carry
                key, step_key = jax.random.split(key)
                env_state, buffer_state, action_abs_mean = get_random_experience(
                    env_state, buffer_state, step_key
                )
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (training_state, env_state, buffer_state, key), action_abs_mean

            (training_state, env_state, buffer_state, key), action_abs_mean = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=random_snapshot_actor_steps,
            )
            metrics = {
                "random_snapshot/action_abs_mean": jnp.asarray(action_abs_mean).mean(),
                "random_snapshot/buffer_size": replay_buffer.size(buffer_state),
                "random_snapshot/env_steps": training_state.env_steps,
            }
            return training_state, env_state, buffer_state, key, metrics

        @jax.jit
        def update_networks(carry, batch):
            training_state, key = carry
            key, critic_key, actor_key, high_actor_key = jax.random.split(key, 4)

            training_state, critic_metrics = update_critic(
                crl_config, networks, batch, training_state, critic_key
            )
            training_state, actor_metrics, low_actor_sg_grad = update_actor_and_alpha(
                crl_config, networks, batch, training_state, actor_key
            )

            # A planner supplies the subgoal directly, so the learned high actor
            # is not part of behavior and must not be updated in planner mode.
            high_actor_is_active = not (_flat_policy or _use_planner)
            if high_actor_is_active:
                training_state, high_actor_metrics, high_actor_sg_grad = update_high_actor(
                    crl_config, networks, batch, training_state, high_actor_key
                )
            else:
                # Do not emit placeholder high-actor losses: their presence in
                # W&B is misleading because the learned high actor is not used
                # or updated in flat/planner mode.
                high_actor_metrics = {}
                high_actor_sg_grad = jax.tree_util.tree_map(
                    jnp.zeros_like, training_state.critic_state.params["sg_encoder"]
                )

            # Apply the summed low/high actor gradient to phi(s, g) only.
            # The action-sequence encoder and both values stay on their critic
            # update path; they do not receive actor-loss gradients.
            sg_actor_grad = jax.tree_util.tree_map(
                lambda low_grad, high_grad: low_grad + high_grad,
                low_actor_sg_grad,
                high_actor_sg_grad,
            )
            sg_params = training_state.critic_state.params["sg_encoder"]
            sg_updates, sg_actor_opt_state = sg_actor_tx.update(
                sg_actor_grad, training_state.sg_actor_opt_state, sg_params
            )
            updated_sg_params = optax.apply_updates(sg_params, sg_updates)
            updated_critic_params = dict(training_state.critic_state.params)
            updated_critic_params["sg_encoder"] = updated_sg_params
            training_state = training_state.replace(
                critic_state=training_state.critic_state.replace(params=updated_critic_params),
                sg_actor_opt_state=sg_actor_opt_state,
            )
            actor_metrics["sg_encoder_actor_grad_norm"] = optax.global_norm(sg_actor_grad)
            actor_metrics["sg_encoder_low_actor_grad_norm"] = optax.global_norm(low_actor_sg_grad)
            if high_actor_is_active:
                high_actor_metrics["sg_encoder_high_actor_grad_norm"] = optax.global_norm(
                    high_actor_sg_grad
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
            metrics.update(actor_metrics)
            metrics.update(high_actor_metrics)
            metrics["planner/high_actor_updated"] = jnp.asarray(
                high_actor_is_active, dtype=jnp.float32
            )
            metrics["gradient_steps"] = training_state.gradient_steps
            return (training_state, key), metrics

        def sample_training_batches(buffer_state, permute_key, sampling_key):
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
            return buffer_state, batches

        @jax.jit
        def sample_carl_representations(critic_params, buffer_state, sample_key):
            """Samples matched CARL positives for host-side PCA visualization.

            The replay state returned by sample() is deliberately discarded so
            that enabling this diagnostic never changes subsequent training
            batches or the training random-number sequence.
            """
            _, transitions = replay_buffer.sample(buffer_state)
            relabel_key, select_key = jax.random.split(sample_key)
            batch_keys = jax.random.split(relabel_key, transitions.observation.shape[0])
            batches = jax.vmap(flatten_batch_hcrl)(transitions, batch_keys)
            batches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
                batches,
            )

            total_pairs = batches.observation.shape[0]
            num_pairs = min(int(self.representation_viz_max_points), total_pairs)
            selected = jax.random.permutation(select_key, total_pairs)[:num_pairs]

            state = batches.extras["state"][selected]
            low_goal = batches.extras["low_actor_goal"][selected]
            action_sequence = batches.extras["action_sequence"][selected]

            state_goal_repr = sg_encoder_module.apply(
                critic_params["sg_encoder"],
                jnp.concatenate([state, low_goal], axis=-1),
            )
            action_repr = a_encoder_module.apply(
                critic_params["a_encoder"],
                action_sequence,
            )
            goal_delta = low_goal - state[:, _goal_indices_arr]

            shuffled_action_repr = jnp.roll(action_repr, shift=1, axis=0)
            positive_l2 = jnp.linalg.norm(state_goal_repr - action_repr, axis=-1)
            shuffled_l2 = jnp.linalg.norm(
                state_goal_repr - shuffled_action_repr, axis=-1
            )
            cosine_similarity = jnp.sum(state_goal_repr * action_repr, axis=-1) / (
                jnp.linalg.norm(state_goal_repr, axis=-1)
                * jnp.linalg.norm(action_repr, axis=-1)
                + 1e-8
            )
            representation_metrics = {
                "carl_repr/positive_l2": jnp.mean(positive_l2),
                "carl_repr/shuffled_l2": jnp.mean(shuffled_l2),
                "carl_repr/l2_margin": jnp.mean(shuffled_l2 - positive_l2),
                "carl_repr/positive_cosine": jnp.mean(cosine_similarity),
                "carl_repr/state_goal_norm": jnp.mean(
                    jnp.linalg.norm(state_goal_repr, axis=-1)
                ),
                "carl_repr/action_sequence_norm": jnp.mean(
                    jnp.linalg.norm(action_repr, axis=-1)
                ),
            }
            return state_goal_repr, action_repr, goal_delta, representation_metrics

        @jax.jit
        def replay_update_step(training_state, buffer_state, key):
            permute_key, sampling_key, training_key = jax.random.split(key, 3)
            buffer_state, batches = sample_training_batches(buffer_state, permute_key, sampling_key)
            (training_state, _), metrics = jax.lax.scan(
                update_networks, (training_state, training_key), batches
            )
            return training_state, buffer_state, metrics

        @jax.jit
        def offline_update_step(training_state, buffer_state, key):
            permute_key, sampling_key, training_key = jax.random.split(key, 3)
            buffer_state, batches = sample_training_batches(buffer_state, permute_key, sampling_key)
            batch = jax.tree_util.tree_map(lambda x: x[0], batches)
            (training_state, _), metrics = update_networks(
                (training_state, training_key), batch
            )
            metrics["offline_updates"] = training_state.gradient_steps
            return training_state, buffer_state, metrics

        @jax.jit
        def run_offline_updates(training_state, buffer_state, key):
            def f(carry, unused):
                del unused
                training_state, buffer_state, key = carry
                key, update_key = jax.random.split(key)
                training_state, buffer_state, metrics = offline_update_step(
                    training_state, buffer_state, update_key
                )
                return (training_state, buffer_state, key), metrics

            (training_state, buffer_state, key), metrics = jax.lax.scan(
                f,
                (training_state, buffer_state, key),
                (),
                length=self.offline_update_steps,
            )
            metrics = jax.tree_util.tree_map(lambda x: jnp.asarray(x).mean(), metrics)
            return training_state, buffer_state, key, metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            experience_key, training_key = jax.random.split(key, 2)
            env_state, buffer_state = get_experience(
                training_state, env_state, buffer_state, experience_key
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            training_state, buffer_state, metrics = replay_update_step(
                training_state, buffer_state, training_key
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

        startup_metrics = {
            "training/replay_loaded": 0.0,
            "training/offline_updates": 0.0,
        }
        loaded_or_collected_replay = False

        if self.replay_snapshot_path is not None:
            loaded_buffer_state, snapshot_metadata = load_replay_snapshot(self.replay_snapshot_path)
            validate_replay_snapshot_metadata(
                snapshot_metadata,
                env_name=_env_name,
                obs_size=obs_size,
                action_size=action_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
                subgoal_steps=self.subgoal_steps,
                goal_indices=_goal_indices_tuple,
            )
            buffer_state = loaded_buffer_state
            loaded_or_collected_replay = True
            startup_metrics["training/replay_loaded"] = 1.0
            logging.info("loaded replay snapshot from %s", self.replay_snapshot_path)

        if random_snapshot_actor_steps > 0:
            key, random_snapshot_key = jax.random.split(key, 2)
            training_state, env_state, buffer_state, key, random_snapshot_metrics = (
                collect_random_snapshot(
                    training_state,
                    env_state,
                    buffer_state,
                    random_snapshot_key,
                )
            )
            loaded_or_collected_replay = True
            random_snapshot_metrics = jax.tree_util.tree_map(
                lambda x: float(jnp.asarray(x).block_until_ready()),
                random_snapshot_metrics,
            )
            startup_metrics.update(random_snapshot_metrics)
            logging.info(
                "collected random replay snapshot with %d env steps",
                int(training_state.env_steps.item()),
            )

        if not loaded_or_collected_replay:
            key, prefill_key = jax.random.split(key, 2)
            training_state, env_state, buffer_state, _ = prefill_replay_buffer(
                training_state, env_state, buffer_state, prefill_key
            )

        if self.save_replay_snapshot_path is not None:
            snapshot_metadata = make_replay_snapshot_metadata(
                env_name=_env_name,
                obs_size=obs_size,
                action_size=action_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
                subgoal_steps=self.subgoal_steps,
                goal_indices=_goal_indices_tuple,
                snapshot_step=int(training_state.env_steps.item()),
            )
            save_replay_snapshot(self.save_replay_snapshot_path, buffer_state, snapshot_metadata)
            startup_metrics["random_snapshot/snapshot_saved"] = 1.0
            logging.info("saved replay snapshot to %s", self.save_replay_snapshot_path)

        if int(jax.device_get(replay_buffer.size(buffer_state))) < config.episode_length:
            raise ValueError(
                "HCARL replay buffer does not contain enough data to sample one trajectory: "
                f"size={int(jax.device_get(replay_buffer.size(buffer_state)))}, "
                f"episode_length={config.episode_length}"
            )

        if self.offline_update_steps > 0:
            key, offline_key = jax.random.split(key, 2)
            training_state, buffer_state, key, offline_metrics = run_offline_updates(
                training_state, buffer_state, offline_key
            )
            startup_metrics["training/offline_updates"] = float(self.offline_update_steps)
            startup_metrics.update(
                {
                    f"training/offline_{name}": float(jnp.asarray(value).block_until_ready())
                    for name, value in offline_metrics.items()
                }
            )
            logging.info("completed %d offline replay updates", self.offline_update_steps)

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
                elif _use_planner:
                    raw_subgoal = oracle_subgoal(
                        s,
                        g,
                        _goal_indices_tuple,
                        self.planner_mode,
                        self.planner_step_size,
                    )
                    actor_goal = sg_encoder_module.apply(
                        param["sg_encoder"],
                        jnp.concatenate([s, raw_subgoal], axis=-1),
                    )
                else:
                    high_obs = jnp.concatenate([s, g], axis=-1)
                    actor_goal, _ = high_actor_module.apply(param["high_actor"], high_obs)
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
                **startup_metrics,
                **{f"training/{name}": value for name, value in metrics.items()},
            }

            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            do_render = ne % config.visualization_interval == 0
            make_policy = _make_policy

            if self.log_representation_space and do_render:
                # Fold in the step rather than splitting the training key, so diagnostics do not perturb the learning trajectory.
                representation_key = jax.random.fold_in(
                    jax.random.PRNGKey(config.seed),
                    current_step,
                )
                state_goal_repr, action_repr, goal_delta, representation_metrics = (
                    sample_carl_representations(
                        training_state.critic_state.params,
                        buffer_state,
                        representation_key,
                    )
                )
                representation_metrics = jax.tree_util.tree_map(lambda x: float(jnp.asarray(x).block_until_ready()), representation_metrics)
                metrics.update(representation_metrics)

                representation_figure = make_carl_representation_figure(
                    np.asarray(jax.device_get(state_goal_repr)),
                    np.asarray(jax.device_get(action_repr)),
                    np.asarray(jax.device_get(goal_delta)),
                    self.representation_viz_num_links,
                )
                wandb.log(
                    {
                        "carl_representation_space": wandb.Image(
                            representation_figure
                        )
                    },
                    step=current_step,
                )
                plt.close(representation_figure)

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