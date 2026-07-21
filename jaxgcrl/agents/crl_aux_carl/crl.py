import functools
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
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from .losses import (
    update_actor_and_alpha,
    update_backward_dynamics,
    update_carl_aux,
    update_critic,
    update_forward_dynamics,
)
from .networks import Actor, BackwardDynamics, Encoder, ForwardDynamics

Metrics = types.Metrics
Env = Union[envs.Env, envs.Wrapper]
State = envs.State


@dataclass
class TrainingState:
    """Contains training state for the learner"""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
    carl_state: TrainState
    fd_state: TrainState
    bd_state: TrainState


class Transition(NamedTuple):
    """Container for a transition"""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


class CARLPairs(NamedTuple):
    """Real CARL anchors plus backward-generated, forward-validated pairs."""

    state: jnp.ndarray
    goal: jnp.ndarray
    real_action_sequence: jnp.ndarray
    generated_action_sequence: jnp.ndarray
    negative_action_sequence: jnp.ndarray
    generated_positive_mask: jnp.ndarray
    negative_mask: jnp.ndarray
    valid_mask: jnp.ndarray


def create_carl_pairs(
    backward_dynamics,
    bd_params,
    forward_dynamics,
    fd_params,
    transitions,
    key,
    positive_margin: float,
    negative_margin: float,
    proposal_noise_scale: float,
):
    """Proposal-verification pair construction for CARL.

    1) (s, g, A_real) is always a real positive from replay.
    2) q(A | s, g) proposes A_gen. F(s, A_gen) must predict g before it
       becomes an additional positive.
    3) A validated proposal from another row is borrowed as A_neg. It becomes
       a negative only if F(s, A_neg) predicts an outcome clearly away from g.
    """
    state = transitions.extras["state"]
    goal = transitions.extras["carl_goal"]
    real_action = transitions.extras["action_sequence"]
    valid_mask = transitions.extras["carl_valid"]

    mean, log_std = backward_dynamics.apply(bd_params, state, goal)
    noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
    generated_action = jnp.clip(
        mean + float(proposal_noise_scale) * jnp.exp(log_std) * noise,
        -1.0,
        1.0,
    )

    generated_goal = forward_dynamics.apply(fd_params, state, generated_action)
    generated_distance = jnp.linalg.norm(generated_goal - goal, axis=-1)
    generated_positive_mask = (generated_distance <= positive_margin) & valid_mask

    # Only lend a model-generated action to another pair if it passed its own
    # forward check. Otherwise lend the real replay action from that row.
    verified_source_action = jnp.where(
        generated_positive_mask[:, None], generated_action, real_action
    )
    negative_action = jnp.roll(verified_source_action, shift=1, axis=0)
    negative_goal = forward_dynamics.apply(fd_params, state, negative_action)
    negative_distance = jnp.linalg.norm(negative_goal - goal, axis=-1)
    source_valid = jnp.roll(valid_mask, shift=1, axis=0)
    negative_mask = (negative_distance >= negative_margin) & valid_mask & source_valid

    return CARLPairs(
        state=jax.lax.stop_gradient(state),
        goal=jax.lax.stop_gradient(goal),
        real_action_sequence=jax.lax.stop_gradient(real_action),
        generated_action_sequence=jax.lax.stop_gradient(generated_action),
        negative_action_sequence=jax.lax.stop_gradient(negative_action),
        generated_positive_mask=jax.lax.stop_gradient(generated_positive_mask),
        negative_mask=jax.lax.stop_gradient(negative_mask),
        valid_mask=jax.lax.stop_gradient(valid_mask),
    )


# The planner is kept in this file so this CRL + auxiliary-CARL agent can run
# without requiring the hierarchical HCARL module.  It is an oracle local-goal
# controller for Ant-style two-dimensional goals, not a learned high actor.
PlannerMode = Literal["none", "ant_xy_oracle"]


def validate_planner_config(
    planner_mode: PlannerMode,
    *,
    goal_indices: Tuple[int, ...],
    goal_size: int,
) -> None:
    if planner_mode not in ("none", "ant_xy_oracle"):
        raise ValueError(
            f"Unknown planner_mode={planner_mode!r}. "
            "Expected 'none' or 'ant_xy_oracle'."
        )
    if planner_mode == "ant_xy_oracle":
        if goal_size != 2 or len(goal_indices) != 2:
            raise ValueError(
                "planner_mode='ant_xy_oracle' requires exactly two goal "
                f"coordinates, got goal_size={goal_size}, goal_indices={goal_indices}."
            )


def oracle_subgoal(
    state: jnp.ndarray,
    final_goal: jnp.ndarray,
    goal_indices: Tuple[int, ...],
    planner_mode: PlannerMode,
    planner_step_size: float,
) -> jnp.ndarray:
    """Returns a straight-line, bounded-distance XY waypoint.

    The function is JAX-compatible and therefore can be called inside rollout
    and evaluation JITs.  For ``planner_mode='none'`` it simply returns the
    original final goal.
    """
    if planner_mode == "none":
        return final_goal

    state_xy = jnp.take(
        state,
        jnp.asarray(goal_indices, dtype=jnp.int32),
        axis=-1,
    )
    direction = final_goal - state_xy
    distance = jnp.linalg.norm(direction, axis=-1, keepdims=True)
    step_fraction = jnp.minimum(
        1.0,
        float(planner_step_size) / jnp.maximum(distance, 1e-6),
    )
    return state_xy + step_fraction * direction


def planner_metrics(
    state: jnp.ndarray,
    final_goal: jnp.ndarray,
    subgoal: jnp.ndarray,
    goal_indices: Tuple[int, ...],
) -> dict[str, jnp.ndarray]:
    """Summarizes the oracle planner's current waypoint geometry."""
    state_xy = jnp.take(
        state,
        jnp.asarray(goal_indices, dtype=jnp.int32),
        axis=-1,
    )
    final_distance = jnp.linalg.norm(final_goal - state_xy, axis=-1)
    subgoal_distance = jnp.linalg.norm(subgoal - state_xy, axis=-1)
    return {
        "planner/final_goal_distance": jnp.mean(final_distance),
        "planner/subgoal_distance": jnp.mean(subgoal_distance),
        "planner/subgoal_fraction": jnp.mean(
            subgoal_distance / jnp.maximum(final_distance, 1e-6)
        ),
    }


def _joint_pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Projects vectors to a common two-dimensional PCA coordinate system."""
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


def make_alignment_figure(
    left_repr: np.ndarray,
    right_repr: np.ndarray,
    goal_delta: np.ndarray,
    *,
    title: str,
    left_label: str,
    right_label: str,
    num_links: int,
):
    """Plots two matched embedding sets in one PCA space.

    Each circle and cross with the same color is a positive contrastive pair.
    The faint connecting segments make pair alignment visible without changing
    training or replay sampling.
    """
    left_repr = np.asarray(left_repr)
    right_repr = np.asarray(right_repr)
    goal_delta = np.asarray(goal_delta)

    joint_projection = _joint_pca_2d(np.concatenate([left_repr, right_repr], axis=0))
    num_pairs = left_repr.shape[0]
    left_2d = joint_projection[:num_pairs]
    right_2d = joint_projection[num_pairs:]

    if goal_delta.shape[-1] >= 2:
        color_values = np.arctan2(goal_delta[:, 1], goal_delta[:, 0])
        color_label = "goal direction (radians)"
        cmap = "twilight"
    else:
        color_values = np.linalg.norm(goal_delta, axis=-1)
        color_label = "goal displacement"
        cmap = "viridis"

    color_min = float(np.nanmin(color_values))
    color_max = float(np.nanmax(color_values))
    if np.isclose(color_min, color_max):
        color_max = color_min + 1.0

    positive_l2 = float(np.linalg.norm(left_repr - right_repr, axis=-1).mean())
    shuffled_l2 = float(
        np.linalg.norm(left_repr - np.roll(right_repr, shift=1, axis=0), axis=-1).mean()
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    left_scatter = ax.scatter(
        left_2d[:, 0],
        left_2d[:, 1],
        c=color_values,
        cmap=cmap,
        vmin=color_min,
        vmax=color_max,
        marker="o",
        s=18,
        alpha=0.65,
        label=left_label,
    )
    ax.scatter(
        right_2d[:, 0],
        right_2d[:, 1],
        c=color_values,
        cmap=cmap,
        vmin=color_min,
        vmax=color_max,
        marker="x",
        s=24,
        alpha=0.65,
        label=right_label,
    )

    link_count = min(max(int(num_links), 0), num_pairs)
    if link_count > 0:
        link_indices = np.linspace(0, num_pairs - 1, link_count, dtype=np.int32)
        for index in link_indices:
            ax.plot(
                [left_2d[index, 0], right_2d[index, 0]],
                [left_2d[index, 1], right_2d[index, 1]],
                linewidth=0.5,
                alpha=0.18,
            )

    colorbar = fig.colorbar(left_scatter, ax=ax, pad=0.02)
    colorbar.set_label(color_label)
    ax.set_title(
        f"{title} (joint PCA)\n"
        f"positive L2 = {positive_l2:.3f}, shuffled L2 = {shuffled_l2:.3f}"
    )
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


@functools.partial(jax.jit, static_argnames=("buffer_config"))
def flatten_batch(buffer_config, transition, sample_key):
    gamma, state_size, goal_indices, carl_subgoal_steps = buffer_config

    # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32
    )  # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
    #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
    #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
    #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
    #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
    #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
    # assuming seq_len = 5
    # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )
    # array of seq_len x seq_len where a row is an array of traj_ids that correspond to the episode index from which that time-step was collected
    # timesteps collected from the same episode will have the same traj_id. All rows of the single_trajectories are same.

    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    # ith row of probs will be non zero only for time indices that
    # 1) are greater than i
    # 2) have the same traj_id as the ith time index

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(
        transition.observation, goal_index[:-1], axis=0
    )  # the last goal_index cannot be considered as there is no future.
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    future_goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]
    state = transition.observation[:-1, :state_size]  # all states are considered

    same_traj = jnp.equal(single_trajectories, single_trajectories.T)
    big_neg = jnp.where(same_traj, arrangement[None, :], -1)
    final_idx = jnp.max(big_neg, axis=1)
    current_idx = arrangement[:-1]
    carl_valid = current_idx + carl_subgoal_steps <= final_idx[:-1]
    short_goal_idx = jnp.minimum(current_idx + carl_subgoal_steps, final_idx[:-1])
    local_goal = transition.observation[short_goal_idx][:, goal_indices]

    # Train both the actor and the CRL critic on the same achieved K-step goal.
    # The oracle planner is used only during collection and evaluation.
    new_obs = jnp.concatenate([state, local_goal], axis=1)

    action_offsets = jnp.arange(carl_subgoal_steps)
    action_idx = jnp.minimum(
        current_idx[:, None] + action_offsets[None, :],
        jnp.maximum(final_idx[:-1, None] - 1, current_idx[:, None]),
    )
    action_idx = jnp.minimum(action_idx, seq_len - 2)
    action_sequence = jnp.take(transition.action, action_idx, axis=0)
    action_sequence = jnp.reshape(action_sequence, (action_sequence.shape[0], -1))

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
        },
        "state": state,
        "future_state": future_state,
        "future_goal": future_goal,
        "future_action": future_action,
        "local_goal": local_goal,
        "carl_goal": local_goal,
        "carl_valid": carl_valid,
        "action_sequence": action_sequence,
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),  # this has shape (num_envs, episode_length-1, obs_size)
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


@dataclass
class CRLAuxCARL:
    """CRL control with a separately trained CARL state-goal representation."""

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    carl_lr: float = 3e-4
    fd_lr: float = 3e-4
    bd_lr: float = 3e-4
    batch_size: int = 256

    # gamma
    discounting: float = 0.99

    # forward CRL logsumexp penalty
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

    # phi(s,a) and psi(g) repr dimension
    repr_dim: int = 64
    carl_repr_dim: int = 64
    carl_subgoal_steps: int = 25
    carl_positive_margin: float = 0.10
    carl_negative_margin: float = 0.25
    carl_proposal_noise_scale: float = 1.0
    carl_generated_weight: float = 0.25

    # layer norm
    use_ln: bool = False

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    # Optional oracle planner.  It replaces the final environment goal only when feeding the CRL actor during collection/evaluation.
    planner_mode: PlannerMode = "none"
    planner_step_size: float = 2.0

    # Replay-based CRL/CARL representation diagnostics.  They run only at rendering intervals and never update the replay sampler state.
    log_representation_space: bool = False
    representation_viz_max_points: int = 2048
    representation_viz_num_links: int = 100

    # State coverage visualization
    log_state_coverage: bool = False
    state_coverage_xy_dims: Tuple[int, int] = (0, 1)

    def check_config(self, config):
        """
        episode_length: the maximum length of an episode
            NOTE: `num_envs * (episode_length - 1)` must be divisible by
            `batch_size` due to the way data is stored in replay buffer.
        """
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )

    def train_fn(
        self,
        config: "RunConfig",
        train_env: Env,
        eval_env: Optional[Env] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        self.check_config(config)
        logging.info("CRL auxiliary CARL planner_mode: %s", self.planner_mode)
        logging.info("CRL low actor input: [state, CARL phi(state, goal)]")
        logging.info("CRL auxiliary CARL state coverage: %s", self.log_state_coverage)
        logging.info(
            "CRL auxiliary CARL representation diagnostics: %s",
            self.log_representation_space,
        )

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

        assert num_training_steps_per_epoch > 0, (
            "total_env_steps too small for given num_envs and episode_length"
        )

        logging.info(
            "num_prefill_env_steps: %d",
            num_prefill_env_steps,
        )
        logging.info(
            "num_prefill_actor_steps: %d",
            num_prefill_actor_steps,
        )
        logging.info(
            "num_training_steps_per_epoch: %d",
            num_training_steps_per_epoch,
        )

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        (
            key,
            buffer_key,
            eval_env_key,
            env_key,
            actor_key,
            sa_key,
            g_key,
            sg_key,
            a_key,
            fd_key,
            bd_key,
        ) = jax.random.split(key, 11)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        # Dimensions definitions and sanity checks
        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )

        _goal_indices_tuple = tuple(int(x) for x in np.asarray(train_env.goal_indices))
        _goal_indices_arr = jnp.asarray(_goal_indices_tuple, dtype=jnp.int32)
        _use_planner = self.planner_mode != "none"
        validate_planner_config(
            self.planner_mode,
            goal_indices=_goal_indices_tuple,
            goal_size=goal_size,
        )

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

        # Network setup
        # Actor
        actor = Actor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        # The flat SAC/CRL actor is conditioned on raw state, raw local goal,
        # and the CARL state-goal representation.
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(
                actor_key,
                np.ones([1, state_size + goal_size + self.carl_repr_dim]),
            ),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        # Critic
        sa_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, state_size + action_size]))
        g_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        g_encoder_params = g_encoder.init(g_key, np.ones([1, goal_size]))
        critic_state = TrainState.create(
            apply_fn=None,
            params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        # Entropy coefficient
        target_entropy = -0.5 * action_size
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        # CARL learns a state-goal/action-sequence reachability representation.
        # The actor consumes the state-goal representation but does not backpropagate
        # into either CARL encoder.
        sg_encoder = Encoder(
            repr_dim=self.carl_repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        a_encoder = Encoder(
            repr_dim=self.carl_repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        carl_state = TrainState.create(
            apply_fn=None,
            params={
                "sg_encoder": sg_encoder.init(sg_key, np.ones([1, state_size + goal_size])),
                "a_encoder": a_encoder.init(a_key, np.ones([1, self.carl_subgoal_steps * action_size])),
            },
            tx=optax.adam(learning_rate=self.carl_lr),
        )

        forward_dynamics = ForwardDynamics(
            goal_size=goal_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            use_relu=self.use_relu,
        )
        fd_state = TrainState.create(
            apply_fn=forward_dynamics.apply,
            params=forward_dynamics.init(
                fd_key,
                np.ones([1, state_size]),
                np.ones([1, self.carl_subgoal_steps * action_size]),
            ),
            tx=optax.adam(learning_rate=self.fd_lr),
        )

        backward_dynamics = BackwardDynamics(
            action_sequence_size=self.carl_subgoal_steps * action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            use_relu=self.use_relu,
        )
        bd_state = TrainState.create(
            apply_fn=backward_dynamics.apply,
            params=backward_dynamics.init(
                bd_key,
                np.ones([1, state_size]),
                np.ones([1, goal_size]),
            ),
            tx=optax.adam(learning_rate=self.bd_lr),
        )

        # Trainstate
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
            carl_state=carl_state,
            fd_state=fd_state,
            bd_state=bd_state,
        )

        # Replay Buffer
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

        def _planned_actor_observation(raw_obs, sg_encoder_params):
            """Builds [state, local goal, phi_CARL(state, local goal)] for the flat actor.

            With ``planner_mode='ant_xy_oracle'`` the local goal is the oracle
            XY waypoint.  There is no high actor in this CRL + CARL-aux agent,
            so no high-level parameters are used or updated.
            """
            state = raw_obs[:, :state_size]
            final_goal = raw_obs[:, state_size:]
            actor_goal = oracle_subgoal(state, final_goal, _goal_indices_tuple, self.planner_mode, self.planner_step_size)
            sg_repr = sg_encoder.apply(
                sg_encoder_params,
                jnp.concatenate([state, actor_goal], axis=-1),
            )
            return jnp.concatenate([state, actor_goal, sg_repr], axis=-1)

        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            actor_obs = _planned_actor_observation(
                env_state.obs,
                training_state.carl_state.params["sg_encoder"],
            )
            means, _ = actor.apply(training_state.actor_state.params, actor_obs)
            actions = nn.tanh(means)

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}

            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def actor_step(actor_state, carl_state, env, env_state, key, extra_fields):
            actor_obs = _planned_actor_observation(
                env_state.obs,
                carl_state.params["sg_encoder"],
            )
            means, log_stds = actor.apply(actor_state.params, actor_obs)
            stds = jnp.exp(log_stds)
            actions = nn.tanh(
                means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
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
        def get_experience(actor_state, carl_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused_t):
                env_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                env_state, transition = actor_step(
                    actor_state,
                    carl_state,
                    train_env,
                    env_state,
                    current_key,
                    extra_fields=("truncation", "traj_id"),
                )
                return (env_state, next_key), transition

            (env_state, _), data = jax.lax.scan(
                f,
                (env_state, key),
                (),
                length=self.unroll_length,
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
                    training_state.actor_state,
                    training_state.carl_state,
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

        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            key, critic_key, actor_key = jax.random.split(key, 3)

            context = dict(
                **vars(self),
                **vars(config),
                state_size=state_size,
                action_size=action_size,
                goal_size=goal_size,
                obs_size=obs_size,
                goal_indices=train_env.goal_indices,
                target_entropy=target_entropy,
            )

            networks = dict(
                actor=actor,
                sa_encoder=sa_encoder,
                g_encoder=g_encoder,
                sg_encoder=sg_encoder,
                a_encoder=a_encoder,
                forward_dynamics=forward_dynamics,
                backward_dynamics=backward_dynamics,
            )

            # 1) Learn both directions of the local K-step dynamics from real
            #    replay tuples only.
            training_state, fd_metrics = update_forward_dynamics(
                context, networks, transitions, training_state
            )
            training_state, bd_metrics = update_backward_dynamics(
                context, networks, transitions, training_state
            )

            # 2) Backward model proposes A ~ q(A | s, g). Forward model verifies
            #    generated positives and verifies borrowed proposals as negatives.
            key, proposal_key = jax.random.split(key)
            carl_pairs = create_carl_pairs(
                backward_dynamics,
                training_state.bd_state.params,
                forward_dynamics,
                training_state.fd_state.params,
                transitions,
                proposal_key,
                self.carl_positive_margin,
                self.carl_negative_margin,
                self.carl_proposal_noise_scale,
            )
            training_state, carl_metrics = update_carl_aux(
                context, networks, carl_pairs, training_state
            )

            # 3) The actor consumes CARL features, but actor gradients do not
            #    update the CARL encoder.
            training_state, actor_metrics = update_actor_and_alpha(
                context, networks, transitions, training_state, actor_key
            )
            training_state, critic_metrics = update_critic(
                context, networks, transitions, training_state, critic_key
            )
            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

            metrics = {}
            metrics.update(actor_metrics)
            metrics.update(critic_metrics)
            metrics.update(carl_metrics)
            metrics.update(fd_metrics)
            metrics.update(bd_metrics)
            if _use_planner:
                state = transitions.extras["state"]
                local_goal = transitions.observation[:, state_size:]
                current_goal_state = state[:, _goal_indices_arr]
                metrics["planner/training_local_goal_distance"] = jnp.mean(
                    jnp.linalg.norm(local_goal - current_goal_state, axis=-1)
                )
                metrics["planner/configured_step_size"] = jnp.asarray(
                    self.planner_step_size, dtype=jnp.float32
                )

            return (training_state, key), metrics

        @jax.jit
        def sample_representation_pairs(
            critic_params,
            carl_params,
            buffer_state,
            sample_key,
        ):
            """Reads matched CRL and CARL positives without changing replay state."""
            _, raw_transitions = replay_buffer.sample(buffer_state)
            relabel_key, select_key = jax.random.split(sample_key)
            batch_keys = jax.random.split(
                relabel_key,
                raw_transitions.observation.shape[0],
            )
            batches = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                (
                    self.discounting,
                    state_size,
                    _goal_indices_tuple,
                    self.carl_subgoal_steps,
                ),
                raw_transitions,
                batch_keys,
            )
            batches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
                batches,
            )

            total_pairs = batches.observation.shape[0]
            num_pairs = min(int(self.representation_viz_max_points), total_pairs)
            selected = jax.random.permutation(select_key, total_pairs)[:num_pairs]

            state = batches.extras["state"][selected]
            crl_goal = batches.observation[selected, state_size:]
            action = batches.action[selected]
            carl_goal = batches.extras["carl_goal"][selected]
            action_sequence = batches.extras["action_sequence"][selected]

            crl_sa_repr = sa_encoder.apply(
                critic_params["sa_encoder"],
                jnp.concatenate([state, action], axis=-1),
            )
            crl_goal_repr = g_encoder.apply(
                critic_params["g_encoder"],
                crl_goal,
            )
            carl_sg_repr = sg_encoder.apply(
                carl_params["sg_encoder"],
                jnp.concatenate([state, carl_goal], axis=-1),
            )
            carl_action_repr = a_encoder.apply(
                carl_params["a_encoder"],
                action_sequence,
            )

            crl_goal_delta = crl_goal - state[:, _goal_indices_arr]
            carl_goal_delta = carl_goal - state[:, _goal_indices_arr]

            def alignment_metrics(left, right, prefix):
                shuffled_right = jnp.roll(right, shift=1, axis=0)
                positive_l2 = jnp.linalg.norm(left - right, axis=-1)
                shuffled_l2 = jnp.linalg.norm(left - shuffled_right, axis=-1)
                cosine = jnp.sum(left * right, axis=-1) / (
                    jnp.linalg.norm(left, axis=-1)
                    * jnp.linalg.norm(right, axis=-1)
                    + 1e-8
                )
                return {
                    f"{prefix}/positive_l2": jnp.mean(positive_l2),
                    f"{prefix}/shuffled_l2": jnp.mean(shuffled_l2),
                    f"{prefix}/l2_margin": jnp.mean(shuffled_l2 - positive_l2),
                    f"{prefix}/positive_cosine": jnp.mean(cosine),
                    f"{prefix}/left_norm": jnp.mean(jnp.linalg.norm(left, axis=-1)),
                    f"{prefix}/right_norm": jnp.mean(jnp.linalg.norm(right, axis=-1)),
                }

            representation_metrics = {}
            representation_metrics.update(
                alignment_metrics(crl_sa_repr, crl_goal_repr, "crl_repr")
            )
            representation_metrics.update(
                alignment_metrics(carl_sg_repr, carl_action_repr, "carl_repr")
            )
            return crl_sa_repr, crl_goal_repr, crl_goal_delta, carl_sg_repr, carl_action_repr, carl_goal_delta, representation_metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)

            # update buffer
            env_state, buffer_state = get_experience(
                training_state.actor_state,
                training_state.carl_state,
                env_state,
                buffer_state,
                experience_key1,
            )

            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            # sample actor-step worth of transitions
            buffer_state, transitions = replay_buffer.sample(buffer_state)

            # process transitions for training
            batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
            transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                (
                    self.discounting,
                    state_size,
                    tuple(np.asarray(train_env.goal_indices)),
                    self.carl_subgoal_steps,
                ),
                transitions,
                batch_keys,
            )
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )

            # permute transitions
            permutation = jax.random.permutation(experience_key2, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                transitions,
            )

            # take actor-step worth of training-step
            (
                (
                    training_state,
                    _,
                ),
                metrics,
            ) = jax.lax.scan(update_networks, (training_state, training_key), transitions)

            return (
                training_state,
                env_state,
                buffer_state,
            ), metrics

        @jax.jit
        def training_epoch(
            training_state,
            env_state,
            buffer_state,
            key,
        ):
            @jax.jit
            def f(carry, unused_t):
                ts, es, bs, k = carry
                k, train_key = jax.random.split(k, 2)
                (
                    (
                        ts,
                        es,
                        bs,
                    ),
                    metrics,
                ) = training_step(ts, es, bs, train_key)
                return (ts, es, bs, k), metrics

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )

            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            return training_state, env_state, buffer_state, metrics

        key, prefill_key = jax.random.split(key, 2)

        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        """Setting up evaluator"""
        evaluator = ActorEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        training_walltime = 0
        logging.info("starting training....")
        for ne in range(config.num_evals):
            t = time.time()

            key, epoch_key = jax.random.split(key)

            training_state, env_state, buffer_state, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key
            )

            # accumulate current env positions
            if _coverage_history is not None:
                # env_state.obs shape: (local_devices, num_envs_per_device, obs_size)
                obs_np = np.array(jax.device_get(env_state.obs)).reshape(-1, obs_size)
                _coverage_history.append(obs_np[:, [_cov_xy0, _cov_xy1]])

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

            def make_policy(policy_params):
                def policy(obs, rng):
                    del rng
                    actor_obs = _planned_actor_observation(
                        obs,
                        policy_params["sg_encoder"],
                    )
                    return actor.apply(policy_params["actor"], actor_obs)
                return policy

            policy_params = {
                "actor": training_state.actor_state.params,
                "sg_encoder": training_state.carl_state.params["sg_encoder"],
            }
            params = (
                training_state.alpha_state.params,
                training_state.actor_state.params,
                training_state.critic_state.params,
                training_state.carl_state.params,
                training_state.fd_state.params,
                training_state.bd_state.params,
            )

            if self.log_representation_space and do_render:
                # A folded-in key and the discarded replay sampler state ensure
                # diagnostics cannot alter future training batches or RNG use.
                representation_key = jax.random.fold_in(
                    jax.random.PRNGKey(config.seed),
                    current_step,
                )
                crl_sa_repr, crl_goal_repr, crl_goal_delta, carl_sg_repr, carl_action_repr, carl_goal_delta, representation_metrics = sample_representation_pairs(
                    training_state.critic_state.params,
                    training_state.carl_state.params,
                    buffer_state,
                    representation_key,
                )
                representation_metrics = jax.tree_util.tree_map(
                    lambda x: float(jnp.asarray(x).block_until_ready()),
                    representation_metrics,
                )
                metrics.update(representation_metrics)

                crl_figure = make_alignment_figure(
                    np.asarray(jax.device_get(crl_sa_repr)),
                    np.asarray(jax.device_get(crl_goal_repr)),
                    np.asarray(jax.device_get(crl_goal_delta)),
                    title="CRL state-action / goal representation space",
                    left_label=r"$f(s_t, a_t)$",
                    right_label=r"$g(g_t)$",
                    num_links=self.representation_viz_num_links,
                )
                carl_figure = make_alignment_figure(
                    np.asarray(jax.device_get(carl_sg_repr)),
                    np.asarray(jax.device_get(carl_action_repr)),
                    np.asarray(jax.device_get(carl_goal_delta)),
                    title="Auxiliary CARL state-goal / action-sequence space",
                    left_label=r"$\phi(s_t, s_{t+K})$",
                    right_label=r"$e(a_{t:t+K-1})$",
                    num_links=self.representation_viz_num_links,
                )
                wandb.log(
                    {
                        "crl_representation_space": wandb.Image(crl_figure),
                        "carl_representation_space": wandb.Image(carl_figure),
                    },
                    step=current_step,
                )
                plt.close(crl_figure)
                plt.close(carl_figure)

            if _coverage_history:
                all_pos = np.concatenate(_coverage_history, axis=0)
                bins = 50
                h, xedges, yedges = np.histogram2d(
                    all_pos[:, 0], all_pos[:, 1], bins=bins
                )
                
                # Coverage entropy: higher = more uniform exploration.
                p = h / max(float(h.sum()), 1.0)
                p_nz = p[p > 0]
                coverage_entropy = float(-np.sum(p_nz * np.log(p_nz)))
                occupied_cells = int(np.sum(h > 0))
                metrics["state_coverage_entropy"] = coverage_entropy
                metrics["state_coverage_cells"] = occupied_cells

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
                policy_params,
                unwrapped_env,
                do_render=do_render,
            )

            if config.checkpoint_logdir:
                # Save current policy and critic params.
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

        total_steps = current_step
        assert total_steps >= config.total_env_steps

        logging.info("total steps: %s", total_steps)

        return make_policy, params, metrics