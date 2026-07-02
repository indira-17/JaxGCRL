import logging
from typing import Optional, Union, Tuple, Callable, NamedTuple

import jax
import jax.numpy as jnp

from brax import base, envs
from brax.training import types
from brax.training.acme.types import NestedArray
from brax.training.acme import running_statistics

from flax import nnx
from flax.struct import dataclass

from jaxgcrl.agents.planner import PlannerMode, validate_planner_config
from jaxgcrl.envs.wrappers import TrajectoryIdWrapper

Metrics = types.Metrics
Env = Union[envs.Env, envs.Wrapper]
State = envs.State

class GCTransition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    goal: NestedArray
    action: NestedArray # Should be temporally extended
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray

@dataclass
class HAC:
    """Hierarchical Actor-Critic (HAC) agent."""

    # Hierarchy Hyperparams
    k_step: int = 25
    num_levels: int = 2
    subgoal_testing_rate: float = 0.2
    enable_temporal_abstraction: bool = True
    use_high_level_target_networks: bool = False

    # Sublevel Hyperparams
    learning_rate: float = 1e-4
    discounting: float = 0.9
    batch_size: int = 256
    normalize_observations: bool = True
    reward_scaling: float = 1.0
    # target update rate
    tau: float = 0.005
    min_replay_size: int = 0
    max_replay_size: Optional[int] = 10000
    deterministic_eval: bool = False
    train_step_multiplier: int = 1
    unroll_length: int = 50
    h_dim: int = 256
    n_hidden: int = 2
    # layer norm
    use_ln: bool = True
    # exploration
    random_action_epsilon: float = 0.2
    random_action_noise: float = 0.1
    planner_mode: PlannerMode = "none"
    planner_step_size: float = 2.0
    disable_high_actor_update_with_planner: bool = False


def train_fn(
    self,
    config: "RunConfig",
    train_env: Env,
    eval_env: Optional[Env] = None,
    randomization_fn: Optional[Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]] = None,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
):
    from jaxgcrl.agents.hac.networks import make_networks_and_buffers

    if self.min_replay_size >= config.total_env_steps:
        raise ValueError("No training will happen because min_replay_size >= total_env_steps")

    if self.max_replay_size is None:
        max_replay_size = config.total_env_steps
    else:
        max_replay_size = self.max_replay_size

    # The number of environment steps executed for every `actor_step()` call.
    env_steps_per_actor_step = config.action_repeat * config.num_envs * self.unroll_length
    num_prefill_actor_steps = self.min_replay_size // self.unroll_length + 1
    logging.info("Num_prefill_actor_steps: %s", num_prefill_actor_steps)
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert config.total_env_steps - self.min_replay_size >= 0
    num_evals_after_init = max(config.num_evals - 1, 1)
    # The number of epoch calls per training
    # equals to
    # ceil(total_env_steps - num_prefill_env_steps /
    #      (num_evals_after_init * env_steps_per_actor_step))
    num_training_steps_per_epoch = -(
        -(config.total_env_steps - num_prefill_env_steps)
        // (num_evals_after_init * env_steps_per_actor_step)
    )

    logging.info("num_evals_after_init: %s", num_evals_after_init)
    logging.info("num_training_steps_per_epoch: %s", num_training_steps_per_epoch)

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

    rng = nnx.Rngs(
        replay_buffer=config.seed,
        policy=config.seed,
        loss=config.seed,
    )

    obs_size = unwrapped_env.observation_size
    action_size = unwrapped_env.action_size
    goal_indices = tuple(int(x) for x in getattr(unwrapped_env, "goal_indices", (0, 1)))
    validate_planner_config(
        self.planner_mode,
        goal_indices=goal_indices,
        goal_size=len(goal_indices),
    )

    hac_agent, replay_buffers = make_networks_and_buffers(
        hac_config=self,
        run_config=config,
        observation_size=obs_size,
        action_size=action_size,
        subgoal_size=obs_size,
        rngs=rng
    )
