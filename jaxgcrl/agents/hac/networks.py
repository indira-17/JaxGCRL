from typing import TYPE_CHECKING, Any, Sequence

import jax
import jax.numpy as jnp
from brax.training import networks, types
from brax.training.acme import running_statistics, specs
from brax.training.networks import ActivationFn, Initializer
from flax import nnx

from jaxgcrl.agents.planner import oracle_subgoal
from jaxgcrl.agents.hac.hac import GCTransition, HAC
from jaxgcrl.utils.replay_buffer_nnx import TrajectoryUniformSamplingQueueNNX

if TYPE_CHECKING:
    from jaxgcrl.utils.config import RunConfig


def _polyak_average(target_state, source_state, tau: float):
    return jax.tree.map(lambda target, source: (1.0 - tau) * target + tau * source, target_state, source_state)


class ObservationPreprocessor(nnx.Module):
    """Stateful observation preprocessing wrapper."""

    def __init__(
        self,
        observation_size: int,
        *,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        dtype: jnp.dtype = jnp.dtype("float32"),
    ):
        self._observation_size = observation_size
        self._preprocess_observations_fn = preprocess_observations_fn
        self.normalizer_params = nnx.BatchStat(running_statistics.init_state(specs.Array((observation_size,), dtype)))

    def __call__(
        self,
        observations: types.Observation,
        *,
        update_stats: bool = False,
    ) -> types.Observation:
        if update_stats:
            self.normalizer_params.set_value(
                running_statistics.update(
                    self.normalizer_params.get_value(),
                    observations,
                )
            )

        return self._preprocess_observations_fn(observations, self.normalizer_params.get_value())


class MLP(nnx.Module):
    """MLP module with explicit input size."""

    def __init__(
        self,
        input_size: int,
        layer_sizes: Sequence[int],
        rngs: nnx.Rngs,
        activation: ActivationFn = nnx.relu,
        kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
        activate_final: bool = False,
        bias: bool = True,
        layer_norm: bool = False,
    ):
        if not layer_sizes:
            raise ValueError("layer_sizes must be non-empty")

        layers = []
        in_dim = input_size
        for i, hidden_dim in enumerate(layer_sizes):
            layers.append(
                nnx.Linear(
                    in_features=in_dim,
                    out_features=hidden_dim,
                    kernel_init=kernel_init,
                    use_bias=bias,
                    rngs=rngs,
                )
            )
            if i != len(layer_sizes) - 1 or activate_final:
                if layer_norm:
                    layers.append(nnx.LayerNorm(hidden_dim, rngs=rngs))
                layers.append(activation)
            in_dim = hidden_dim

        self.mlp = nnx.Sequential(*layers)

    def __call__(self, data: jax.Array) -> jax.Array:
        return self.mlp(data)


class GoalConditionedPolicyNetwork(nnx.Module):
    """Deterministic actor that conditions on observation and goal."""

    def __init__(
        self,
        observation_size: int,
        goal_size: int,
        action_size: int,
        rngs: nnx.Rngs,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: ActivationFn = nnx.relu,
        kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
        layer_norm: bool = False,
    ):
        self._action_size = action_size
        self.mlp = MLP(
            input_size=observation_size + goal_size,
            layer_sizes=[*hidden_layer_sizes, action_size],
            rngs=rngs,
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
        )

    def __call__(
        self,
        observations: types.Observation,
        goals: jax.Array,
    ) -> jax.Array:
        actor_inputs = jnp.concatenate((observations, goals), axis=-1)
        return nnx.tanh(self.mlp(actor_inputs))


class TwinQNetwork(nnx.Module):
    """Twin critics used by TD3."""

    def __init__(
        self,
        observation_size: int,
        goal_size: int,
        action_size: int,
        rngs: nnx.Rngs,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: ActivationFn = nnx.relu,
        kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
        layer_norm: bool = False,
    ):
        critic_input_size = observation_size + goal_size + action_size
        self.q1 = MLP(
            input_size=critic_input_size,
            layer_sizes=[*hidden_layer_sizes, 1],
            rngs=rngs,
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
        )
        self.q2 = MLP(
            input_size=critic_input_size,
            layer_sizes=[*hidden_layer_sizes, 1],
            rngs=rngs,
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
        )

    def __call__(
        self,
        observations: types.Observation,
        actions: jax.Array,
        goals: jax.Array,
    ) -> jax.Array:
        critic_inputs = jnp.concatenate((observations, actions, goals), axis=-1)
        q1 = self.q1(critic_inputs)
        q2 = self.q2(critic_inputs)
        return jnp.concatenate((q1, q2), axis=-1)


class GCTD3Networks(nnx.Module):
    """Goal-conditioned TD3 actor/critic bundle."""

    def __init__(
        self,
        observation_size: int,
        subgoal_size: int,
        action_size: int,
        rngs: nnx.Rngs,
        observation_preprocessor: ObservationPreprocessor | None = None,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: networks.ActivationFn = nnx.relu,
        kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
        layer_norm: bool = False,
        use_target_networks: bool = True,
    ):
        self.observation_preprocessor = (
            ObservationPreprocessor(observation_size=observation_size) if observation_preprocessor is None else observation_preprocessor
        )
        self.policy_network = GoalConditionedPolicyNetwork(
            observation_size=observation_size,
            goal_size=subgoal_size,
            action_size=action_size,
            rngs=rngs,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
        )
        self.q_networks = TwinQNetwork(
            observation_size=observation_size,
            goal_size=subgoal_size,
            action_size=action_size,
            rngs=rngs,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
        )

        self.target_policy_network = nnx.clone(self.policy_network) if use_target_networks else None
        self.target_q_networks = nnx.clone(self.q_networks) if use_target_networks else None

        self.rngs = rngs

    def _processed_observations(
        self,
        observations: types.Observation,
        *,
        update_preprocessor: bool = False,
    ) -> types.Observation:
        return self.observation_preprocessor(
            observations,
            update_stats=update_preprocessor,
        )

    def action(
        self,
        observations: types.Observation,
        subgoals: jax.Array,
        *,
        exploration_noise: float = 0.0,
        noise_clip: float = 0.0,
        deterministic: bool = False,
        update_preprocessor: bool = False,
    ) -> jax.Array:
        processed_observations = self._processed_observations(
            observations,
            update_preprocessor=update_preprocessor,
        )
        actions = self.policy_network(processed_observations, subgoals)

        if deterministic or exploration_noise <= 0.0:
            return actions

        noise = self.rngs.policy.normal(actions.shape) * exploration_noise
        if noise_clip > 0.0:
            noise = noise.clip(-noise_clip, noise_clip)
        return (actions + noise).clip(-1.0, 1.0)

    def target_action(
        self,
        observations: types.Observation,
        subgoals: jax.Array,
        *,
        update_preprocessor: bool = False,
    ) -> jax.Array:
        if self.target_policy_network is None:
            raise ValueError("Target policy network is not initialized for this HAC level.")
        processed_observations = self._processed_observations(
            observations,
            update_preprocessor=update_preprocessor,
        )
        return self.target_policy_network(processed_observations, subgoals)

    def q_values(
        self,
        observations: types.Observation,
        actions: jax.Array,
        subgoals: jax.Array,
        *,
        update_preprocessor: bool = False,
    ) -> jax.Array:
        processed_observations = self._processed_observations(
            observations,
            update_preprocessor=update_preprocessor,
        )
        return self.q_networks(processed_observations, actions, subgoals)

    def target_q_values(
        self,
        observations: types.Observation,
        actions: jax.Array,
        subgoals: jax.Array,
        *,
        update_preprocessor: bool = False,
    ) -> jax.Array:
        if self.target_q_networks is None:
            raise ValueError("Target critic networks are not initialized for this HAC level.")
        processed_observations = self._processed_observations(
            observations,
            update_preprocessor=update_preprocessor,
        )
        return self.target_q_networks(processed_observations, actions, subgoals)

    def soft_update_targets(self, tau: float) -> None:
        if self.target_policy_network is None or self.target_q_networks is None:
            raise ValueError("Target networks are not initialized for this HAC level.")

        nnx.update(
            self.target_policy_network,
            _polyak_average(
                nnx.state(self.target_policy_network, nnx.Param),
                nnx.state(self.policy_network, nnx.Param),
                tau,
            ),
        )
        nnx.update(
            self.target_q_networks,
            _polyak_average(
                nnx.state(self.target_q_networks, nnx.Param),
                nnx.state(self.q_networks, nnx.Param),
                tau,
            ),
        )

    def __call__(
        self,
        observations: types.Observation,
        subgoals: jax.Array,
        *,
        exploration_noise: float = 0.0,
        noise_clip: float = 0.0,
        deterministic: bool = False,
        update_preprocessor: bool = False,
    ) -> tuple[types.Action, types.Extra]:
        return self.action(
            observations,
            subgoals,
            exploration_noise=exploration_noise,
            noise_clip=noise_clip,
            deterministic=deterministic,
            update_preprocessor=update_preprocessor,
        ), {}


def _make_dummy_transition(
    observation_size: int,
    goal_size: int,
    action_size: int,
) -> GCTransition:
    dummy_obs = jnp.zeros((observation_size,), dtype=jnp.float32)
    dummy_goal = jnp.zeros((goal_size,), dtype=jnp.float32)
    dummy_action = jnp.zeros((action_size,), dtype=jnp.float32)
    return GCTransition(
        observation=dummy_obs,
        goal=dummy_goal,
        next_observation=dummy_obs,
        action=dummy_action,
        reward=jnp.array(0.0, dtype=jnp.float32),
        discount=jnp.array(0.0, dtype=jnp.float32),
        extras={
            "state_extras": {
                "truncation": jnp.array(0.0, dtype=jnp.float32),
                "traj_id": jnp.array(0.0, dtype=jnp.float32),
            },
            "policy_extras": {},
        },
    )


def make_networks_and_buffers(
    *,
    hac_config: HAC,
    run_config: "RunConfig",
    observation_size: int,
    action_size: int,
    subgoal_size: int,
    rngs: nnx.Rngs,
) -> tuple[list[GCTD3Networks], list[TrajectoryUniformSamplingQueueNNX]]:
    """Builds one network and replay buffer per HAC level."""
    if hac_config.num_levels <= 0:
        raise ValueError(f"num_levels must be positive, got {hac_config.num_levels}")

    hidden_layer_sizes = (hac_config.h_dim,) * hac_config.n_hidden
    max_replay_size = hac_config.max_replay_size
    if max_replay_size is None:
        max_replay_size = run_config.total_env_steps

    normalize_fn = (
        running_statistics.normalize
        if hac_config.normalize_observations
        else types.identity_observation_preprocessor
    )
    base_preprocessor = ObservationPreprocessor(
        observation_size=observation_size,
        preprocess_observations_fn=normalize_fn,
    )

    networks_list = []
    replay_buffers = []

    for level in range(hac_config.num_levels):
        level_action_size = action_size if level == 0 else subgoal_size
        level_goal_size = observation_size if level == hac_config.num_levels - 1 else subgoal_size
        level_use_target_networks = level == 0 or hac_config.use_high_level_target_networks

        networks_list.append(
            GCTD3Networks(
                observation_size=observation_size,
                subgoal_size=level_goal_size,
                action_size=level_action_size,
                rngs=rngs,
                observation_preprocessor=nnx.clone(base_preprocessor),
                hidden_layer_sizes=hidden_layer_sizes,
                layer_norm=hac_config.use_ln,
                use_target_networks=level_use_target_networks,
            )
        )

        replay_buffers.append(
            TrajectoryUniformSamplingQueueNNX.create(
                max_replay_size=max_replay_size,
                dummy_data_sample=_make_dummy_transition(
                    observation_size=observation_size,
                    goal_size=level_goal_size,
                    action_size=level_action_size,
                ),
                sample_batch_size=hac_config.batch_size,
                num_envs=run_config.num_envs,
                episode_length=run_config.episode_length,
                rngs=rngs,
            )
        )

    return networks_list, replay_buffers


class HACAgent(nnx.Module):
    """Stateful hierarchical actor that maintains intermediate subgoals."""

    def __init__(
        self,
        subgoal_dim: int,
        networks: Sequence[GCTD3Networks],
        k_step: int = 25,
        num_levels: int = 2,
        enable_temporal_abstraction: bool = True,
        planner_mode: str = "none",
        planner_step_size: float = 2.0,
        goal_indices: Sequence[int] = (0, 1),
    ):
        if num_levels <= 0:
            raise ValueError(f"num_levels must be positive, got {num_levels}")
        if num_levels != len(networks):
            raise ValueError(f"Expected {num_levels} networks, got {len(networks)}")
        if k_step <= 0:
            raise ValueError(f"k_step must be positive, got {k_step}")

        self.networks = nnx.List(networks)
        self.counters = nnx.Variable(jnp.full((num_levels - 1,), k_step, dtype=jnp.int32))
        self.subgoals = nnx.Variable(jnp.zeros((num_levels - 1, subgoal_dim), dtype=jnp.float32))

        self._k_step = k_step
        self._num_levels = num_levels
        self._enable_temporal_abstraction = enable_temporal_abstraction
        self._planner_mode = planner_mode
        self._planner_step_size = planner_step_size
        self._goal_indices = tuple(int(x) for x in goal_indices)

    def _top_level_goal(self, observations: jax.Array, goals: jax.Array) -> jax.Array:
        if self._planner_mode == "none" or self._num_levels == 1:
            return goals

        if goals.shape[-1] == len(self._goal_indices):
            final_goal_xy = goals
        else:
            final_goal_xy = goals[:, jnp.asarray(self._goal_indices)]

        waypoint = oracle_subgoal(
            observations,
            final_goal_xy,
            self._goal_indices,
            self._planner_mode,
            self._planner_step_size,
        )

        if self.subgoals[0].shape[-1] == waypoint.shape[-1]:
            return waypoint
        return observations.at[:, jnp.asarray(self._goal_indices)].set(waypoint)

    def reset(self) -> None:
        self.counters[...] = jnp.full_like(self.counters[...], self._k_step)
        self.subgoals[...] = jnp.zeros_like(self.subgoals[...])

    def __call__(
        self,
        observations: jax.Array,
        goals: jax.Array,
        *,
        exploration_noise: float = 0.0,
        noise_clip: float = 0.0,
        deterministic: bool = False,
    ) -> tuple[types.Action, types.Extra]:
        for level in reversed(range(1, self._num_levels)):
            conditioned_goal = self._top_level_goal(
                observations,
                goals,
            ) if level == self._num_levels - 1 else self.subgoals[level]

            should_refresh_subgoal = not self._enable_temporal_abstraction or jnp.all(self.counters[:level] == self._k_step)
            if level < self._num_levels - 1:
                self.counters[level] = jnp.where(should_refresh_subgoal, self.counters[level] + 1, self.counters[level])
            self.counters[level - 1] = jnp.where(should_refresh_subgoal, 0, self.counters[level - 1])
            self.subgoals[level - 1] = jnp.where(
                should_refresh_subgoal,
                self.networks[level].action(
                    observations,
                    conditioned_goal,
                    exploration_noise=exploration_noise,
                    noise_clip=noise_clip,
                    deterministic=deterministic
                ),
                self.subgoals[level - 1]
            )

        if self._num_levels > 1:
            self.counters[0] += 1
        return self.networks[0].action(
            observations,
            goals if self._num_levels == 1 else self.subgoals[0],
            exploration_noise=exploration_noise,
            noise_clip=noise_clip,
            deterministic=deterministic
        ), {
            "counters": self.counters,
            "subgoals": self.subgoals
        }
