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

"""SAC networks."""

from typing import Any, Callable, Optional, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
from brax.training import distribution, networks, types
from brax.training.types import PRNGKey
from flax import linen

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


class MLP(linen.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                if self.layer_norm:
                    hidden = linen.LayerNorm()(hidden)
                hidden = self.activation(hidden)
        return hidden


class GoalRep(linen.Module):
    """Subgoal representation phi([s; g]). L2-normalized to radius sqrt(rep_dim)."""

    layer_sizes: Sequence[int] = (256, 256)
    rep_dim: int = 10
    layer_norm: bool = True

    @linen.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray):
        x = jnp.concatenate([observations, goals], axis=-1)
        x = MLP(
            layer_sizes=list(self.layer_sizes) + [self.rep_dim],
            activate_final=False,
            layer_norm=self.layer_norm,
        )(x)
        x = x / optax.safe_norm(x, min_norm=1e-8, ord=2, axis=-1, keepdims=True) * jnp.sqrt(
            x.shape[-1]
        )
        return x


class ActionsEncoder(linen.Module):
    """MLP encoder for flattened action sequences -> rep_dim (for InfoNCE keys)."""

    layer_sizes: Sequence[int] = (256, 256)
    rep_dim: int = 10
    layer_norm: bool = True

    @linen.compact
    def __call__(self, actions_flat: jnp.ndarray):
        x = MLP(
            layer_sizes=list(self.layer_sizes) + [self.rep_dim],
            activate_final=False,
            layer_norm=self.layer_norm,
        )(actions_flat)
        return x


@flax.struct.dataclass
class SACNetworks:
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    goal_rep_network: Optional[GoalRep] = None
    actions_encoder_network: Optional[ActionsEncoder] = None


def make_q_network(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2,
    layer_norm: bool = False,
) -> networks.FeedForwardNetwork:
    """Creates a value network."""

    class QModule(linen.Module):
        """Q Module."""

        n_critics: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            res = []
            for _ in range(self.n_critics):
                q = MLP(
                    layer_sizes=list(hidden_layer_sizes) + [1],
                    activation=activation,
                    layer_norm=layer_norm,
                )(hidden)
                res.append(q)
            return jnp.concatenate(res, axis=-1)

    q_module = QModule(n_critics=n_critics)

    def apply(processor_params, q_params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        return q_module.apply(q_params, obs, actions)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    return networks.FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply
    )


def make_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    layer_norm: bool = False,
) -> networks.FeedForwardNetwork:
    """Creates a policy network."""

    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        layer_norm=layer_norm,
    )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))
    return networks.FeedForwardNetwork(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_inference_fn(sac_networks: SACNetworks):
    """Creates params and inference function for the SAC agent."""

    def make_policy(params: types.PolicyParams, deterministic: bool = False) -> types.Policy:
        def policy(observations: types.Observation, key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
            logits = sac_networks.policy_network.apply(*params, observations)
            if deterministic:
                return sac_networks.parametric_action_distribution.mode(logits), {}
            return (
                sac_networks.parametric_action_distribution.sample(logits, key_sample),
                {},
            )

        return policy

    return make_policy


def make_goal_rep_inference_fn(sac_networks: SACNetworks, state_dim: int, goal_dim: int):
    """Creates inference function that applies goal_rep before the policy.

    Params tuple: (normalizer_params, policy_params, goal_rep_params).
    The policy network was created with identity preprocessing so normalizer_params
    is only used structurally (identity fn ignores it).
    """
    goal_rep_network = sac_networks.goal_rep_network

    def make_policy(params: types.PolicyParams, deterministic: bool = False) -> types.Policy:
        normalizer_params, policy_params, goal_rep_params = params

        def policy(observations: types.Observation, key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
            state = observations[..., :state_dim]
            goal = observations[..., state_dim : state_dim + goal_dim]
            phi = goal_rep_network.apply(goal_rep_params, state, goal)
            actor_input = jnp.concatenate([state, phi], axis=-1)
            logits = sac_networks.policy_network.apply(normalizer_params, policy_params, actor_input)
            if deterministic:
                return sac_networks.parametric_action_distribution.mode(logits), {}
            return (
                sac_networks.parametric_action_distribution.sample(logits, key_sample),
                {},
            )

        return policy

    return make_policy


def make_sac_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    layer_norm: bool = False,
) -> SACNetworks:
    """Make SAC networks."""
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    policy_network = make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=layer_norm,
    )
    q_network = make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=layer_norm,
    )
    return SACNetworks(
        policy_network=policy_network,
        q_network=q_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_sac_networks_with_goal_rep(
    state_dim: int,
    goal_dim: int,
    action_size: int,
    rep_dim: int = 10,
    goal_rep_hidden: Sequence[int] = (256, 256),
    use_info_nce: bool = False,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    layer_norm: bool = False,
) -> SACNetworks:
    """Make SAC networks with goal_rep as both actor and critic input."""
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    actor_obs_size = state_dim + rep_dim   # policy input: [s, φ(s,g)]
    raw_obs_size = state_dim + goal_dim    # Q input: raw [s, g]
    policy_network = make_policy_network(
        parametric_action_distribution.param_size,
        actor_obs_size,
        preprocess_observations_fn=types.identity_observation_preprocessor,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=layer_norm,
    )
    q_network = make_q_network(
        raw_obs_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=layer_norm,
    )
    goal_rep_network = GoalRep(
        layer_sizes=list(goal_rep_hidden),
        rep_dim=rep_dim,
        layer_norm=layer_norm,
    )
    actions_encoder_network = None
    if use_info_nce:
        actions_encoder_network = ActionsEncoder(
            layer_sizes=list(goal_rep_hidden),
            rep_dim=rep_dim,
            layer_norm=layer_norm,
        )
    return SACNetworks(
        policy_network=policy_network,
        q_network=q_network,
        parametric_action_distribution=parametric_action_distribution,
        goal_rep_network=goal_rep_network,
        actions_encoder_network=actions_encoder_network,
    )
