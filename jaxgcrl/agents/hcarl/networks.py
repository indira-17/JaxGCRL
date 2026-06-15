"""Only the HCRL high actor lives here.

Actor, SAEncoder, and GEncoder must be imported from jaxgcrl.agents.crl.networks
inside hcarl.py. This prevents HCRL from silently drifting away from CRL.
"""

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    layer_sizes: Sequence[int]
    activation: Callable = nn.relu
    kernel_init: Callable = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False

    @nn.compact
    def __call__(self, data: jnp.ndarray) -> jnp.ndarray:
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                if self.layer_norm:
                    hidden = nn.LayerNorm()(hidden)
                hidden = self.activation(hidden)
        return hidden


class HighActor(nn.Module):
    """High actor pi_H(g_sub | concat(s, g_final)).

    It predicts raw goal coordinates, not a latent representation.
    """

    goal_dim: int
    layer_sizes: Sequence[int] = (512, 512, 512)
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, high_observation: jnp.ndarray):
        x = MLP(layer_sizes=list(self.layer_sizes), activate_final=True)(high_observation)
        mean = nn.Dense(self.goal_dim, name="mean")(x)
        log_std = nn.Dense(self.goal_dim, name="log_std")(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

class Value(nn.Module):
    """Value function V(s)."""

    layer_sizes: Sequence[int] = (512, 512, 512)
    rep_dim: int = 10
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observation: jnp.ndarray, goal_rep: jnp.ndarray):
        x = jnp.concatenate([observation, goal_rep], axis=-1)
        x = MLP(
            layer_sizes=list(self.layer_sizes) + [1],
            activate_final=False,
            layer_norm=self.layer_norm,
        )(x)
        return x