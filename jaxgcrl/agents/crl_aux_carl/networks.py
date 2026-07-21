import logging

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling


class Encoder(nn.Module):
    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(self, data: jnp.ndarray):
        logging.info("encoder input shape: %s", data.shape)
        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        normalize = (lambda x: nn.LayerNorm()(x)) if self.use_ln else (lambda x: x)
        activation = nn.relu if self.use_relu else nn.swish

        x = data
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)

            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        return nn.Dense(self.repr_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)


class ForwardDynamics(nn.Module):
    """Predicts the achieved K-step goal from state and action sequence."""

    goal_size: int
    network_width: int = 256
    network_depth: int = 2
    use_relu: bool = False

    @nn.compact
    def __call__(self, state: jnp.ndarray, action_sequence: jnp.ndarray):
        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        activation = nn.relu if self.use_relu else nn.swish

        x = jnp.concatenate([state, action_sequence], axis=-1)
        for _ in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = activation(x)

        return nn.Dense(self.goal_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)


class BackwardDynamics(nn.Module):
    """Models q(A | s, g) as a diagonal Gaussian over K-step action sequences."""

    action_sequence_size: int
    network_width: int = 256
    network_depth: int = 2
    use_relu: bool = False
    log_std_min: float = -5.0
    log_std_max: float = 0.0

    @nn.compact
    def __call__(self, state: jnp.ndarray, goal: jnp.ndarray):
        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        activation = nn.relu if self.use_relu else nn.swish

        x = jnp.concatenate([state, goal], axis=-1)
        for _ in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = activation(x)

        mean = nn.tanh(
            nn.Dense(self.action_sequence_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        )
        log_std = nn.Dense(
            self.action_sequence_size, kernel_init=lecun_unfirom, bias_init=bias_init
        )(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class Actor(nn.Module):
    action_size: int
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        normalize = (lambda x: nn.LayerNorm()(x)) if self.use_ln else (lambda x: x)
        activation = nn.relu if self.use_relu else nn.swish
        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        logging.info("actor input shape: %s", x.shape)
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)

            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std