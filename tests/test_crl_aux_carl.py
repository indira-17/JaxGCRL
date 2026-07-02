import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from jaxgcrl.agents.crl_aux_carl.crl import TrainingState
from jaxgcrl.agents.crl_aux_carl.losses import update_carl_aux
from jaxgcrl.agents.crl_aux_carl.networks import Encoder


def _tree_allclose(a, b):
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y), a, b)
    )
    return bool(jnp.all(jnp.asarray(leaves)))


def test_carl_aux_update_changes_only_carl_params():
    key = jax.random.PRNGKey(0)
    sg_key, a_key = jax.random.split(key)
    sg_encoder = Encoder(repr_dim=8, network_width=16, network_depth=1, skip_connections=0)
    a_encoder = Encoder(repr_dim=8, network_width=16, network_depth=1, skip_connections=0)

    actor_state = TrainState.create(
        apply_fn=None,
        params={"actor": jnp.ones((2,))},
        tx=optax.sgd(1e-2),
    )
    critic_state = TrainState.create(
        apply_fn=None,
        params={"critic": jnp.ones((2,))},
        tx=optax.sgd(1e-2),
    )
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": jnp.array(0.0)},
        tx=optax.sgd(1e-2),
    )
    carl_state = TrainState.create(
        apply_fn=None,
        params={
            "sg_encoder": sg_encoder.init(sg_key, jnp.ones((1, 5))),
            "a_encoder": a_encoder.init(a_key, jnp.ones((1, 6))),
        },
        tx=optax.sgd(1e-2),
    )
    training_state = TrainingState(
        env_steps=jnp.array(0),
        gradient_steps=jnp.array(0),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
        carl_state=carl_state,
    )
    transitions = SimpleNamespace(
        extras={
            "state": jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10.0,
            "carl_goal": jnp.array(
                [[0.0, 0.0], [1.0, 0.5], [0.5, 1.0], [1.0, 1.0]],
                dtype=jnp.float32,
            ),
            "action_sequence": jnp.arange(24, dtype=jnp.float32).reshape(4, 6) / 10.0,
        }
    )
    config = {
        "energy_fn": "dot",
        "contrastive_loss_fn": "fwd_infonce",
        "logsumexp_penalty_coeff": 0.1,
    }
    networks = {"sg_encoder": sg_encoder, "a_encoder": a_encoder}

    new_state, metrics = update_carl_aux(config, networks, transitions, training_state)

    assert _tree_allclose(new_state.actor_state.params, training_state.actor_state.params)
    assert _tree_allclose(new_state.critic_state.params, training_state.critic_state.params)
    assert _tree_allclose(new_state.alpha_state.params, training_state.alpha_state.params)
    assert not _tree_allclose(new_state.carl_state.params, training_state.carl_state.params)
    assert all(jnp.isfinite(v) for v in metrics.values())
