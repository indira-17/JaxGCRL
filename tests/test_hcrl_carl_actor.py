import os
from copy import deepcopy

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from jaxgcrl.agents.crl.networks import Actor, Encoder
from jaxgcrl.agents.hcrl.hcrl import TrainingState, Transition
from jaxgcrl.agents.hcrl.losses import update_actor_and_alpha_carl


def _tree_allclose(a, b):
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y), a, b)
    )
    return bool(jnp.all(jnp.asarray(leaves)))


def _make_state(*, encoder_grad):
    key = jax.random.PRNGKey(0)
    actor_key, sa_key, g_key, sg_key, a_key, high_key = jax.random.split(key, 6)
    state_size = 3
    goal_size = 2
    action_size = 2
    carl_dim = 5
    seq_steps = 2

    actor = Actor(action_size=action_size, network_width=16, network_depth=1, skip_connections=0)
    sa_encoder = Encoder(repr_dim=8, network_width=16, network_depth=1, skip_connections=0)
    g_encoder = Encoder(repr_dim=8, network_width=16, network_depth=1, skip_connections=0)
    sg_encoder = Encoder(repr_dim=carl_dim, network_width=16, network_depth=1, skip_connections=0)
    a_encoder = Encoder(repr_dim=carl_dim, network_width=16, network_depth=1, skip_connections=0)
    high_actor = Actor(action_size=carl_dim, network_width=16, network_depth=1, skip_connections=0)

    training_state = TrainingState(
        env_steps=jnp.array(0),
        gradient_steps=jnp.array(0),
        actor_state=TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, jnp.ones((1, state_size + carl_dim))),
            tx=optax.sgd(1e-2),
        ),
        critic_state=TrainState.create(
            apply_fn=None,
            params={
                "sa_encoder": sa_encoder.init(sa_key, jnp.ones((1, state_size + action_size))),
                "g_encoder": g_encoder.init(g_key, jnp.ones((1, goal_size))),
            },
            tx=optax.sgd(1e-2),
        ),
        alpha_state=TrainState.create(
            apply_fn=None,
            params={"log_alpha": jnp.array(0.0)},
            tx=optax.sgd(1e-2),
        ),
        high_actor_state=TrainState.create(
            apply_fn=high_actor.apply,
            params=high_actor.init(high_key, jnp.ones((1, state_size + goal_size))),
            tx=optax.sgd(1e-2),
        ),
        carl_state=TrainState.create(
            apply_fn=None,
            params={
                "sg_encoder": sg_encoder.init(sg_key, jnp.ones((1, state_size + goal_size))),
                "a_encoder": a_encoder.init(a_key, jnp.ones((1, seq_steps * action_size))),
            },
            tx=optax.sgd(1e-2),
        ),
    )

    batch_size = 4
    state = jnp.arange(batch_size * state_size, dtype=jnp.float32).reshape(batch_size, state_size) / 10.0
    goal = jnp.arange(batch_size * goal_size, dtype=jnp.float32).reshape(batch_size, goal_size) / 10.0
    transitions = Transition(
        observation=jnp.concatenate([state, goal], axis=-1),
        action=jnp.zeros((batch_size, action_size), dtype=jnp.float32),
        reward=jnp.zeros((batch_size,), dtype=jnp.float32),
        discount=jnp.ones((batch_size,), dtype=jnp.float32),
        extras={},
    )
    config = {
        "state_size": state_size,
        "goal_indices": (0, 1),
        "target_entropy": -float(action_size),
        "energy_fn": "dot",
        "carl_actor_encoder_grad": encoder_grad,
    }
    networks = {
        "actor": actor,
        "sa_encoder": sa_encoder,
        "g_encoder": g_encoder,
        "sg_encoder": sg_encoder,
        "a_encoder": a_encoder,
    }
    return config, networks, transitions, training_state


def test_carl_actor_update_stopgrad_keeps_carl_params_fixed():
    config, networks, transitions, training_state = _make_state(encoder_grad=False)
    old_carl_params = deepcopy(training_state.carl_state.params)

    new_state, metrics = update_actor_and_alpha_carl(
        config, networks, transitions, training_state, jax.random.PRNGKey(1)
    )

    assert _tree_allclose(new_state.carl_state.params, old_carl_params)
    assert metrics["carl_actor_encoder_grad_norm"] == 0.0


def test_carl_actor_update_encoder_grad_changes_only_sg_encoder():
    config, networks, transitions, training_state = _make_state(encoder_grad=True)
    old_carl_params = deepcopy(training_state.carl_state.params)

    new_state, metrics = update_actor_and_alpha_carl(
        config, networks, transitions, training_state, jax.random.PRNGKey(1)
    )

    assert not _tree_allclose(
        new_state.carl_state.params["sg_encoder"],
        old_carl_params["sg_encoder"],
    )
    assert _tree_allclose(
        new_state.carl_state.params["a_encoder"],
        old_carl_params["a_encoder"],
    )
    assert metrics["carl_actor_encoder_grad_norm"] > 0.0
