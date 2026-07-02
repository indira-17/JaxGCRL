import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

from jaxgcrl.agents.hcarl.hcarl import (
    make_replay_snapshot_metadata,
    load_replay_snapshot,
    save_replay_snapshot,
    validate_replay_snapshot_metadata,
)
from jaxgcrl.utils.replay_buffer import ReplayBufferState


def _metadata(**overrides):
    metadata = make_replay_snapshot_metadata(
        env_name="ant",
        obs_size=31,
        action_size=8,
        num_envs=4,
        episode_length=5,
        subgoal_steps=3,
        goal_indices=(0, 1),
        snapshot_step=20,
    )
    metadata.update(overrides)
    return metadata


def test_replay_snapshot_round_trip(tmp_path):
    buffer_state = ReplayBufferState(
        data=jnp.ones((5, 4, 3), dtype=jnp.float32),
        insert_position=jnp.array(5, dtype=jnp.int32),
        sample_position=jnp.array(0, dtype=jnp.int32),
        key=jax.random.PRNGKey(0),
    )
    metadata = _metadata()
    path = tmp_path / "snapshot.pkl"

    save_replay_snapshot(str(path), buffer_state, metadata)
    loaded_state, loaded_metadata = load_replay_snapshot(str(path))

    assert loaded_metadata == metadata
    assert jnp.array_equal(loaded_state.data, buffer_state.data)
    assert int(loaded_state.insert_position) == 5


def test_replay_snapshot_metadata_validation_rejects_mismatch():
    metadata = _metadata(episode_length=6)

    with pytest.raises(ValueError, match="episode_length"):
        validate_replay_snapshot_metadata(
            metadata,
            env_name="ant",
            obs_size=31,
            action_size=8,
            num_envs=4,
            episode_length=5,
            subgoal_steps=3,
            goal_indices=(0, 1),
        )


def test_random_snapshot_actions_are_bounded():
    key = jax.random.PRNGKey(7)
    actions = jax.random.uniform(key, shape=(128, 6), minval=-1.0, maxval=1.0)

    assert bool(jnp.all(actions >= -1.0))
    assert bool(jnp.all(actions <= 1.0))
