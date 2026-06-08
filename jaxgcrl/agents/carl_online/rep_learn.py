import functools
import os
from typing import Sequence

import ogbench
import flax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import linen as nn
from flax.training import checkpoints
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from ..sac.networks import ActionsEncoder, GoalRep

class RepLearnAgent(nn.Module):
    use_goal_rep: bool
    rep_dim: int
    hidden_sizes: Sequence[int]

    @nn.compact
    def __call__(self, states, goals, actions):
        action_encoder = ActionsEncoder(layer_sizes=self.hidden_sizes, rep_dim=self.rep_dim)
        action_rep = action_encoder(actions)

        if self.use_goal_rep:
            goal_rep = GoalRep(layer_sizes=self.hidden_sizes, rep_dim=self.rep_dim)
            state_goal_rep = goal_rep(states, goals)
        else:
            state_goal_rep = states

        return state_goal_rep, action_rep


def info_nce_loss(state_goal_rep, action_rep, temperature, min_norm, norm_ord, norm_axis, norm_keepdims):
    """Symmetric InfoNCE loss. queries and keys are (batch, rep_dim).

    Both are L2-normalized before computing similarity.
    Uses optax.safe_norm to avoid NaN gradients when norm approaches zero.
    """
    queries = state_goal_rep
    keys = action_rep
    q_norm = optax.safe_norm(queries, min_norm=min_norm, ord=norm_ord, axis=norm_axis, keepdims=norm_keepdims)
    k_norm = optax.safe_norm(keys, min_norm=min_norm, ord=norm_ord, axis=norm_axis, keepdims=norm_keepdims)
    queries = queries / q_norm
    keys = keys / k_norm
    sim = jnp.matmul(queries, keys.T) / temperature
    diag = jnp.diag(sim)
    loss_fwd = jnp.mean(-diag + jax.nn.logsumexp(sim, axis=-1))
    loss_rev = jnp.mean(-diag + jax.nn.logsumexp(sim, axis=0))
    return loss_fwd + loss_rev

class GCDatasetSampler:
    """Trajectory-aware goal sampler matching OGBench's GCDataset.sample() logic.

    Goals are sampled as future states from the same trajectory using geometric
    (or uniform) lookahead, with a fallback to random goals with probability
    p_randomgoal. This is richer than using next_observations directly.

    If goal_indices is provided, goals are sliced to only those dimensions
    (e.g. [0, 1] for xy position), matching how SAC/HER defines goals.
    """

    def __init__(self, dataset, discount, geom_sample, p_trajgoal, p_randomgoal, goal_indices=None, nce_k_step=1):
        self.dataset = dataset
        self.discount = discount
        self.geom_sample = geom_sample
        self.p_trajgoal = p_trajgoal
        self.p_randomgoal = p_randomgoal
        self.goal_indices = goal_indices  # e.g. [0, 1] for ant xy
        self.nce_k_step = nce_k_step
        self.size = len(dataset['observations'])
        # Pre-compute trajectory boundaries from terminal flags.
        (self.terminal_locs,) = np.nonzero(dataset['terminals'] > 0)
        assert self.terminal_locs[-1] == self.size - 1, (
            "Last transition must be terminal. Check dataset integrity."
        )

    def _get_random_idxs(self, n):
        return np.random.randint(self.size, size=n)

    def sample(self, batch_size):
        idxs = self._get_random_idxs(batch_size)
        # End-of-trajectory index for each sampled transition.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        if self.geom_sample:
            # Geometric sampling: distance ~ Geom(1 - discount), biased toward near future.
            offsets = np.random.geometric(p=1 - self.discount, size=batch_size)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling: uniformly between next state and end of trajectory.
            distances = np.random.rand(batch_size)
            traj_goal_idxs = np.round(
                np.minimum(idxs + 1, final_state_idxs) * distances
                + final_state_idxs * (1 - distances)
            ).astype(int)

        random_goal_idxs = self._get_random_idxs(batch_size)
        pick_random = np.random.rand(batch_size) < self.p_randomgoal
        goal_idxs = np.where(pick_random, random_goal_idxs, traj_goal_idxs)

        goal_obs = self.dataset['observations'][goal_idxs]
        if self.goal_indices is not None:
            goal_obs = goal_obs[:, self.goal_indices]

        # Sample k-step action sequence: actions at t, t+1, ..., t+k-1 (clipped to traj end)
        k = self.nce_k_step
        action_dim = self.dataset['actions'].shape[1]
        k_indices = np.minimum(
            idxs[:, None] + np.arange(k)[None, :],
            final_state_idxs[:, None],
        )  # (batch, k)
        actions_flat = self.dataset['actions'][k_indices].reshape(len(idxs), k * action_dim)

        return {
            'state':   jnp.array(self.dataset['observations'][idxs]),
            'goal':    jnp.array(goal_obs),
            'action':  jnp.array(actions_flat),
        }

def load_goal_conditioned_dataset(cfg):
    if cfg.dataset.source != 'ogbench':
        raise ValueError(f"Unsupported dataset source: {cfg.dataset.source}")

    dataset_name = cfg.dataset.name
    dataset_dir = os.path.expanduser(cfg.dataset.dir)

    # OGBench docs: datasets are goal-conditioned and auto-download on first load.
    # We additionally expose an explicit pre-download option for large jobs.
    if cfg.dataset.download_first:
        ogbench.download_datasets([dataset_name], dataset_dir=dataset_dir)

    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        dataset_name,
        dataset_dir=dataset_dir,
        compact_dataset=cfg.dataset.compact_dataset,
    )

    goal_indices_cfg = getattr(cfg.gc_sampling, 'goal_indices', None)
    goal_indices = list(goal_indices_cfg) if goal_indices_cfg is not None else None
    nce_k_step = int(getattr(cfg.rep_learn, 'nce_k_step', 1))
    sampler_kwargs = dict(
        discount=cfg.gc_sampling.discount,
        geom_sample=cfg.gc_sampling.geom_sample,
        p_trajgoal=cfg.gc_sampling.p_trajgoal,
        p_randomgoal=cfg.gc_sampling.p_randomgoal,
        goal_indices=goal_indices,
        nce_k_step=nce_k_step,
    )
    train_sampler = GCDatasetSampler(train_dataset, **sampler_kwargs)
    val_sampler = GCDatasetSampler(val_dataset, **sampler_kwargs)
    return env, train_sampler, val_sampler

@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def train_step(state, batch, temperature, min_norm, norm_ord, norm_axis, norm_keepdims):
    """
    state: TrainState object holding params and opt_state
    batch: Dictionary of sampled JAX arrays
    """
    def loss_fn(params):
        state_goal_rep, action_rep = state.apply_fn(
            {'params': params}, batch['state'], batch['goal'], batch['action']
        )
        return info_nce_loss(state_goal_rep, action_rep, temperature, min_norm, norm_ord, norm_axis, norm_keepdims)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # Update parameters using Optax
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    
    # Return updated state
    return state.replace(params=new_params, opt_state=new_opt_state), loss

@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def eval_step(state, batch, temperature, min_norm, norm_ord, norm_axis, norm_keepdims):
    state_goal_rep, action_rep = state.apply_fn(
        {'params': state.params}, batch['state'], batch['goal'], batch['action']
    )
    return info_nce_loss(state_goal_rep, action_rep, temperature, min_norm, norm_ord, norm_axis, norm_keepdims)

@hydra.main(version_base=None, config_path="config", config_name="rep_learn_config")
def main(cfg: OmegaConf):
    """Main training function."""
    OmegaConf.set_struct(cfg, False)

    # Load demo parameters from config
    env_name = cfg.dataset.name
    jaxgcrl_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ckpt_dir = os.path.join(jaxgcrl_root_dir, "checkpoints")
    batch_size = cfg.rep_learn.batch_size
    val_num_batches = cfg.rep_learn.val_num_batches
    eval_interval = cfg.rep_learn.eval_interval
    epochs = cfg.rep_learn.epochs
    best_val_loss = float(cfg.rep_learn.init_best_loss)

    wandb.init(
        project=cfg.wandb.project,
        name=env_name + '-rep-learn',
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # load a goal-conditioned offline dataset (downloaded if needed)
    env, train_sampler, val_sampler = load_goal_conditioned_dataset(cfg)

    # initialize Network
    rng = jax.random.PRNGKey(cfg.rep_learn.seed)
    model = RepLearnAgent(
        use_goal_rep=cfg.rep_learn.use_goal_rep,
        rep_dim=cfg.rep_learn.rep_dim,
        hidden_sizes=tuple(cfg.rep_learn.hidden_sizes),
    )
    obs_dim = train_sampler.dataset['observations'].shape[1]
    action_dim = train_sampler.dataset['actions'].shape[1]
    goal_indices_cfg = getattr(cfg.gc_sampling, 'goal_indices', None)
    goal_dim = len(goal_indices_cfg) if goal_indices_cfg is not None else obs_dim
    nce_k_step = int(getattr(cfg.rep_learn, 'nce_k_step', 1))
    dummy_state = jnp.ones((1, obs_dim))
    dummy_goal = jnp.ones((1, goal_dim))
    dummy_action = jnp.ones((1, nce_k_step * action_dim))
    params = model.init(rng, dummy_state, dummy_goal, dummy_action)['params']
    
    # setup optimiser
    tx = optax.adam(learning_rate=cfg.rep_learn.learning_rate)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    
    # train the representation using InfoNCE loss 
    print(cfg.logging.start_message)
    for step in range(epochs):
        batch = train_sampler.sample(batch_size)
        state, loss_value = train_step(state, batch, cfg.rep_learn.temperature, cfg.rep_learn.nce_min_norm, cfg.rep_learn.nce_norm_ord, cfg.rep_learn.nce_norm_axis, cfg.rep_learn.nce_norm_keepdims)
        loss_value = float(loss_value)

        val_loss_value = None
        if step % eval_interval == 0:
            val_losses = []
            for _ in range(val_num_batches):
                val_batch = val_sampler.sample(batch_size)
                v = eval_step(state, val_batch, cfg.rep_learn.temperature, cfg.rep_learn.nce_min_norm, cfg.rep_learn.nce_norm_ord, cfg.rep_learn.nce_norm_axis, cfg.rep_learn.nce_norm_keepdims)
                val_losses.append(float(v))
            val_loss_value = float(np.mean(val_losses))
            log_payload = {
                cfg.logging.wandb_loss_key: loss_value,
                cfg.logging.wandb_val_loss_key: val_loss_value,
            }

            # Save best model based on validation loss.
            if val_loss_value < best_val_loss:
                best_val_loss = val_loss_value
                model_dir = os.path.join(ckpt_dir, env_name)
                os.makedirs(model_dir, exist_ok=True)
                checkpoints.save_checkpoint(
                    ckpt_dir=model_dir,
                    target=flax.serialization.to_state_dict(state.params),
                    step=step,
                    prefix=cfg.rep_learn.checkpoint_prefix,
                    overwrite=cfg.rep_learn.checkpoint_overwrite,
                )

            wandb.log(log_payload, step=step)

            print(
                cfg.logging.step_message_template.format(
                    step=step,
                    loss=loss_value,
                    val_loss=val_loss_value,
                )
            )

    wandb.finish()

    return state.params

if __name__ == '__main__':
    main()