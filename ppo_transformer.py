import os
from argparse import ArgumentParser

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import flax.nnx as nnx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import wandb

from transformer_nnx import TransformerBackbone, TransformerConfig, TransformerState, reset_done_in_state


MODEL_DTYPE = jnp.float32
PARAM_DTYPE = jnp.float32


class ReplayBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_shape):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs = np.zeros((num_steps, num_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.reset()

    def reset(self):
        self.size = 0

    def add(self, obs, actions, log_probs, rewards, dones, values):
        if self.size >= self.num_steps:
            raise ValueError("ReplayBuffer is full. Call reset() before adding new data.")
        t = self.size
        self.obs[t] = np.asarray(obs, dtype=np.float32)
        self.actions[t] = np.asarray(actions, dtype=np.int32)
        self.log_probs[t] = np.asarray(log_probs, dtype=np.float32)
        self.rewards[t] = np.asarray(rewards, dtype=np.float32)
        self.dones[t] = np.asarray(dones, dtype=np.float32)
        self.values[t] = np.asarray(values, dtype=np.float32)
        self.size += 1

    def get(self):
        if self.size != self.num_steps:
            raise ValueError(f"ReplayBuffer not full: expected {self.num_steps}, got {self.size}")
        return (
            jnp.asarray(self.obs),
            jnp.asarray(self.actions),
            jnp.asarray(self.log_probs),
            jnp.asarray(self.rewards),
            jnp.asarray(self.dones),
            jnp.asarray(self.values),
        )


class PPOTransformer(nnx.Module):
    def __init__(self, obs_dim: int, num_actions: int, transformer_cfg: TransformerConfig, *, rngs: nnx.Rngs):
        self.encoder = nnx.Linear(obs_dim, transformer_cfg.hidden_dim, rngs=rngs)
        self.backbone = TransformerBackbone(transformer_cfg, rngs=rngs)
        self.policy_head = nnx.Linear(transformer_cfg.hidden_dim, num_actions, rngs=rngs)
        self.value_head = nnx.Linear(transformer_cfg.hidden_dim, 1, rngs=rngs)

    def init_state(self, batch_size: int) -> TransformerState:
        return self.backbone.init_state(batch_size, dtype=jnp.float32)

    def step(self, obs, state: TransformerState):
        hidden = self.encoder(jnp.asarray(obs, dtype=jnp.float32))
        next_state, hidden = self.backbone.step(hidden, state)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value, next_state

    def unroll(self, obs_seq, done_seq):
        hidden = self.encoder(jnp.asarray(obs_seq, dtype=jnp.float32))
        hidden = self.backbone.unroll(jnp.swapaxes(hidden, 0, 1), jnp.swapaxes(done_seq, 0, 1))
        logits = jnp.swapaxes(self.policy_head(hidden), 0, 1)
        value = jnp.swapaxes(self.value_head(hidden).squeeze(-1), 0, 1)
        return logits, value


@nnx.jit
def sample_action(model, obs, state, rngs):
    logits, value, next_state = model.step(obs, state)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    actions = rngs.categorical(logits, axis=-1)
    sampled_log_prob = jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)
    return sampled_log_prob, actions, value, next_state


@nnx.jit
def bootstrap_value(model, obs, state):
    _, value, _ = model.step(obs, state)
    return value


def calculate_gae(rewards, values, dones, next_value, next_done, gamma: float, lmbda: float):
    next_values = jnp.concatenate([values[1:], next_value[None, :]], axis=0)
    next_nonterminal = 1.0 - jnp.concatenate([dones[1:], next_done[None, :]], axis=0)
    deltas = rewards + gamma * next_values * next_nonterminal - values

    def scan_step(last_advantage, inputs):
        delta_t, nonterminal_t = inputs
        advantage = delta_t + gamma * lmbda * nonterminal_t * last_advantage
        return advantage, advantage

    init_advantage = jnp.zeros_like(next_value)
    _, advantages = jax.lax.scan(scan_step, init_advantage, (deltas, next_nonterminal), reverse=True)
    return advantages, advantages + values


def normalize_advantages(advantages):
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def loss_fn(model, batch, clip_eps, ent_coef):
    obs, dones, actions, old_log_probs, advantages, returns = batch
    logits, values = model.unroll(obs, dones)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    selected_log_probs = jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)
    ratio = jnp.exp(selected_log_probs - old_log_probs)

    actor_loss = rlax.clipped_surrogate_pg_loss(ratio.reshape(-1), advantages.reshape(-1), clip_eps).mean()
    critic_loss = optax.huber_loss(values, jax.lax.stop_gradient(returns)).mean()
    entropy = -jnp.sum(jax.nn.softmax(logits, axis=-1) * log_probs, axis=-1).mean()
    total_loss = actor_loss + 0.5 * critic_loss - ent_coef * entropy
    return total_loss, (actor_loss, critic_loss, entropy)


@nnx.jit
def update_ppo(model: nnx.Module, optimizer: nnx.Optimizer, minibatches, metrics: nnx.metrics.MultiMetric, clip_eps, ent_coef):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
    def train_minibatch(carry, minibatch):
        model, optimizer, metrics = carry
        (_, (actor_loss, critic_loss, entropy)), grad = grad_fn(model, minibatch, clip_eps, ent_coef)
        optimizer.update(model, grad)
        metrics.update(actor_loss=actor_loss, critic_loss=critic_loss, entropy=entropy)
        return model, optimizer, metrics

    train_minibatch((model, optimizer, metrics), minibatches)


def make_minibatches(batch, env_indices, envs_per_batch: int):
    env_ids = jnp.asarray(env_indices, dtype=jnp.int32).reshape(-1, envs_per_batch)

    def select_time_env(x):
        return jnp.swapaxes(jnp.take(x, env_ids, axis=1), 0, 1)

    return tuple(select_time_env(b) for b in batch)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--env-name", type=str, default="LunarLander-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iter", type=int, default=100000)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-minibatch", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.97)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.001)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--multiple-of", type=int, default=256)
    parser.add_argument("--ffn-dim-multiplier", type=float, default=None)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-theta", type=float, default=500000.0)
    return parser.parse_args()


def validate_args(args):
    if args.context_len < 1:
        raise ValueError(f"context_len must be >= 1, got {args.context_len}")
    if args.num_minibatch < 1:
        raise ValueError(f"num_minibatch must be >= 1, got {args.num_minibatch}")
    if args.num_envs % args.num_minibatch != 0:
        raise ValueError(
            f"num_envs must be divisible by num_minibatch, got {args.num_envs}, {args.num_minibatch}"
        )


def main():
    args = parse_arguments()
    validate_args(args)
    envs_per_batch = args.num_envs // args.num_minibatch

    envs = gym.make_vec(
        args.env_name,
        num_envs=args.num_envs,
        vectorization_mode="sync",
    )
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    obs_dim = envs.single_observation_space.shape[0]
    num_actions = envs.single_action_space.n

    rngs = nnx.Rngs(args.seed)
    model = PPOTransformer(
        obs_dim=obs_dim,
        num_actions=num_actions,
        transformer_cfg=TransformerConfig(
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            context_len=args.context_len,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            norm_eps=args.norm_eps,
            rope_theta=args.rope_theta,
        ),
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adamw(args.learning_rate),
        ),
        wrt=nnx.Param,
    )
    metrics = nnx.metrics.MultiMetric(
        actor_loss=nnx.metrics.Average("actor_loss"),
        critic_loss=nnx.metrics.Average("critic_loss"),
        entropy=nnx.metrics.Average("entropy"),
    )

    wandb.init(
        project="minimal-flaxrl",
        name=f"ppo_transformer_{args.env_name}",
        config=vars(args),
    )

    replay_buffer = ReplayBuffer(args.context_len, args.num_envs, envs.single_observation_space.shape)

    global_env_step = 0
    for iteration in range(args.num_iter):
        obs, _ = envs.reset(seed=args.seed + iteration)
        state = model.init_state(args.num_envs)
        done = np.zeros(args.num_envs, dtype=np.float32)
        rollout_rewards = []
        rollout_lengths = []

        for _ in range(args.context_len):
            state_for_step = reset_done_in_state(state, done)
            log_prob, action, value, state = sample_action(model, obs, state_for_step, rngs)

            next_obs, reward, terminated, truncated, info = envs.step(np.asarray(action))
            next_done = np.maximum(terminated, truncated).astype(np.float32)

            replay_buffer.add(obs, action, log_prob, reward, done, value)
            global_env_step += args.num_envs

            if "_episode" in info:
                for idx, finished in enumerate(info["_episode"]):
                    if finished:
                        rollout_rewards.append(float(info["episode"]["r"][idx]))
                        rollout_lengths.append(int(info["episode"]["l"][idx]))

            obs = next_obs
            done = next_done

        obs_batch, actions_batch, log_probs_batch, rewards_batch, dones_batch, values_batch = replay_buffer.get()

        next_value = bootstrap_value(model, obs, reset_done_in_state(state, done))
        advantages, returns = calculate_gae(
            rewards_batch,
            values_batch,
            dones_batch,
            next_value,
            jnp.asarray(done, dtype=jnp.float32),
            gamma=args.gamma,
            lmbda=args.lmbda,
        )
        train_batch = (
            obs_batch,
            dones_batch,
            actions_batch,
            log_probs_batch,
            normalize_advantages(advantages),
            returns,
        )

        for _ in range(args.num_epochs):
            env_indices = np.asarray(jax.random.permutation(rngs(), args.num_envs))
            minibatches = make_minibatches(train_batch, env_indices, envs_per_batch)
            update_ppo(model, optimizer, minibatches, metrics, clip_eps=args.clip_eps, ent_coef=args.ent_coef)

        metric_values = {k: float(v) for k, v in metrics.compute().items()}
        log_data = {
            "train/iteration": iteration,
            "train/global_env_step": global_env_step,
            "train/actor_loss": metric_values["actor_loss"],
            "train/critic_loss": metric_values["critic_loss"],
            "train/entropy": metric_values["entropy"],
        }
        if rollout_rewards:
            log_data["episode/reward_mean"] = float(np.mean(rollout_rewards))
            log_data["episode/reward_max"] = float(np.max(rollout_rewards))
            log_data["episode/length_mean"] = float(np.mean(rollout_lengths))
            log_data["episode/count"] = len(rollout_rewards)
        wandb.log(log_data, step=global_env_step)

        metrics.reset()
        replay_buffer.reset()

    envs.close()


if __name__ == "__main__":
    main()
