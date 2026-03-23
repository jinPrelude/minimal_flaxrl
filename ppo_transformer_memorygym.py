import os
import time
from argparse import ArgumentParser

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import flax.nnx as nnx
import gymnasium as gym
import memory_gym  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import wandb

from transformer_nnx import TransformerBackbone, TransformerConfig, TransformerState, reset_done_in_state


MODEL_DTYPE = jnp.bfloat16
PARAM_DTYPE = jnp.bfloat16
SUPPORTED_ENVS = {"MortarMayhem-Grid-v0", "MysteryPath-Grid-v0"}


class ReplayBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_shape):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs = np.zeros((num_steps, num_envs, *obs_shape), dtype=np.uint8)
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
        self.obs[t] = np.asarray(obs, dtype=np.uint8)
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


class PPOTransformerMemoryGym(nnx.Module):
    def __init__(self, obs_shape: tuple[int, int, int], num_actions: int, transformer_cfg: TransformerConfig, *, rngs: nnx.Rngs):
        if len(obs_shape) != 3:
            raise ValueError(f"`obs_shape` must be rank-3 HWC image shape, got {obs_shape}")

        self.obs_shape = obs_shape
        self.transformer_cfg = transformer_cfg

        self.conv1 = nnx.Conv(
            in_features=obs_shape[-1],
            out_features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            dtype=MODEL_DTYPE,
            param_dtype=PARAM_DTYPE,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            dtype=MODEL_DTYPE,
            param_dtype=PARAM_DTYPE,
            rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            in_features=64,
            out_features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            dtype=MODEL_DTYPE,
            param_dtype=PARAM_DTYPE,
            rngs=rngs,
        )

        dummy = jnp.zeros((1, *obs_shape), dtype=MODEL_DTYPE)
        dummy = nnx.relu(self.conv1(dummy))
        dummy = nnx.relu(self.conv2(dummy))
        dummy = nnx.relu(self.conv3(dummy))
        self.encoder = nnx.Linear(
            int(np.prod(dummy.shape[1:])),
            transformer_cfg.hidden_dim,
            dtype=transformer_cfg.dtype,
            param_dtype=transformer_cfg.param_dtype,
            rngs=rngs,
        )
        self.backbone = TransformerBackbone(transformer_cfg, rngs=rngs)
        self.policy_head = nnx.Linear(
            transformer_cfg.hidden_dim,
            num_actions,
            dtype=transformer_cfg.dtype,
            param_dtype=transformer_cfg.param_dtype,
            rngs=rngs,
        )
        self.value_head = nnx.Linear(
            transformer_cfg.hidden_dim,
            1,
            dtype=transformer_cfg.dtype,
            param_dtype=transformer_cfg.param_dtype,
            rngs=rngs,
        )

    def _encode_obs(self, obs):
        x = jnp.asarray(obs, dtype=MODEL_DTYPE) / jnp.asarray(255.0, dtype=MODEL_DTYPE)
        if x.ndim == 4:
            x = nnx.relu(self.conv1(x))
            x = nnx.relu(self.conv2(x))
            x = nnx.relu(self.conv3(x))
            x = x.reshape((x.shape[0], -1))
            return nnx.relu(self.encoder(x))
        if x.ndim == 5:
            t, b = x.shape[:2]
            x = x.reshape((t * b, *self.obs_shape))
            x = nnx.relu(self.conv1(x))
            x = nnx.relu(self.conv2(x))
            x = nnx.relu(self.conv3(x))
            x = x.reshape((t * b, -1))
            x = nnx.relu(self.encoder(x))
            return x.reshape((t, b, -1))
        raise ValueError(f"`obs` must be rank-4 or rank-5 image batch, got shape={x.shape}")

    def init_state(self, batch_size: int) -> TransformerState:
        return self.backbone.init_state(batch_size, dtype=self.transformer_cfg.dtype)

    def step(self, obs, state: TransformerState):
        hidden = self._encode_obs(obs)
        next_state, hidden = self.backbone.step(hidden, state)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value, next_state

    def unroll(self, obs_seq, done_seq):
        hidden = self._encode_obs(obs_seq)
        hidden = self.backbone.unroll(jnp.swapaxes(hidden, 0, 1), jnp.swapaxes(done_seq, 0, 1))
        logits = jnp.swapaxes(self.policy_head(hidden), 0, 1)
        value = jnp.swapaxes(self.value_head(hidden).squeeze(-1), 0, 1)
        return logits, value


@nnx.jit
def sample_action(model, obs, state, rngs):
    logits, value, next_state = model.step(obs, state)
    logits = logits.astype(jnp.float32)
    value = value.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    actions = rngs.categorical(logits, axis=-1)
    sampled_log_prob = jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)
    return sampled_log_prob, actions, value, next_state


@nnx.jit
def bootstrap_value(model, obs, state):
    _, value, _ = model.step(obs, state)
    return value.astype(jnp.float32)


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

    logits = logits.astype(jnp.float32)
    values = values.astype(jnp.float32)
    old_log_probs = old_log_probs.astype(jnp.float32)
    advantages = advantages.astype(jnp.float32)
    returns = returns.astype(jnp.float32)
    clip_eps = jnp.asarray(clip_eps, dtype=jnp.float32)
    ent_coef = jnp.asarray(ent_coef, dtype=jnp.float32)

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
    parser.add_argument("--env-name", type=str, default="MortarMayhem-Grid-v0")
    parser.add_argument("--render-mode", type=str, default="debug_rgb_array")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iter", type=int, default=100000)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-minibatch", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=1e-4)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--multiple-of", type=int, default=256)
    parser.add_argument("--ffn-dim-multiplier", type=float, default=None)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-theta", type=float, default=500000.0)
    return parser.parse_args()


def validate_args(args):
    if args.env_name not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported env_name {args.env_name}. Supported: {sorted(SUPPORTED_ENVS)}")
    if args.context_len < 1:
        raise ValueError(f"context_len must be >= 1, got {args.context_len}")
    if args.num_minibatch < 1:
        raise ValueError(f"num_minibatch must be >= 1, got {args.num_minibatch}")
    if args.num_envs % args.num_minibatch != 0:
        raise ValueError(f"num_envs must be divisible by num_minibatch, got {args.num_envs}, {args.num_minibatch}")


def main():
    args = parse_arguments()
    validate_args(args)
    envs_per_batch = args.num_envs // args.num_minibatch

    envs = gym.make_vec(
        args.env_name,
        num_envs=args.num_envs,
        vectorization_mode="sync",
        render_mode=args.render_mode,
    )
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    if not isinstance(obs_space, gym.spaces.Box):
        raise ValueError(f"Only Box observation space is supported, got {obs_space}")
    if len(obs_space.shape) != 3:
        raise ValueError(f"Observation must be HWC image shape, got {obs_space.shape}")
    if obs_space.dtype != np.uint8:
        raise ValueError(f"Observation dtype must be uint8 image, got {obs_space.dtype}")
    if not isinstance(act_space, gym.spaces.Discrete):
        raise ValueError(f"Only Discrete action space is supported, got {act_space}")

    vector_env = envs.env if hasattr(envs, "env") else envs
    vector_env.envs[0].reset(seed=args.seed)
    max_episode_steps = int(vector_env.envs[0].get_wrapper_attr("max_episode_steps"))
    if max_episode_steps <= 0:
        max_episode_steps = args.context_len
    effective_context_len = min(args.context_len, max_episode_steps)

    rngs = nnx.Rngs(args.seed)
    model = PPOTransformerMemoryGym(
        obs_shape=obs_space.shape,
        num_actions=act_space.n,
        transformer_cfg=TransformerConfig(
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            context_len=effective_context_len,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            norm_eps=args.norm_eps,
            rope_theta=args.rope_theta,
            dtype=MODEL_DTYPE,
            param_dtype=PARAM_DTYPE,
        ),
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(model, optax.adamw(args.learning_rate), wrt=nnx.Param)
    metrics = nnx.metrics.MultiMetric(
        actor_loss=nnx.metrics.Average("actor_loss"),
        critic_loss=nnx.metrics.Average("critic_loss"),
        entropy=nnx.metrics.Average("entropy"),
    )

    wandb.init(
        project="minimal-flaxrl",
        name=f"ppo_transformer_memorygym_{args.env_name}",
        config={
            **vars(args),
            "requested_context_len": args.context_len,
            "effective_context_len": effective_context_len,
            "max_episode_steps": max_episode_steps,
        },
    )

    obs, _ = envs.reset(seed=args.seed)
    state = model.init_state(args.num_envs)
    done = np.zeros(args.num_envs, dtype=np.float32)
    replay_buffer = ReplayBuffer(effective_context_len, args.num_envs, obs_space.shape)

    global_env_step = 0
    start_time = time.time()
    for iteration in range(args.num_iter):
        state = model.init_state(args.num_envs)
        rollout_rewards = []
        rollout_lengths = []

        for _ in range(effective_context_len):
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
        sps = int(global_env_step / max(time.time() - start_time, 1e-6))
        log_data = {
            "train/iteration": iteration,
            "train/global_env_step": global_env_step,
            "train/sps": sps,
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
