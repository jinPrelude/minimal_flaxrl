"""Minimal LLaMA-style Transformer using Flax NNX built-in modules."""

from dataclasses import dataclass

from flax import struct
import flax.nnx as nnx
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class TransformerConfig:
    hidden_dim: int = 1024
    n_layers: int = 4
    n_heads: int = 8
    context_len: int = 256
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32


class TransformerState(struct.PyTreeNode):
    k_cache: jax.Array
    v_cache: jax.Array
    valid_len: jax.Array
    pos: jax.Array


def reset_done_in_state(state: TransformerState, done_mask) -> TransformerState:
    done = jnp.asarray(done_mask, dtype=jnp.bool_)
    keep = (~done).astype(state.k_cache.dtype)
    keep_int = (~done).astype(state.valid_len.dtype)
    return TransformerState(
        k_cache=state.k_cache * keep[:, None, None, None, None],
        v_cache=state.v_cache * keep[:, None, None, None, None],
        valid_len=state.valid_len * keep_int,
        pos=state.pos * keep_int,
    )


def _validate_config(cfg: TransformerConfig):
    if cfg.hidden_dim % cfg.n_heads != 0:
        raise ValueError(f"hidden_dim must be divisible by n_heads, got {cfg.hidden_dim}, {cfg.n_heads}")
    head_dim = cfg.hidden_dim // cfg.n_heads
    if head_dim % 2 != 0:
        raise ValueError(
            f"hidden_dim // n_heads must be even for RoPE, got head_dim={head_dim} from {cfg.hidden_dim}, {cfg.n_heads}"
        )


def apply_rope(x, positions, inv_freq):
    """Apply rotary positional encoding. x: [B, S, H, D], positions: [B, S]."""
    x_dtype = x.dtype
    angles = positions.astype(inv_freq.dtype)[..., None] * inv_freq[None, None, :]
    cos = jnp.cos(angles)[:, :, None, :]
    sin = jnp.sin(angles)[:, :, None, :]
    x_pair = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    out = jnp.stack([x0 * cos - x1 * sin, x1 * cos + x0 * sin], axis=-1)
    return out.reshape(x.shape).astype(x_dtype)


def compute_positions_and_mask(done_seq, context_len: int):
    done = jnp.asarray(done_seq, dtype=jnp.bool_)
    batch_size, seq_len = done.shape

    t = jnp.arange(seq_len, dtype=jnp.int32)[None, :]

    episode_ids = jnp.cumsum(done.astype(jnp.int32), axis=1)
    reset_points = jnp.where(done, jnp.broadcast_to(t, (batch_size, seq_len)), -jnp.ones((batch_size, seq_len), dtype=jnp.int32))
    last_reset = jnp.maximum.accumulate(reset_points, axis=1)
    # First episode: position starts from 0 at rollout start
    # Subsequent episodes: position resets to 0 at each done
    query_pos = jnp.where(episode_ids == 0, t, t - last_reset)

    # mask shape: [B, query_pos, key_pos]
    same_episode = (episode_ids[:, None, :] == episode_ids[:, :, None])
    causal = (query_pos[:, None, :] <= query_pos[:, :, None])
    within_context = (query_pos[:, None, :] >= query_pos[:, :, None] - (context_len - 1))
    attn_mask = same_episode & causal & within_context

    return query_pos, attn_mask


class TransformerBlock(nnx.Module):
    def __init__(self, cfg: TransformerConfig, *, rngs: nnx.Rngs):
        head_dim = cfg.hidden_dim // cfg.n_heads
        self.inv_freq = 1.0 / (cfg.rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.hidden_dim // cfg.n_heads
        self.wq = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, use_bias=False, dtype=cfg.dtype, param_dtype=cfg.param_dtype, rngs=rngs)
        self.wk = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, use_bias=False, dtype=cfg.dtype, param_dtype=cfg.param_dtype, rngs=rngs)
        self.wv = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, use_bias=False, dtype=cfg.dtype, param_dtype=cfg.param_dtype, rngs=rngs)
        self.wo = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, use_bias=False, dtype=cfg.dtype, param_dtype=cfg.param_dtype, rngs=rngs)
        self.attn_norm = nnx.RMSNorm(
            num_features=cfg.hidden_dim,
            epsilon=cfg.norm_eps,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
            rngs=rngs,
        )
        self.ffn_norm = nnx.RMSNorm(
            num_features=cfg.hidden_dim,
            epsilon=cfg.norm_eps,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
            rngs=rngs,
        )

        ff_dim = int(2 * (4 * cfg.hidden_dim) / 3)
        if cfg.ffn_dim_multiplier is not None:
            ff_dim = int(ff_dim * cfg.ffn_dim_multiplier)
        ff_dim = cfg.multiple_of * ((ff_dim + cfg.multiple_of - 1) // cfg.multiple_of)
        self.w1 = nnx.Linear(cfg.hidden_dim, ff_dim, use_bias=False, dtype=cfg.dtype, param_dtype=cfg.param_dtype, rngs=rngs)
        self.w2 = nnx.Linear(ff_dim, cfg.hidden_dim, use_bias=False, dtype=cfg.dtype, param_dtype=cfg.param_dtype, rngs=rngs)
        self.w3 = nnx.Linear(cfg.hidden_dim, ff_dim, use_bias=False, dtype=cfg.dtype, param_dtype=cfg.param_dtype, rngs=rngs)

    def parallel(self, x, positions, mask):
        # Attention
        B = x.shape[0]
        x_norm = self.attn_norm(x)
        q = apply_rope(self.wq(x_norm).reshape(B, -1, self.n_heads, self.head_dim), positions, self.inv_freq)
        k = apply_rope(self.wk(x_norm).reshape(B, -1, self.n_heads, self.head_dim), positions, self.inv_freq)
        v = self.wv(x_norm).reshape(B, -1, self.n_heads, self.head_dim)

        q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        scale = 1.0 / jnp.sqrt(self.head_dim).astype(q.dtype)
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        big_neg = jnp.finfo(q.dtype).min
        attn_weights = jnp.where(jnp.asarray(mask, dtype=jnp.bool_)[:, None, :, :], attn_weights, big_neg)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_out = jnp.matmul(attn_weights, v).transpose(0, 2, 1, 3).reshape(B, -1, self.n_heads * self.head_dim)
        h = x + self.wo(attn_out)

        # FFN
        ff = self.ffn_norm(h)
        return h + self.w2(jax.nn.silu(self.w1(ff)) * self.w3(ff))

    def step(self, x_t, cached_k, cached_v, positions, prefix_mask):
        x = x_t[:, None, :]
        B = x.shape[0]

        # Attention with KV-cache
        x_norm = self.attn_norm(x)
        q = apply_rope(self.wq(x_norm).reshape(B, 1, self.n_heads, self.head_dim), positions, self.inv_freq)
        k = apply_rope(self.wk(x_norm).reshape(B, 1, self.n_heads, self.head_dim), positions, self.inv_freq)
        v = self.wv(x_norm).reshape(B, 1, self.n_heads, self.head_dim)

        k_full = jnp.concatenate([cached_k, k], axis=1)
        v_full = jnp.concatenate([cached_v, v], axis=1)
        mask = jnp.concatenate([prefix_mask, jnp.ones((B, 1), dtype=jnp.bool_)], axis=1)[:, None, :]

        q = q.transpose(0, 2, 1, 3)
        k_full = k_full.transpose(0, 2, 1, 3)
        v_full = v_full.transpose(0, 2, 1, 3)
        scale = 1.0 / jnp.sqrt(self.head_dim).astype(q.dtype)
        attn_weights = jnp.matmul(q, k_full.transpose(0, 1, 3, 2)) * scale
        big_neg = jnp.finfo(q.dtype).min
        attn_weights = jnp.where(mask[:, None, :, :], attn_weights, big_neg)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_out = jnp.matmul(attn_weights, v_full).transpose(0, 2, 1, 3).reshape(B, 1, self.n_heads * self.head_dim)
        h = x + self.wo(attn_out)

        # FFN
        ff = self.ffn_norm(h)
        h = h + self.w2(jax.nn.silu(self.w1(ff)) * self.w3(ff))
        return h[:, 0], k[:, -1:], v[:, -1:]


class TransformerBackbone(nnx.Module):
    def __init__(self, cfg: TransformerConfig, *, rngs: nnx.Rngs):
        _validate_config(cfg)
        self.cfg = cfg
        self.head_dim = cfg.hidden_dim // cfg.n_heads
        self.layers = nnx.List([TransformerBlock(cfg, rngs=rngs) for _ in range(cfg.n_layers)])
        self.final_norm = nnx.RMSNorm(
            num_features=cfg.hidden_dim,
            epsilon=cfg.norm_eps,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
            rngs=rngs,
        )

    def init_state(self, batch_size, dtype=None) -> TransformerState:
        if dtype is None:
            dtype = self.cfg.dtype
        cache_shape = (batch_size, self.cfg.n_layers, self.cfg.context_len, self.cfg.n_heads, self.head_dim)
        zeros = jnp.zeros(cache_shape, dtype=dtype)
        return TransformerState(
            k_cache=zeros,
            v_cache=zeros,
            valid_len=jnp.zeros((batch_size,), dtype=jnp.int32),
            pos=jnp.zeros((batch_size,), dtype=jnp.int32),
        )

    def step(self, x_t, state: TransformerState):
        # prefix_len entries are valid in the cache; prefix_mask marks which positions are valid
        prefix_len = jnp.minimum(state.valid_len, self.cfg.context_len - 1)
        prefix_idx = jnp.arange(self.cfg.context_len, dtype=jnp.int32)[None, :]
        prefix_mask = prefix_idx >= (self.cfg.context_len - prefix_len[:, None])  # most recent prefix_len entries are valid
        positions = state.pos[:, None]

        # Each layer reads from original cache, collects new k/v into lists
        k_list, v_list = [], []
        for layer_idx, layer in enumerate(self.layers):
            x_t, k_new, v_new = layer.step(
                x_t,
                state.k_cache[:, layer_idx],
                state.v_cache[:, layer_idx],
                positions,
                prefix_mask,
            )
            merged_k = jnp.concatenate([state.k_cache[:, layer_idx], k_new], axis=1)
            merged_v = jnp.concatenate([state.v_cache[:, layer_idx], v_new], axis=1)
            k_list.append(merged_k[:, -self.cfg.context_len:])
            v_list.append(merged_v[:, -self.cfg.context_len:])

        return (
            TransformerState(
                k_cache=jnp.stack(k_list, axis=1),
                v_cache=jnp.stack(v_list, axis=1),
                valid_len=jnp.minimum(state.valid_len + 1, self.cfg.context_len),
                pos=state.pos + 1,
            ),
            self.final_norm(x_t),
        )

    def unroll(self, x_seq, done_seq):
        """Parallel training forward pass. x_seq: [batch, seq, hidden_dim], done_seq: [batch, seq]."""
        positions, attn_mask = compute_positions_and_mask(done_seq, self.cfg.context_len)
        x = x_seq
        for layer in self.layers:
            x = layer.parallel(x, positions, attn_mask)
        return self.final_norm(x)

