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


def build_parallel_unroll_metadata(done_seq, context_len: int):
    done = jnp.asarray(done_seq, dtype=jnp.bool_)
    batch_size, seq_len = done.shape

    t = jnp.arange(seq_len, dtype=jnp.int32)[None, :]

    episode_ids = jnp.cumsum(done.astype(jnp.int32), axis=1)
    reset_points = jnp.where(done, jnp.broadcast_to(t, (batch_size, seq_len)), -jnp.ones((batch_size, seq_len), dtype=jnp.int32))
    last_reset = jnp.maximum.accumulate(reset_points, axis=1)
    # First episode: position starts from 0 at rollout start
    # Subsequent episodes: position resets to 0 at each done
    query_pos = jnp.where(episode_ids == 0, t, t - last_reset)

    attn_mask = (
        (episode_ids[:, None, :] == episode_ids[:, :, None])
        & (query_pos[:, None, :] <= query_pos[:, :, None])
        & (query_pos[:, None, :] >= query_pos[:, :, None] - (context_len - 1))
    )
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

    def _project_qkv(self, x, positions):
        B = x.shape[0]
        x_norm = self.attn_norm(x)
        q = self.wq(x_norm).reshape(B, -1, self.n_heads, self.head_dim)
        k = self.wk(x_norm).reshape(B, -1, self.n_heads, self.head_dim)
        v = self.wv(x_norm).reshape(B, -1, self.n_heads, self.head_dim)
        q = apply_rope(q, positions, self.inv_freq)
        k = apply_rope(k, positions, self.inv_freq)
        return q, k, v

    def _attend(self, q, k, v, mask):
        # q/k/v: [B, S, n_heads, head_dim] -> transpose to [B, n_heads, S, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        scale = 1.0 / jnp.sqrt(self.head_dim).astype(q.dtype)
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        # mask: [B, query, key] -> [B, 1, query, key] for broadcasting over heads
        big_neg = jnp.finfo(q.dtype).min
        attn_weights = jnp.where(jnp.asarray(mask, dtype=jnp.bool_)[:, None, :, :], attn_weights, big_neg)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_out = jnp.matmul(attn_weights, v)  # [B, n_heads, S, head_dim]
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(q.shape[0], -1, self.n_heads * self.head_dim)
        return self.wo(attn_out)

    def _feed_forward(self, h):
        ff = self.ffn_norm(h)
        return h + self.w2(jax.nn.silu(self.w1(ff)) * self.w3(ff))

    def parallel(self, x, positions, mask):
        query, key, value = self._project_qkv(x, positions)
        h = x + self._attend(query, key, value, mask)
        return self._feed_forward(h)

    def step(self, x_t, prefix_k, prefix_v, positions, prefix_mask):
        x = x_t[:, None, :]
        query, key, value = self._project_qkv(x, positions)
        key_full = jnp.concatenate([prefix_k, key], axis=1)
        value_full = jnp.concatenate([prefix_v, value], axis=1)
        mask = jnp.concatenate([prefix_mask, jnp.ones((x.shape[0], 1), dtype=jnp.bool_)], axis=1)[:, None, :]
        h = x + self._attend(query, key_full, value_full, mask)
        return self._feed_forward(h)[:, 0], key[:, -1:], value[:, -1:]


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
        prefix_len = jnp.minimum(state.valid_len, self.cfg.context_len - 1)
        prefix_idx = jnp.arange(self.cfg.context_len, dtype=jnp.int32)[None, :]
        prefix_mask = prefix_idx >= (self.cfg.context_len - prefix_len[:, None])
        positions = state.pos[:, None]

        k_cache = state.k_cache
        v_cache = state.v_cache
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
            k_cache = k_cache.at[:, layer_idx].set(merged_k[:, -self.cfg.context_len :])
            v_cache = v_cache.at[:, layer_idx].set(merged_v[:, -self.cfg.context_len :])

        return (
            TransformerState(
                k_cache=k_cache,
                v_cache=v_cache,
                valid_len=jnp.minimum(state.valid_len + 1, self.cfg.context_len),
                pos=state.pos + 1,
            ),
            self.final_norm(x_t),
        )

    def _unroll_scan_reference(self, x_seq, done_seq, init_state: TransformerState):
        def scan_step(state, inputs):
            x_t, done_t = inputs
            return self.step(x_t, reset_done_in_state(state, done_t))

        _, hidden_seq = jax.lax.scan(
            scan_step,
            init_state,
            (jnp.swapaxes(x_seq, 0, 1), jnp.swapaxes(done_seq, 0, 1)),
        )
        return jnp.swapaxes(hidden_seq, 0, 1)

    def unroll(self, x_seq, done_seq):
        """Parallel training forward pass. x_seq: [batch, seq, hidden_dim], done_seq: [batch, seq]."""
        positions, attn_mask = build_parallel_unroll_metadata(done_seq, self.cfg.context_len)
        x = x_seq
        for layer in self.layers:
            x = layer.parallel(x, positions, attn_mask)
        return self.final_norm(x)

