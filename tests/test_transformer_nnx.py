import unittest

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from ppo_transformer import PPOTransformer
from transformer_nnx import TransformerBackbone, TransformerConfig, TransformerState

OBS_DIM = 8
NUM_ACTIONS = 4


def make_state(cfg: TransformerConfig, batch_size: int, valid_len: int, pos: int, seed: int) -> TransformerState:
    head_dim = cfg.hidden_dim // cfg.n_heads
    key = jax.random.PRNGKey(seed)
    key_k, key_v = jax.random.split(key)
    cache_shape = (batch_size, cfg.n_layers, cfg.context_len, cfg.n_heads, head_dim)
    return TransformerState(
        k_cache=jax.random.normal(key_k, cache_shape, dtype=cfg.dtype),
        v_cache=jax.random.normal(key_v, cache_shape, dtype=cfg.dtype),
        valid_len=jnp.full((batch_size,), valid_len, dtype=jnp.int32),
        pos=jnp.full((batch_size,), pos, dtype=jnp.int32),
    )


def make_done(pattern):
    return jnp.asarray(pattern, dtype=jnp.float32)


class TransformerUnrollTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3

    def _assert_close(self, actual, expected):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-5)

    def _cases(self):
        return [
            {
                "name": "no_done_empty_prefix",
                "context_len": 4,
                "seq_len": 5,
                "valid_len": 0,
                "pos": 0,
                "done": [[0, 0, 0, 0, 0]] * self.batch_size,
            },
            {
                "name": "single_reset_mid",
                "context_len": 4,
                "seq_len": 5,
                "valid_len": 3,
                "pos": 5,
                "done": [[0, 0, 1, 0, 0]] * self.batch_size,
            },
            {
                "name": "multi_reset_full_prefix",
                "context_len": 4,
                "seq_len": 6,
                "valid_len": 4,
                "pos": 9,
                "done": [[0, 1, 0, 1, 0, 0]] * self.batch_size,
            },
            {
                "name": "context_len_one",
                "context_len": 1,
                "seq_len": 4,
                "valid_len": 1,
                "pos": 2,
                "done": [[0, 1, 0, 1]] * self.batch_size,
            },
        ]

    def test_backbone_parallel_unroll_matches_scan_reference(self):
        for case_idx, case in enumerate(self._cases()):
            with self.subTest(case=case["name"]):
                cfg = TransformerConfig(
                    hidden_dim=16,
                    n_layers=2,
                    n_heads=4,
                    context_len=case["context_len"],
                    dtype=jnp.float32,
                    param_dtype=jnp.float32,
                )
                backbone = TransformerBackbone(cfg, rngs=nnx.Rngs(case_idx))
                x_key = jax.random.PRNGKey(100 + case_idx)
                x_seq = jax.random.normal(x_key, (self.batch_size, case["seq_len"], cfg.hidden_dim), dtype=cfg.dtype)
                done_seq = make_done(case["done"])
                init_state = make_state(cfg, self.batch_size, case["valid_len"], case["pos"], 200 + case_idx)

                hidden_parallel = backbone.unroll(x_seq, done_seq, init_state)
                hidden_reference = backbone._unroll_scan_reference(x_seq, done_seq, init_state)

                self._assert_close(hidden_parallel, hidden_reference)

    def test_ppo_transformer_parallel_unroll_matches_scan_reference(self):
        for case_idx, case in enumerate(self._cases()):
            with self.subTest(case=case["name"]):
                cfg = TransformerConfig(
                    hidden_dim=16,
                    n_layers=2,
                    n_heads=4,
                    context_len=case["context_len"],
                    dtype=jnp.float32,
                    param_dtype=jnp.float32,
                )
                model = PPOTransformer(OBS_DIM, NUM_ACTIONS, cfg, rngs=nnx.Rngs(1000 + case_idx))
                obs_key = jax.random.PRNGKey(300 + case_idx)
                obs_seq = jax.random.normal(obs_key, (case["seq_len"], self.batch_size, OBS_DIM), dtype=jnp.float32)
                done_seq = make_done(case["done"]).T
                init_state = make_state(cfg, self.batch_size, case["valid_len"], case["pos"], 400 + case_idx)

                logits_parallel, values_parallel = model.unroll(obs_seq, done_seq, init_state)

                hidden = model.encoder(obs_seq.astype(jnp.float32))
                hidden = jnp.swapaxes(hidden, 0, 1)
                hidden_reference = model.backbone._unroll_scan_reference(hidden, jnp.swapaxes(done_seq, 0, 1), init_state)
                logits_reference = jnp.swapaxes(model.policy_head(hidden_reference), 0, 1)
                values_reference = jnp.swapaxes(model.value_head(hidden_reference).squeeze(-1), 0, 1)

                self._assert_close(logits_parallel, logits_reference)
                self._assert_close(values_parallel, values_reference)


if __name__ == "__main__":
    unittest.main()
