# minimal-rl-nnx

Single-file minimal RL implementations in [Flax NNX](https://flax.readthedocs.io/en/latest/index.html), inspired by [minimalRL](https://github.com/seungeunrho/minimalRL).

## Quick Start

```bash
pip install -r requirements.txt   # Python >= 3.12
wandb login                       # for experiment logging

# for running ppo_gtrxl_atari.py:
pip install ale_py gymnasium[ale]

# for running ppo_gtrxl_memorygym.py:
pip install git+https://github.com/jinPrelude/endless-memory-gym.git
```

## Core Algorithms

- minimal implementations for the LunarLander environment.

| Algorithm | Lines | Command | Training time (MacBook Air M2) | Environment |
|-----------|-------|---------|--------------------------------|-------------|
| [PPO](ppo.py) | 228 | `python ppo.py` | ~40 sec | [LunarLander-v3](ppo.py) |
| [A2C](a2c.py) | 180 | `python a2c.py` | ~100 sec | [LunarLander-v3](a2c.py) |
| [Impala](impala.py) ([cleanba](https://github.com/vwxyzjn/cleanba) style)| 263 | `python impala.py` | ~100 sec | [LunarLander-v3](impala.py) |

## Advanced Implementations

- implementation files that build on the core algorithms with more advanced extensions, including:
  - recurrent policy (LSTM)
  - transformer model (TrXL, GTrXL)
  - harder tasks (atari, memorygym)

| Algorithm | Lines | Command | Environment |
|-----------|-------|---------|-------------|
| [PPO_LSTM](ppo_lstm.py) | 278 | `python ppo_lstm.py` | [LunarLander-v3](ppo_lstm.py) |
| [PPO_TrXL](ppo_trxl.py) | 669 | `python ppo_trxl.py` | [LunarLander-v3](ppo_trxl.py) |
| [PPO_GTrXL](ppo_gtrxl.py) | 692 | `python ppo_gtrxl.py`<br>`python ppo_gtrxl_atari.py`<br>`python ppo_gtrxl_memorygym.py` | [LunarLander-v3](ppo_gtrxl.py)<br>[ALE/Breakout-v5](ppo_gtrxl_atari.py)<br>[MemoryGym](ppo_gtrxl_memorygym.py) |
| [PPO_Transformer](ppo_transformer.py) | 326 | `python ppo_transformer.py`<br>`python ppo_transformer_memorygym.py`<br>`python ppo_transformer_popgym.py` | [LunarLander-v3](ppo_transformer.py)<br>[MemoryGym](ppo_transformer_memorygym.py)<br>[PopGym](ppo_transformer_popgym.py) |
| [Impala_LSTM](impala_lstm.py) | 294 | `python impala_lstm.py` | [LunarLander-v3](impala_lstm.py) |


If you'd like to see a specific algorithm implemented, feel free to open an [issue](../../issues).

## Tuning Tips

- Training failed with `gamma=0.97`. Setting it to `0.99` was critical for learning.
- Increasing hidden dim from 128 to 256 improved both convergence speed and final performance.
- For A2C, updating the actor with `V` instead of `G - V` (advantage) caused training to fail.
- TrXL appears to be highly sensitive to hyperparameter tuning. For example, increasing `trxl_dim` from 128 to 256 (and `trxl-num-heads` from 2 to 4) caused training to fail.
- In contrast, GTrXL was more stable and still trained well when increasing `trxl_dim` to 256.
- For PPO_Transformer on MemoryGym and PopGym environments, regression critic fails to train effectively. Instead, we use a categorical critic based on [HL-Gauss distributional RL](https://arxiv.org/abs/2403.03950).

## References
* Heavily Inspired by the philosophy of the [minimalrl](https://github.com/seungeunrho/minimalRL) repository.
* The Impala implementation closely follows [cleanba](https://github.com/vwxyzjn/cleanba), with the main change being a migration from Flax Linen to Flax NNX. Their Impala design is outstanding - huge thanks to their codebase!

## Performance graph

<img src="assets/performance_graph.png" width="300" />
