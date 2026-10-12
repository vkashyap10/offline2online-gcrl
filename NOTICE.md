# Third-party code

This project builds on two open-source projects. Files derived from them keep their original licences, and every
derived file has been modified.

## OGBench (MIT)

https://github.com/seohongpark/ogbench, Copyright (c) 2024 OGBench Authors. Licence: [LICENSE](LICENSE).

- `offline2online_gcrl/impls/agents/` (`gcivl_jax.py`, `gciql_jax.py`, `gcql_jax.py`, `iql.py`, ported from OGBench's GCIVL, GCIQL and GCQL agents)
- `offline2online_gcrl/impls/utils/networks.py`, `encoders.py`, `flax_utils.py`
- `offline2online_gcrl/data_generation/oracles/` (modelled on the OGBench manipspace oracles)

## JaxGCRL (Apache-2.0)

https://github.com/MichalBortkiewicz/JaxGCRL, Copyright 2023 FLAIR. Licence: [LICENSE-JAXGCRL](LICENSE-JAXGCRL).
Derived from commit `b54cd4c` (January 2025).

- `offline2online_gcrl/environments/manipulation/` and `offline2online_gcrl/environments/assets/*.xml`
- `offline2online_gcrl/environments/wrappers.py`
- `offline2online_gcrl/impls/utils/replay_buffer.py`, `evaluator.py`, `metrics_recorder.py`
- `offline2online_gcrl/impls/utils/training.py`, `env_utils.py`, `buffer_setup.py` (training loop structure)

## MuJoCo Menagerie (Apache-2.0)

`offline2online_gcrl/environments/assets/franka_emika_panda/` is the Franka Emika Panda model from
https://github.com/google-deepmind/mujoco_menagerie, with its licence in that folder.
