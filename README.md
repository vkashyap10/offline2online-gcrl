# offline2online_gcrl

Offline-to-online goal-conditioned reinforcement learning in JAX.

An agent is first trained on a fixed dataset of demonstrations, then fine-tuned online by collecting its own
experience in many simulated environments running in parallel on the GPU (Brax + MuJoCo MJX). Environment stepping,
replay buffers and gradient updates all run inside one compiled JAX program.

The repository combines goal-conditioned agents from [OGBench](https://github.com/seohongpark/ogbench) with the
GPU environments and trajectory replay buffer from [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL), and adds:

- an offline phase followed by an online phase, where each online batch is half offline and half online data;
- optional filtering of online experience so that only episodes that reach their goal enter the online buffer;
- a scripted demonstration generator for a Franka Panda pick-and-place task.

## How it works

1. **Offline phase.** The agent trains on the offline buffer only. Each sample is a trajectory segment; for every
   state, a later state from the same trajectory is drawn as the goal (hindsight relabelling, with geometric weighting
   by `discount`). The reward is 1 when the goal is reached.
2. **Seeding.** The online buffer is filled with `seed_steps` environment steps from the current policy.
3. **Online phase.** Each iteration collects `unroll_length` steps in every parallel environment, adds them to the
   online buffer, and trains on batches drawn equally from the offline and online buffers.

With `insert_only_successful_episodes: True`, step 3 keeps only transitions from episodes that reached the goal within
the unroll. Without it, the online buffer fills with failed attempts, and since goals are relabelled from whatever
trajectories are in the buffer, training is dominated by them.

## Installation

Requires Python 3.10 and an NVIDIA GPU with CUDA 12.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # exact versions used for development
pip install -e .
```

## Data

Datasets are stored as `.npz` files with one stacked array per field (`observation`, `next_observation`, `action`,
`reward`, `terminate`, `truncation`, `traj_id`) in `offline2online_gcrl/data_generation/data/`.

Generate demonstrations for `arm_binpick_easy` with the scripted controller:

```bash
python -m offline2online_gcrl.data_generation.generate_binpick \
    --env_name=arm_binpick_easy --num_episodes=100 --num_envs=10 \
    --save_path=./offline2online_gcrl/data_generation/data/arm_binpick_easy.pkl --html_viz=False
```

This writes the raw rollouts to `arm_binpick_easy.pkl` and the training dataset to `arm_binpick_easy.npz`.
The generator uses joint-angle inverse kinematics and does not yet support `arm_binpick_easy_EEF`.

Older datasets saved as a pickled list of per-transition dicts can be converted once:

```bash
PYTHONPATH=. python scripts/convert_dataset_to_npz.py path/to/dataset.pkl
```

## Training

```bash
PYTHONPATH=offline2online_gcrl/impls:. python offline2online_gcrl/impls/main_off_on.py \
    --train_config_path=configs/train_config.yaml \
    --offline_agent=offline2online_gcrl/impls/agents/gcivl_jax.py \
    --offline_agent.batch_size=100
```

Settings are read from [configs/train_config.yaml](configs/train_config.yaml). Agent hyperparameters come from the
agent file and can be overridden on the command line (`--offline_agent.<name>=<value>`). The same agent is used in
both phases.

| Setting | Meaning |
|---|---|
| `env_name`, `dataset_name` | Environment and offline dataset (`<dataset_folder>/<dataset_name>.npz`). |
| `num_envs` | Parallel environments. `(episode_length - 1) * num_envs` must be divisible by the agent's `batch_size`. |
| `offline_train_steps`, `online_train_steps` | Length of each phase, in environment-step units. |
| `num_training_steps_per_iteration` | Training steps per collected unroll (update-to-data ratio). |
| `insert_only_successful_episodes` | Keep only episodes that reach their goal in the online buffer. |
| `action_sample_temperature` | Scales the policy's action noise during online collection. |
| `max_offline_transitions` | Use only the first N offline transitions (`null` for all). |
| `visualization_interval` | Render an HTML rollout every N epochs; `0` disables rendering. |
| `log_wandb` | Log metrics to Weights & Biases. |

## Agents

| Agent | File | Notes |
|---|---|---|
| GCIVL | `agents/gcivl_jax.py` | Goal-conditioned implicit V-learning with an AWR actor. |
| GCIQL | `agents/gciql_jax.py` | Goal-conditioned implicit Q-learning; DDPG+BC (default) or AWR actor. |
| GCQL | `agents/gcql_jax.py` | Goal-conditioned Q-learning; DDPG (default) or AWR actor. |
| IQL | `agents/iql.py` | Non-goal-conditioned baseline; run with `goal_conditioning: 0`. |

Continuous-action actors use a state-dependent standard deviation and tanh squashing, so actions stay in (-1, 1).

## Environments

| Name | Task | Control |
|---|---|---|
| `arm_binpick_easy` | Move a cube from a random spot in one bin to the centre of another (5-dim actions) | Joint angles of the Franka arm |
| `arm_binpick_easy_EEF` | Same task with a free-floating gripper (4-dim actions) | End-effector position deltas |

Both are adapted from JaxGCRL's manipulation suite. Success means the cube is within `goal_reach_thresh` of the goal
(1 cm for the EEF task), and an episode ends as soon as the goal is reached.

## Acknowledgements and licence

Code derived from OGBench (MIT) and JaxGCRL (Apache-2.0) keeps its original licence; see [NOTICE.md](NOTICE.md) for
which files come from where, [LICENSE](LICENSE) and [LICENSE-JAXGCRL](LICENSE-JAXGCRL).
