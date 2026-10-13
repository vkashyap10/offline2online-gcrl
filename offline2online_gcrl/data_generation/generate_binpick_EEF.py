"""Generate an offline dataset for arm_binpick_easy_EEF with a scripted pick-and-place controller.

Each episode runs the oracle with Gaussian action noise (scale drawn uniformly from [0, noise] per episode) until the
oracle finishes or the environment terminates (goal reached) or truncates. Every episode is kept.

Usage:
    python -m offline2online_gcrl.data_generation.generate_binpick_EEF --num_episodes=1000 \
        --save_path=./offline2online_gcrl/data_generation/data/arm_binpick_easy_EEF.npz
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from brax import envs
from brax.io import html
from tqdm import trange

from offline2online_gcrl.data_generation.oracles.markov.binpick_EEF_oracle import BinpickEEFMarkovOracle
from offline2online_gcrl.environments.wrappers import TrajectoryIdWrapper
from offline2online_gcrl.impls.utils.env_utils import create_brax_env

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("save_path", "./offline2online_gcrl/data_generation/data/arm_binpick_easy_EEF.npz", "Output .npz path.")
flags.DEFINE_float("noise", 0.1, "Maximum action noise scale.")
flags.DEFINE_float("min_norm", 0.4, "Minimum action norm for the oracle.")
flags.DEFINE_integer("num_episodes", 1000, "Number of episodes.")
flags.DEFINE_integer("max_episode_steps", 1001, "Episode length before truncation.")
flags.DEFINE_integer("save_interval", 500, "Save the dataset collected so far every N episodes.")
flags.DEFINE_string("viz_dir", None, "If set, write an HTML rollout every save_interval episodes to this directory.")

ENV_NAME = "arm_binpick_easy_EEF"
FIELDS = ["observation", "next_observation", "action", "reward", "terminate", "truncation", "traj_id"]


def get_info(env_state, env):
    """Oracle inputs read from the observation (batch of one environment)."""
    obs = env_state.obs[0]
    return {
        "proprio/effector_pos": np.asarray(obs[env.eef_pos_indices]),
        "proprio/effector_yaw": np.array([0.0]),
        "privileged/target_block": env.target_block,
        f"privileged/block_{env.target_block}_pos": np.asarray(obs[env.goal_indices]),
        f"privileged/block_{env.target_block}_yaw": np.array([0.0]),
        "privileged/target_block_pos": np.asarray(obs[-3:]),
        "privileged/target_block_yaw": np.array([0.0]),
        "proprio/gripper_contact": np.asarray(obs[env.gripper_distance_index]).squeeze(),
        "proprio/gripper_force": np.asarray(obs[env.gripper_force_index]).squeeze(),
    }


def save(transitions, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **{key: np.stack([t[key] for t in transitions]) for key in FIELDS})
    print(f"[Saved {len(transitions)} transitions to {path}]")


def main(_):
    env = create_brax_env(env_name=ENV_NAME)
    env = TrajectoryIdWrapper(env)
    env = envs.training.wrap(env, episode_length=FLAGS.max_episode_steps, action_repeat=1)
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    oracle = BinpickEEFMarkovOracle(env=env, min_norm=FLAGS.min_norm)

    key = jax.random.PRNGKey(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    transitions = []

    for ep_idx in trange(FLAGS.num_episodes):
        key, subkey = jax.random.split(key)
        env_state = jit_env_reset(rng=subkey[None, :])
        info = get_info(env_state, env)
        oracle.reset(env_state.obs[0], info)
        xi = np.random.uniform(0, FLAGS.noise)  # Action noise scale for this episode.
        rollout_viz = []

        while True:
            rollout_viz.append(jax.tree_util.tree_map(lambda x: x[0], env_state.pipeline_state))

            action = oracle.select_action(env_state.obs[0], info)
            action = action + np.random.normal(0, [xi, xi, xi, 0, xi * 2], action.shape)
            action = np.clip(np.delete(action, 3), -0.999, 0.999)  # Drop yaw, which the environment does not control.

            next_env_state = jit_env_step(env_state, jnp.asarray(action)[None, :])
            transitions.append(
                {
                    "observation": np.asarray(env_state.obs[0]),
                    "next_observation": np.asarray(next_env_state.obs[0]),
                    "action": action.astype(np.float32),
                    "reward": np.asarray(next_env_state.reward[:1], dtype=np.float32),
                    "terminate": np.asarray(next_env_state.reward[:1], dtype=np.float32),  # The episode ends on success.
                    "truncation": np.asarray(next_env_state.info["truncation"][:1], dtype=np.float32),
                    "traj_id": np.array([ep_idx + 1], dtype=np.int64),
                }
            )
            env_state = next_env_state
            info = get_info(env_state, env)

            if oracle.done or bool(env_state.done[0]):
                break

        if FLAGS.save_interval and ep_idx % FLAGS.save_interval == 0:
            save(transitions, FLAGS.save_path)
            if FLAGS.viz_dir:
                os.makedirs(FLAGS.viz_dir, exist_ok=True)
                url = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout_viz, height=1024)
                with open(os.path.join(FLAGS.viz_dir, f"rollout_{ep_idx}.html"), "w") as f:
                    f.write(url)

    save(transitions, FLAGS.save_path)


if __name__ == "__main__":
    app.run(main)
