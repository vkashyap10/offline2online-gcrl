"""
Offline to Online RL framework implemented with JAX.
"""
import os

from absl import app, flags
from ml_collections import config_flags

import wandb
from offline2online_gcrl.impls.utils.config import load_config
from offline2online_gcrl.impls.utils.metrics_recorder import MetricsRecorder
from offline2online_gcrl.impls.utils.training import train_fn

FLAGS = flags.FLAGS
flags.DEFINE_string("train_config_path", "./configs/train_config.yaml", "Folder with offline dataset")

config_flags.DEFINE_config_file("offline_agent", "offline2online_gcrl/impls/agents/gcivl_jax.py", lock_config=False)
config_flags.DEFINE_config_file("online_agent", "offline2online_gcrl/impls/agents/gcivl_jax.py", lock_config=False)


def main(_):
    # set up configs
    train_config = load_config(FLAGS.train_config_path)

    # TODO: switch between offline and onilne configs based on training mode
    offline_agent_config = FLAGS.offline_agent
    online_agent_config = FLAGS.online_agent

    metrics_to_collect = [
        "eval/episode_success",
        "eval/episode_success_any",
        "eval/episode_success_hard",
        "eval/episode_success_easy",
        "eval/episode_dist",
        "eval/episode_reward_survive",
        "training/sps",
        "training/offline_buffer_size",
        "training/online_buffer_size",
        "training/num_gradient_updates",
        "training/num_environment_steps",
        "training/is_online",
    ]

    train_config.save_dir = os.path.join(
        train_config.save_dir,
        train_config.project_name,
        train_config.experiment_name,
        train_config.job_name,
    )
    os.makedirs(train_config.save_dir, exist_ok=True)
    metrics_to_collect.extend(offline_agent_config["metrics"])  # online agent?
    metrics_recorder = MetricsRecorder(
        train_config.offline_train_steps + train_config.online_train_steps,
        metrics_to_collect,
        train_config.save_dir,
        train_config.experiment_name,
    )
    progress_fn = metrics_recorder.progress

    # setup wandb and logger
    wandb.init(
        project=train_config.project_name,
        group=train_config.experiment_name,
        name=train_config.job_name,
        mode="online" if train_config.log_wandb else "disabled",
    )

    train_fn(train_config, offline_agent_config, online_agent_config, progress_fn)


if __name__ == "__main__":
    app.run(main)
