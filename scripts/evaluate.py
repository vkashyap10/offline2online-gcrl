"""Evaluate a saved checkpoint and optionally render an HTML rollout.

Usage:
    PYTHONPATH=offline2online_gcrl/impls:. python scripts/evaluate.py \
        --train_config_path=configs/train_config.yaml \
        --agent=offline2online_gcrl/impls/agents/gcivl_jax.py \
        --checkpoint=path/to/final.pkl --num_episodes=50 --render_dir=path/to/output
"""

import jax
from absl import app, flags
from brax.io.model import load_params
from ml_collections import config_flags

import wandb
from offline2online_gcrl.impls.utils.agent_setup import create_agent
from offline2online_gcrl.impls.utils.buffer_setup import create_dummy_transition
from offline2online_gcrl.impls.utils.config import load_config
from offline2online_gcrl.impls.utils.env_utils import create_wrapped_env
from offline2online_gcrl.impls.utils.evaluator import create_evaluator
from offline2online_gcrl.impls.utils.metrics_recorder import render

FLAGS = flags.FLAGS
flags.DEFINE_string("train_config_path", "./configs/train_config.yaml", "Training config used for the checkpoint.")
flags.DEFINE_string("checkpoint", None, "Path to a checkpoint saved during training (e.g. final.pkl).", required=True)
flags.DEFINE_integer("num_episodes", 50, "Number of evaluation episodes, run in parallel.")
flags.DEFINE_integer("seed", 0, "Random seed for evaluation.")
flags.DEFINE_string("render_dir", None, "If set, write an HTML rollout to this directory.")
config_flags.DEFINE_config_file("agent", "offline2online_gcrl/impls/agents/gcivl_jax.py", lock_config=False)


def main(_):
    train_config = load_config(FLAGS.train_config_path)
    train_config.num_eval_envs = FLAGS.num_episodes

    env, unwrapped_env = create_wrapped_env(train_config.env_name, train_config.episode_length, train_config.action_repeat)
    agent_key, eval_key = jax.random.split(jax.random.PRNGKey(FLAGS.seed))
    agent = create_agent(
        config=FLAGS.agent,
        rng=agent_key,
        env=env,
        dummy_transition=create_dummy_transition(env),
        goal_conditioning=train_config.goal_conditioning,
    )
    params = load_params(FLAGS.checkpoint)

    evaluator, make_policy = create_evaluator(train_config, agent, env, eval_key)
    metrics = evaluator.run_evaluation(params, training_metrics={})
    # Episodes end on success, so the mean of episode_success is the success rate.
    print(f"success rate: {float(metrics['eval/episode_success']):.3f}")
    print(f"episodes with any success: {float(metrics['eval/episode_success_any']):.3f}")
    print(f"mean episode length: {float(metrics['eval/avg_episode_length']):.1f}")

    if FLAGS.render_dir:
        wandb.init(mode="disabled")
        render(make_policy, params, unwrapped_env, FLAGS.render_dir, "eval", 0)
        print(f"Rendered rollout written to {FLAGS.render_dir}/eval_0.html")


if __name__ == "__main__":
    app.run(main)
