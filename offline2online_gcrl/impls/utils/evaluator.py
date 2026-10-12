import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from brax.training import acting
from brax.training.types import Metrics, PolicyParams


class Evaluator(acting.Evaluator):
    """
    CrlEvaluator class for running evaluation epochs in a training process.

    This class extends the base acting.Evaluator and is designed to evaluate the
    performance of policies based on given parameters and training metrics.
    It provides detailed evaluations by aggregating various metrics over episodes
    and calculating statistical summaries such as mean, standard deviation,
    maximum, and minimum values.

    This is an evaluator that behaves in the exact same way as brax Evaluator,
    but additionally it aggregates metrics with max, min.
    It also logs in how many episodes there was any success.
    """

    def run_evaluation(
        self,
        policy_params: PolicyParams,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(policy_params, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        aggregating_fns = [
            (np.mean, ""),
            (np.std, "_std"),
            (np.max, "_max"),
            (np.min, "_min"),
        ]

        for fn, suffix in aggregating_fns:
            metrics.update({f"eval/episode_{name}{suffix}": (fn(value) if aggregate_episodes else value) for name, value in eval_metrics.episode_metrics.items()})

        # We check in how many env there was at least one step where there was success
        if "success" in eval_metrics.episode_metrics:
            metrics["eval/episode_success_any"] = np.mean(eval_metrics.episode_metrics["success"] > 0.0)

        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}

        return metrics  # pytype: disable=bad-return-type  # jax-ndarray


def create_evaluator(train_config, agent, eval_env, eval_key):
    # The brax version of this does not take in the actor and action_distribution arguments; before we pass it to brax evaluator or return it from train(), we do a partial application.
    def make_policy(
        agent,
        env,
        params,
        temperature=0.0,
        goal_conditioning=train_config.goal_conditioning,
    ):
        def policy(observations, key_sample=None):
            """
            Sample actions from the actor.
            Observation in the env is state + goal
            """
            states = observations[:, : env.state_dim]
            goals = observations[:, env.state_dim :]

            if goal_conditioning:
                dist = agent.network.select("actor")(states, goals, params=params, temperature=temperature)
            else:
                dist = agent.network.select("actor")(states, params=params, temperature=temperature)
            actions = dist.sample(seed=key_sample)
            actions = jnp.clip(actions, -0.9999, 0.9999)  # modified to work with tanh_squash
            extras = {}
            return actions, extras

        return policy

    make_policy = functools.partial(
        make_policy,
        agent,
        eval_env,
        temperature=0.0,
        goal_conditioning=train_config.goal_conditioning,
    )
    evaluator = Evaluator(
        eval_env,
        make_policy,
        num_eval_envs=train_config.num_eval_envs,
        episode_length=train_config.episode_length,
        action_repeat=train_config.action_repeat,
        key=eval_key,
    )
    return evaluator, make_policy
