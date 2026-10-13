import math
import os
from datetime import datetime
from typing import List

import jax
from brax.io import html
from matplotlib import pyplot as plt

import wandb


class MetricsRecorder:
    """
    Initialize the MetricsRecorder with the specified number of timesteps
    and the metrics to be collected.

    Parameters:
    num_timesteps (int): The maximum number of timesteps for recording metrics.
    metrics_to_collect (List[str]): List of metric names that are to be collected.
    exp_dir (str): Directory to save renders to.
    exp_name (str): Experiment name for naming rendered trajectory visualizations.
    """

    def __init__(self, num_timesteps: int, metrics_to_collect: List[str], exp_dir, exp_name):
        self.x_data = []
        self.y_data = {}
        self.y_data_err = {}
        self.times = [datetime.now()]
        self.metrics_to_collect = metrics_to_collect
        self.exp_dir = exp_dir
        self.exp_name = exp_name

        self.max_x, self.min_x = num_timesteps * 1.1, 0

    def record(self, num_steps, metrics):
        self.times.append(datetime.now())
        self.x_data.append(num_steps)

        for key, value in metrics.items():
            if key not in self.y_data:
                self.y_data[key] = []
                self.y_data_err[key] = []

            self.y_data[key].append(value)
            self.y_data_err[key].append(metrics.get(f"{key}_std", 0))

    def write_csv(self):
        """Append the latest values of the collected metrics to <exp_dir>/metrics.csv."""
        path = os.path.join(self.exp_dir, "metrics.csv")
        columns = ["step"] + list(self.metrics_to_collect)
        latest = {key: values[-1] for key, values in self.y_data.items()}
        latest["step"] = self.x_data[-1]
        new_file = not os.path.exists(path)
        with open(path, "a") as f:
            if new_file:
                f.write(",".join(columns) + "\n")
            f.write(",".join(str(float(latest.get(key, float("nan")))) for key in columns) + "\n")

    def log_wandb(self):
        data_to_log = {}
        for key, value in self.y_data.items():
            data_to_log[key] = value[-1]
        data_to_log["step"] = self.x_data[-1]
        wandb.log(data_to_log, step=self.x_data[-1])

    def plot_progress(self):
        num_plots = len(self.y_data)
        num_rows = (num_plots + 1) // 2  # Calculate number of rows needed for 2 columns

        fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))

        for idx, (key, y_values) in enumerate(self.y_data.items()):
            row = idx // 2
            col = idx % 2

            axs[row, col].set_xlim(self.min_x, self.max_x)
            axs[row, col].set_xlabel("# environment steps")
            axs[row, col].set_ylabel(key)
            axs[row, col].errorbar(self.x_data, y_values, yerr=self.y_data_err[key])
            axs[row, col].set_title(f"{key}: {y_values[-1]:.3f}")

        # Hide any empty subplots
        for idx in range(num_plots, num_rows * 2):
            row = idx // 2
            col = idx % 2
            axs[row, col].axis("off")
        plt.tight_layout()
        plt.show()

    def print_progress(self):
        for idx, (key, y_values) in enumerate(self.y_data.items()):
            print(f"step: {self.x_data[-1]}, {key}: {y_values[-1]:.3f} +/- {self.y_data_err[key][-1]:.3f}")

    def print_times(self):
        print(f"time to jit: {self.times[1] - self.times[0]}")
        print(f"time to train: {self.times[-1] - self.times[1]}")

    def progress(self, num_steps, metrics, make_policy, params, env, do_render=True):
        for key in self.metrics_to_collect:
            self.ensure_metric(metrics, key)

        if do_render:
            render(make_policy, params, env, self.exp_dir, self.exp_name, num_steps)

        self.record(
            num_steps,
            {key: value for key, value in metrics.items() if key in self.metrics_to_collect},
        )
        self.log_wandb()
        self.write_csv()
        self.print_progress()

    @staticmethod
    def ensure_metric(metrics, key):
        if key not in metrics:
            metrics[key] = 0
        else:
            if math.isnan(metrics[key]):
                raise Exception(f"Metric: {key} is Nan")


def render(make_policy, params, env, exp_dir, exp_name, num_steps):
    """
    Renders a given environment over a series of steps and stores the resulting
    HTML file to a specified directory. Logs the rendered HTML using wandb.

    This function initializes the environment and the inference function, then
    runs the environment for a fixed number of steps, periodically resetting.
    It collects the state of the environment at each step, renders the HTML,
    stores the result, and logs it.

    Parameters:
    inf_fun_factory : Callable
        A factory function that returns an inference function when provided with
        'params'.
    params : any
        Parameters for the 'inf_fun_factory'.
    env : object
        The environment object with 'reset', 'step', and 'sys.tree_replace' methods.
    exp_dir : str
        The directory where the rendered HTML file will be saved.
    exp_name : str
        The file name to be used for the saved HTML (without extension).
    num_steps : int
        The number of environment steps taken so far (used for naming the file).

    Returns:
    None
    """
    policy = make_policy(params)
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_policy = jax.jit(policy)

    rollout = []
    key = jax.random.PRNGKey(seed=1)
    key, subkey = jax.random.split(key)
    state = jit_env_reset(rng=subkey)
    for i in range(5000):
        rollout.append(state.pipeline_state)
        key, subkey = jax.random.split(key)
        action, _ = jit_policy(state.obs[None], subkey)  # Policy requires batched dimension
        action = action[0]  # Remove batch dimension
        state = jit_env_step(state, action)
        if i % 1000 == 0:
            key, subkey = jax.random.split(key)
            state = jit_env_reset(rng=subkey)

    url = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout, height=1024)
    with open(os.path.join(exp_dir, f"{exp_name}_{num_steps}.html"), "w") as file:
        file.write(url)
    wandb.log({"render": wandb.Html(url)})
