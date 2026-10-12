import os
import pickle
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from brax.training.replay_buffers_test import jit_wrap

from offline2online_gcrl.impls.utils.env_utils import get_experience
from offline2online_gcrl.impls.utils.replay_buffer import TrajectoryUniformSamplingQueue, Transition

_PMAP_AXIS_NAME = "i"


def create_dummy_transition(env):
    obs_size = env.observation_size
    action_size = env.action_size
    dummy_obs = jnp.zeros((obs_size,))  # concat of [state, goal]
    dummy_action = jnp.zeros((action_size,))
    dummy_extras = {
        "state_extras": {"truncation": 0.0, "traj_id": 0.0},
        "policy_extras": {},
    }
    dummy_transition = Transition(
        observation=dummy_obs,
        next_observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        terminal=0.0,
        extras=dummy_extras,
    )
    return dummy_transition


def create_replay_buffer(
    sample_batch_size,
    replay_buffer_size,
    episode_length,
    num_envs,
    device_count,
    dummy_transition,
):
    replay_buffer = TrajectoryUniformSamplingQueue(
        max_replay_size=replay_buffer_size // device_count,
        dummy_data_sample=dummy_transition,
        # TODO: modify to handle sampling_ratio
        sample_batch_size=sample_batch_size // device_count // 2,  # divide by 2 to sample equally from online and offline buffers
        num_envs=num_envs,
        episode_length=episode_length,
    )
    replay_buffer = jit_wrap(replay_buffer)  # jit wrap internal functions
    return replay_buffer


def reshape_offline_data_for_pmap(data: Dict[str, np.ndarray], num_local_devices_to_use: int, num_envs: int) -> Dict[str, jnp.ndarray]:
    """Split stacked transitions [num_transitions, ...] into [devices, samples_per_env, num_envs, ...]."""
    total_samples = len(data["observation"])

    # Ensure total_samples is a multiple of (num_local_devices * num_envs)
    num_groups = num_local_devices_to_use * num_envs
    max_usable_samples = (total_samples // num_groups) * num_groups
    samples_per_env = max_usable_samples // num_groups  # Each environment gets a contiguous batch

    print(f"Using {max_usable_samples} samples in offline buffer")

    # Reshape to ensure contiguous data assignment per environment per device
    reshaped_dict = {}
    for key, value in data.items():
        value = value[:max_usable_samples].reshape(num_local_devices_to_use, num_envs, samples_per_env, *value.shape[1:])
        reshaped_dict[key] = jnp.asarray(value.swapaxes(1, 2))  # Swap env & batch dimensions to keep contiguous samples together

    return reshaped_dict


def stack_transitions(transitions: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Convert the legacy list-of-dicts dataset format into one stacked array per key."""
    return {key: np.stack([np.asarray(t[key]) for t in transitions]) for key in transitions[0]}


def load_offline_data(dataset_folder, dataset_name, max_transitions=None) -> Dict[str, np.ndarray]:
    """Load an offline dataset as a dict of stacked arrays [num_transitions, ...].

    Prefers `<dataset_name>.npz`. Falls back to the legacy `<dataset_name>.pkl` (a list of per-transition dicts), which
    uses far more memory to load; convert it once with `scripts/convert_dataset_to_npz.py`.
    """
    dataset_path = os.path.join(dataset_folder, dataset_name)
    if os.path.exists(f"{dataset_path}.npz"):
        with np.load(f"{dataset_path}.npz") as f:
            data = {key: f[key] for key in f.files}
    else:
        logging.warning("Loading legacy dataset %s.pkl. Convert it to .npz to reduce memory use.", dataset_path)
        with open(f"{dataset_path}.pkl", "rb") as f:
            data = stack_transitions(pickle.load(f))
    if max_transitions is not None:
        data = {key: value[:max_transitions] for key, value in data.items()}
    return data


def fill_offline_replay_buffer(offline_train_data, offline_buffer_state, offline_replay_buffer):
    """
    Inserts offline data into the replay buffer.
    """

    @jax.jit
    def f(carry, data):
        """
        Convert each dictionary sample into a `Transition` object.
        """
        (
            observation,
            next_observation,
            action,
            reward,
            terminate,
            truncation,
            traj_id,
        ) = data
        state_extras = {"truncation": truncation, "traj_id": traj_id}
        transition = Transition(
            observation=observation,
            next_observation=next_observation,
            action=action,
            reward=reward,
            terminal=jnp.logical_or(terminate, truncation),
            extras={"policy_extras": {}, "state_extras": state_extras},
        )

        return carry, transition

    scan_data = tuple(
        offline_train_data[key]
        for key in [
            "observation",
            "next_observation",
            "action",
            "reward",
            "terminate",
            "truncation",
            "traj_id",
        ]
    )
    _, transitions = jax.lax.scan(f, None, scan_data)
    offline_buffer_state = offline_replay_buffer.insert(offline_buffer_state, transitions)

    return offline_buffer_state


fill_offline_replay_buffer = jax.pmap(
    fill_offline_replay_buffer,
    axis_name=_PMAP_AXIS_NAME,
    static_broadcasted_argnums=(2,),  # Treat the 3rd argument (index 2) as static
)


def setup_offline_replay_buffer(
    train_config,
    agent_config,
    dummy_transition,
    offline_rb_key,
    num_local_devices_to_use,
    device_count,
):
    offline_train_data = load_offline_data(train_config.dataset_folder, train_config.dataset_name, train_config.get("max_offline_transitions"))
    offline_train_data = reshape_offline_data_for_pmap(offline_train_data, num_local_devices_to_use, num_envs=train_config.num_envs)
    offline_train_data["action"] = jnp.clip(offline_train_data["action"], -0.999, 0.999)  # clip for tanhsquash

    ## create offline replay buffer sized to hold the whole dataset and initialise state
    rows_per_device = offline_train_data["observation"].shape[1]
    offline_replay_buffer = create_replay_buffer(
        sample_batch_size=agent_config["batch_size"],
        replay_buffer_size=rows_per_device * device_count,
        num_envs=train_config.num_envs,
        device_count=device_count,
        dummy_transition=dummy_transition,
        episode_length=train_config.episode_length,
    )
    offline_buffer_state = jax.pmap(offline_replay_buffer.init)(jax.random.split(offline_rb_key, num_local_devices_to_use))

    # fill buffer
    offline_buffer_state = fill_offline_replay_buffer(offline_train_data, offline_buffer_state, offline_replay_buffer)
    offline_replay_size = jnp.sum(jax.vmap(offline_replay_buffer.size)(offline_buffer_state)) * jax.process_count()
    logging.info("offline replay size after filling %s", offline_replay_size)
    return offline_replay_buffer, offline_buffer_state


def prefill_replay_buffer(
    agent,
    env,
    env_state,
    online_replay_buffer,
    online_buffer_state,
    key,
    num_env_steps_per_experience,
    num_prefill_experiences,
    action_sample_temperature,
    goal_conditioning,
    episode_length,
    unroll_length,
):
    def f(carry, unused):
        # Collect experience
        agent, env_state, online_buffer_state, key = carry
        key, new_key = jax.random.split(key)
        env_state, online_buffer_state = get_experience(
            agent,
            env,
            env_state,
            online_replay_buffer,
            online_buffer_state,
            key,
            action_sample_temperature,
            goal_conditioning,
            episode_length,
            unroll_length,
        )

        agent = agent.replace(network=agent.network.replace(env_steps=agent.network.env_steps + num_env_steps_per_experience))
        return (agent, env_state, online_buffer_state, new_key), ()

    return jax.lax.scan(
        f,
        (agent, env_state, online_buffer_state, key),
        (),
        length=num_prefill_experiences,
    )[0]


prefill_replay_buffer_pmap = jax.pmap(
    prefill_replay_buffer,
    axis_name=_PMAP_AXIS_NAME,
    static_broadcasted_argnums=(1, 3, 6, 7, 8, 9, 10, 11),
)


def prepare_transitions_for_training(train_config, agent_config, env, transitions_1, transitions_2, experience_key):
    experience_key, sampling_key_1, sampling_key_2 = jax.random.split(experience_key, 3)

    def flatten_gcrl(args):
        transitions_1, transitions_2, sampling_key_1, sampling_key_2 = args
        batch_keys_1 = jax.random.split(sampling_key_1, transitions_1.observation.shape[0])
        batch_keys_2 = jax.random.split(sampling_key_2, transitions_2.observation.shape[0])
        vmap_flatten_gcrl_fn = jax.vmap(TrajectoryUniformSamplingQueue.flatten_gcrl_fn, in_axes=(None, None, 0, 0))
        transitions_1 = vmap_flatten_gcrl_fn(agent_config["discount"], env, transitions_1, batch_keys_1)
        transitions_2 = vmap_flatten_gcrl_fn(agent_config["discount"], env, transitions_2, batch_keys_2)
        return transitions_1, transitions_2

    def flatten_rl(args):
        transitions_1, transitions_2, sampling_key_1, sampling_key_2 = args
        vmap_flatten_rl_fn = jax.vmap(TrajectoryUniformSamplingQueue.flatten_rl_fn, in_axes=(None, 0))
        transitions_1 = vmap_flatten_rl_fn(env, transitions_1)  # transitions: [num_envs, (episode_length - 1), feature shape]
        transitions_2 = vmap_flatten_rl_fn(env, transitions_2)
        return transitions_1, transitions_2

    transitions_1, transitions_2 = jax.lax.cond(
        train_config.goal_conditioning,
        flatten_gcrl,
        flatten_rl,
        (transitions_1, transitions_2, sampling_key_1, sampling_key_2),
    )

    transitions = jax.tree_util.tree_map(lambda x, y: jnp.concatenate([x, y], axis=1), transitions_1, transitions_2)

    ## Shuffle transitions and reshape them into (number_of_sgd_steps, batch_size, ...)
    transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions)
    permutation = jax.random.permutation(experience_key, len(transitions.observation))
    transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1, agent_config["batch_size"]) + x.shape[1:]),
        transitions,
    )
    return transitions
