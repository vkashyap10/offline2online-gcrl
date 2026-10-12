import jax
from brax import envs
from jax import numpy as jnp

from offline2online_gcrl.environments.manipulation.arm_binpick import ArmBinpickEasy
from offline2online_gcrl.environments.manipulation.arm_binpick_EEF import ArmBinpickEasyEEF
from offline2online_gcrl.environments.wrappers import TrajectoryIdWrapper
from offline2online_gcrl.impls.utils.replay_buffer import Transition


def create_brax_env(env_name: str, backend: str = None, **kwargs) -> object:
    """
    This function creates and returns an appropriate environment object based on the specified environment name and
    backend.

    Args:
        env_name (str): Name of the environment.
        backend (str): Backend to be used for the environment.

    Returns:
        object: The instantiated environment object.

    Raises:
        ValueError: If the specified environment name is unknown.
    """
    if env_name == "arm_binpick_easy":
        env = ArmBinpickEasy(backend=backend or "mjx")
    elif env_name == "arm_binpick_easy_EEF":
        env = ArmBinpickEasyEEF(backend=backend or "mjx")
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env


def create_wrapped_env(env_name, episode_length, action_repeat):
    """
    envs.training.wrap:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap wrapped.

    brax.envs.wrappers.training.EpisodeWrapper: Maintains episode step count and sets done at episode end.

    brax.envs.wrappers.training: Automatically resets Brax envs that are done.
    This does not randomise initial and goal states. See: https://github.com/google/brax/discussions/452


    """
    unwrapped_env = create_brax_env(env_name)
    wrap_for_training = envs.training.wrap
    env = TrajectoryIdWrapper(unwrapped_env)
    env = wrap_for_training(env, episode_length=episode_length, action_repeat=action_repeat)
    return env, unwrapped_env


# Functions for training
def environment_step(
    env,
    env_state,
    agent,
    key,
    action_sample_temperature,
    goal_conditioning,
    extra_fields=(),
):
    """
    Executes one step of an actor in the environment by selecting an action based on the
    policy, stepping the environment, and returning the updated state and transition data.

    Returns
    -------
    Tuple[State, Transition]
        A tuple containing the new state after taking the action and the transition data
        encompassing observation, action, reward, discount, and extra information.

    """

    expl_rng, key = jax.random.split(key)

    def sample_actions_with_goals(args):
        """
        args is a tuple (obs, rng).
        """
        obs, rng = args
        return agent.sample_actions(
            observations=obs[:, : env.state_dim],
            goals=obs[:, env.state_dim :],  # goal portion
            seed=rng,
            temperature=action_sample_temperature,
        )

    def sample_actions_no_goals(args):
        """
        args is a tuple (obs, rng).
        """
        obs, rng = args
        return agent.sample_actions(
            observations=obs[:, : env.state_dim],
            goals=obs[:, env.state_dim :],
            seed=rng,
            temperature=action_sample_temperature,
        )

    action = jax.lax.cond(
        bool(goal_conditioning),
        sample_actions_with_goals,
        sample_actions_no_goals,
        (env_state.obs, expl_rng),
    )

    new_env_state = env.step(env_state, action)
    state_extras = {x: new_env_state.info[x] for x in extra_fields}
    return new_env_state, Transition(
        observation=env_state.obs,
        next_observation=new_env_state.obs,
        action=action,
        reward=new_env_state.reward,
        terminal=new_env_state.done,  # terminate or truncate
        extras={"policy_extras": {}, "state_extras": state_extras},
    )


def get_experience(
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
    success_only: bool = False,
):
    """Collect `unroll_length` steps per env and insert them into the online buffer.

    If `success_only`, only transitions from episodes that reach the goal within this unroll are inserted.
    """

    # ---------------------------------------------------------------- roll-out
    @jax.jit
    def one_step(carry, _):
        st, rng = carry
        rng, nxt = jax.random.split(rng)
        st, tr = environment_step(
            env,
            st,
            agent,
            rng,
            action_sample_temperature,
            goal_conditioning,
            extra_fields=("truncation", "traj_id"),
        )

        return (st, nxt), tr

    (env_state, _), data = jax.lax.scan(one_step, (env_state, key), (), length=unroll_length)

    if success_only:
        packed, num_rows = pack_successful_episodes(data)
        online_buffer_state = jax.lax.cond(
            num_rows > 0,
            lambda s: online_replay_buffer.insert_rows(s, packed, num_rows),
            lambda s: s,
            online_buffer_state,
        )
    else:
        online_buffer_state = online_replay_buffer.insert(online_buffer_state, data)

    return env_state, online_buffer_state


def pack_successful_episodes(data: Transition):
    """Keep transitions from episodes that succeed within the unroll and pack them into the top rows.

    Episodes end early on success (done = success), so their lengths vary and they are identified by traj_id rather
    than by fixed-length chunks. An episode still running at the end of the unroll has not succeeded yet and is dropped.

    Kept transitions are laid out env by env, in time order, as one stream that fills each buffer column (env slot) to
    the same height, so an episode can continue from the bottom of one column to the top of the next. The last column
    is padded with copies of the final kept transition, which is the goal-reaching step of its episode.

    Args:
        data: Transitions of shape [unroll_length, num_envs, ...].

    Returns:
        packed: Transitions with the same shape as `data`. Only the first `num_rows` rows are valid.
        num_rows: Number of valid rows, ceil(num_kept / num_envs). 0 if nothing succeeded.
    """
    T, N = data.reward.shape[:2]
    traj_id = data.extras["state_extras"]["traj_id"]  # [T, N], non-decreasing in time for each env

    def episode_succeeded(ids, success):
        segments = (ids - ids[0]).astype(jnp.int32)
        return jax.ops.segment_max(success, segments, num_segments=T)[segments]

    keep = jax.vmap(episode_succeeded, in_axes=1, out_axes=1)(traj_id, (data.reward > 0).astype(jnp.int32)) > 0

    # traj_id counts episodes per env, so ids from different envs collide once they share a column.
    # Make them unique across envs; goal relabelling only pairs states with equal traj_id.
    unique_id = traj_id * N + jnp.arange(N)[None, :]
    data = data._replace(
        extras={
            **data.extras,
            "state_extras": {**data.extras["state_extras"], "traj_id": unique_id},
        }
    )

    # Stream index = env * T + t. Stable argsort moves kept transitions to the front in stream order.
    keep_stream = keep.T.reshape(-1)
    order = jnp.argsort(jnp.logical_not(keep_stream), stable=True)
    num_kept = jnp.sum(keep_stream)
    num_rows = (num_kept + N - 1) // N

    rows = jnp.arange(T)[:, None]
    cols = jnp.arange(N)[None, :]
    stream_idx = jnp.minimum(cols * num_rows + rows, jnp.maximum(num_kept - 1, 0))
    src = order[stream_idx]  # [T, N] indices into the env-major stream

    def gather(x):
        stream = jnp.swapaxes(x, 0, 1).reshape(T * N, *x.shape[2:])
        return stream[src]

    return jax.tree_util.tree_map(gather, data), num_rows
