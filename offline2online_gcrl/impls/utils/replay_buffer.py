import functools
from typing import Generic, NamedTuple, Tuple, TypeVar

import jax
from brax.training.acme.types import NestedArray
from brax.training.replay_buffers import ReplayBuffer, ReplayBufferState
from brax.training.types import PRNGKey
from jax import flatten_util
from jax import numpy as jnp

Sample = TypeVar("Sample")


class QueueBase(ReplayBuffer[ReplayBufferState, Sample], Generic[Sample]):
    """Base class for limited-size FIFO reply buffers.

    Implements an `insert()` method which behaves like a limited-size queue.
    I.e. it adds samples to the end of the queue and, if necessary, removes the
    oldest samples form the queue in order to keep the maximum size within the
    specified limit.

    Derived classes must implement the `sample()` method.
    """

    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample: Sample,
        sample_batch_size: int,
        num_envs: int,
        episode_length: int,
    ):
        self._flatten_fn = jax.vmap(jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0]))

        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(jax.vmap(self._unflatten_fn))
        data_size = len(dummy_flatten)

        self._data_shape = (max_replay_size, num_envs, data_size)
        self._data_dtype = dummy_flatten.dtype
        self._sample_batch_size = sample_batch_size
        self._size = 0
        self.num_envs = num_envs
        self.episode_length = episode_length

    def init(self, key: PRNGKey) -> ReplayBufferState:
        return ReplayBufferState(
            data=jnp.zeros(self._data_shape, self._data_dtype),
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def check_can_insert(self, buffer_state, samples, shards):
        """Checks whether insert operation can be performed."""
        assert isinstance(shards, int), "This method should not be JITed."
        insert_size = jax.tree_util.tree_flatten(samples)[0][0].shape[0] // shards
        if self._data_shape[0] < insert_size:
            raise ValueError("Trying to insert a batch of samples larger than the maximum replay" f" size. num_samples: {insert_size}, max replay size" f" {self._data_shape[0]}")
        self._size = min(self._data_shape[0], self._size + insert_size)

    def insert_internal(self, buffer_state: ReplayBufferState, samples: Sample) -> ReplayBufferState:
        """Insert data in the replay buffer.

        Args:
          buffer_state: Buffer state
          samples: Sample to insert with a leading batch size.

        Returns:
          New buffer state.
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(f"buffer_state.data.shape ({buffer_state.data.shape}) " f"doesn't match the expected value ({self._data_shape})")

        update = self._flatten_fn(samples)
        data = buffer_state.data

        # If needed, roll the buffer to make sure there's enough space to fit
        # `update` after the current position.
        position = buffer_state.insert_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll

        # Update the buffer and the control numbers.
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + len(update)) % (len(data) + 1)
        sample_position = jnp.maximum(0, buffer_state.sample_position + roll)

        return buffer_state.replace(
            data=data,
            insert_position=position,
            sample_position=sample_position,
        )

    def insert_rows(self, buffer_state: ReplayBufferState, samples: Sample, num_rows) -> ReplayBufferState:
        """Insert only the first `num_rows` rows of `samples` (a traced value).

        All rows are written so that shapes stay static under jit, but the insert position only advances by `num_rows`.
        Rows past the insert position are never sampled and get overwritten by the next insert.
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(f"buffer_state.data.shape ({buffer_state.data.shape}) " f"doesn't match the expected value ({self._data_shape})")

        update = self._flatten_fn(samples)
        data = buffer_state.data

        # Roll so that the full `update` fits after the current position, dropping the oldest rows.
        position = buffer_state.insert_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll

        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        sample_position = jnp.maximum(0, buffer_state.sample_position + roll)

        return buffer_state.replace(
            data=data,
            insert_position=position + num_rows,
            sample_position=sample_position,
        )

    def sample_internal(self, buffer_state: ReplayBufferState) -> Tuple[ReplayBufferState, Sample]:
        raise NotImplementedError(f"{self.__class__}.sample() is not implemented.")

    def size(self, buffer_state: ReplayBufferState) -> int:
        return buffer_state.insert_position - buffer_state.sample_position  # pytype: disable=bad-return-type  # jax-ndarray


class Transition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray  # concat of [state, goal]
    next_observation: NestedArray
    action: NestedArray
    reward: NestedArray
    terminal: NestedArray  # end of episode (terminate or truncate)
    extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray


class TrajectoryUniformSamplingQueue(QueueBase[Sample], Generic[Sample]):
    """Implements an uniform sampling limited-size replay queue BUT WITH TRAJECTORIES."""

    def sample_internal(self, buffer_state: ReplayBufferState) -> Tuple[ReplayBufferState, Sample]:
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(f"Data shape expected by the replay buffer ({self._data_shape}) does " f"not match the shape of the buffer state ({buffer_state.data.shape})")
        key, sample_key, shuffle_key = jax.random.split(buffer_state.key, 3)
        # NOTE: this is the number of envs to sample but it can be modified if there is OOM
        shape = self.num_envs

        # Sampling envs idxs
        envs_idxs = jax.random.choice(sample_key, jnp.arange(self.num_envs), shape=(shape,), replace=False)

        @functools.partial(jax.jit, static_argnames=("rows", "cols"))
        def create_matrix(rows, cols, min_val, max_val, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            start_values = jax.random.randint(subkey, shape=(rows,), minval=min_val, maxval=max_val)
            row_indices = jnp.arange(cols)
            matrix = start_values[:, jnp.newaxis] + row_indices
            return matrix

        @jax.jit
        def create_batch(arr_2d, indices):
            return jnp.take(arr_2d, indices, axis=0, mode="wrap")

        create_batch_vmaped = jax.vmap(create_batch, in_axes=(1, 0))

        matrix = create_matrix(
            shape,
            self.episode_length,
            buffer_state.sample_position,
            buffer_state.insert_position - self.episode_length,
            sample_key,
        )

        batch = create_batch_vmaped(buffer_state.data[:, envs_idxs, :], matrix)
        transitions = self._unflatten_fn(batch)
        return buffer_state.replace(key=key), transitions

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["env"])
    def flatten_rl_fn(env, transition: Transition) -> Transition:
        """
        Process transitions for RL training.
        - Truncation: End of episode without reaching goal state.
        - Termination: End of episode on reaching goal state.

        Because it's vmaped transition obs.shape is of shape (transitions,obs_dim)

        Args:
            env: Environment containing state dimensions.
            transition: A `Transition` object containing trajectory data.

        Returns:
            Processed `Transition` with truncated transitions removed.
        """

        truncation = transition.extras["state_extras"]["truncation"][:-1]
        non_truncated_mask = jnp.logical_not(truncation)

        # TODO: what if episode terminates and truncates at the same time step? highly unlikely if enough env steps.
        terminate = jnp.logical_and(transition.terminal[:-1], non_truncated_mask)
        masks = jnp.where(terminate > 0, 0, 1)

        state = transition.observation[:-1, : env.state_dim]
        next_state = transition.next_observation[:-1, : env.state_dim]

        extras = {
            "policy_extras": {},
            "state_extras": {
                "truncation": jnp.squeeze(truncation),
                "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
            },
            "states": state,
            "next_states": next_state,
            "masks": masks,
            "rewards": transition.reward[:-1],
            # To maintain compatibility with GC replay buffer. This is done so that we can have a single training script for gc and standard rl. These values are never used.
            "value_goals": transition.observation[:-1, env.state_dim :],
            "actor_goals": transition.observation[:-1, env.state_dim :],
        }

        return transition._replace(
            observation=jnp.squeeze(transition.observation[:-1]),
            next_observation=jnp.squeeze(transition.next_observation[:-1]),
            action=jnp.squeeze(transition.action[:-1]),
            reward=jnp.squeeze(transition.reward[:-1]),
            terminal=jnp.squeeze(transition.terminal[:-1]),
            extras=extras,
        )

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["discount", "env"])
    def flatten_gcrl_fn(discount, env, transition: Transition, sample_key: PRNGKey) -> Transition:
        gamma = discount
        state_size = env.state_dim
        goal_indices = env.goal_indices

        # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)  # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
        discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
        probs = is_future_mask * discount

        # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
        #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
        #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
        #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
        #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
        #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
        # assuming seq_len = 5
        # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))

        single_trajectories = jnp.concatenate(
            [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
            axis=0,
        )
        # array of seq_len x seq_len where a row is an array of traj_ids that correspond to the episode index from which that time-step was collected
        # timesteps collected from the same episode will have the same traj_id. All rows of the single_trajectories are same.

        probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
        # ith row of probs will be non zero only for time indices that
        # 1) are greater than i
        # 2) have the same traj_id as the ith time index
        # jax.debug.print("probs {}", probs)

        goal_index = jax.random.categorical(sample_key, jnp.log(probs))

        terminate = jnp.zeros(seq_len - 1, dtype=jnp.float32).at[goal_index[:-1]].set(1.0)
        rewards = jnp.where(terminate > 0.0, 1.0, 0.0)  # reward when goal state reached
        masks = jnp.where(terminate > 0, 0, 1)

        future_state = jnp.take(transition.observation, goal_index[:-1], axis=0)  # the last goal_index cannot be considered as there is no future.
        # future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
        goal = future_state[:, goal_indices]
        # future_state = future_state[:, :state_size]
        state = transition.observation[:-1, :state_size]  # all states are considered
        next_state = transition.next_observation[:-1, :state_size]
        new_obs = jnp.concatenate([state, goal], axis=1)
        new_next_obs = jnp.concatenate([next_state, goal], axis=1)

        extras = {
            "policy_extras": {},
            "state_extras": {
                "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
                "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
            },
            "states": state,
            "next_states": next_state,
            "masks": masks,
            "rewards": rewards,
            "value_goals": goal,
            "actor_goals": goal,
        }

        return transition._replace(
            observation=jnp.squeeze(new_obs),  # this has shape (num_envs, episode_length-1, obs_size)
            next_observation=jnp.squeeze(new_next_obs),
            action=jnp.squeeze(transition.action[:-1]),
            reward=rewards,
            terminal=terminate,
            extras=extras,
        )
