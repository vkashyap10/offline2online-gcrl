import os
import pickle
from functools import partial
from importlib.resources import files

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from brax import envs, kinematics
from brax.io import html, mjcf
from jaxopt import GaussNewton
from tqdm import trange

from offline2online_gcrl.data_generation.oracles.markov.binpick_oracle import OracleParams, OracleState, select_action
from offline2online_gcrl.environments import assets
from offline2online_gcrl.environments.wrappers import TrajectoryIdWrapper
from offline2online_gcrl.impls.utils.env_utils import create_brax_env

FLAGS = flags.FLAGS

# total number of episodes collect: num_episodes * num_envs
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "arm_binpick_easy", "Environment name.")
flags.DEFINE_integer(
    "num_episodes",
    1,
    "Number of parallel rollouts to record (each rollout runs num_envs in parallel).",
)
flags.DEFINE_integer("max_episode_steps", 201, "Max steps per environment rollout.")
flags.DEFINE_integer("num_envs", 2, "Number of parallel environments per rollout.")
flags.DEFINE_string(
    "save_path",
    "./offline2online_gcrl/data_generation/data/temp.pkl",
    "Path to save dataset.",
)
flags.DEFINE_integer("save_interval", 1, "Save interval.")
flags.DEFINE_float("noise", 0.002, "Action noise level.")
flags.DEFINE_bool("html_viz", True, "Render HTML of environment 0 per rollout if True.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(SCRIPT_DIR, "exp")

################################################################################
# LOCKED vs. ACTIVE JOINTS CONFIG
################################################################################
# We have 7 total Panda joints: indices [0..6].
# Suppose we want to fix joints #2 and #4 (the “3rd and 5th” in 0-based indexing).
FIXED_JOINTS = jnp.array([2, 4])
ACTIVE_JOINTS = jnp.array([0, 1, 3, 5, 6])

# Full 7-D min/max angles:
FULL_MIN = jnp.array([0.3491, 0.0, 0.0, -3.0718, 0.0, 2.3562, 1.57])
FULL_MAX = jnp.array([2.7925, 1.48353, 0.0, -0.0698, 0.0, 3.7525, -1.57])

# Sliced min/max for the active subset:
MIN_ACTIVE = FULL_MIN[ACTIVE_JOINTS]  # shape (5,)
MAX_ACTIVE = FULL_MAX[ACTIVE_JOINTS]  # shape (5,)

# We'll fix those locked joints at 0.0 (or any angles you want)
FIXED_VALUES = jnp.array([0.0, 0.0])  # shape (2,) for joints #2 and #4


################################################################################
# Oracle State Init
################################################################################


def batched_initial_state(num_envs):
    state = OracleState(
        gripper_state=jnp.array(-1.0),
        above_once=jnp.array(False),
        orientation_alligned=jnp.array(False),
        target_orientation_alligned=jnp.array(False),
        grasping=jnp.array(False),
        step=jnp.array(0),
        done=jnp.array(False),
        final_pos=jnp.array([0.5, 0.5, 0.5]),
    )
    return jax.tree_map(lambda x: jnp.broadcast_to(x, (num_envs,) + x.shape), state)


################################################################################
# Info Extraction (example)
################################################################################


def extract_info(env_state, env):
    """
    Extract required info from the batched env_state.
    This example is simplistic — adapt as needed for your real environment.
    """

    return {
        "proprio/effector_pos": env_state.pipeline_state.x.pos[..., 8, :],
        "proprio/effector_rot": env_state.pipeline_state.x.rot[..., 8, :],
        # 'privileged/target_block': jnp.zeros((env_state.obs.shape[0],), dtype=jnp.int32),
        "privileged/block_0_pos": env_state.pipeline_state.x.pos[..., 0, :],
        "privileged/block_0_rot": env_state.pipeline_state.x.rot[..., 0, :],
        "privileged/target_block_pos": env_state.pipeline_state.x.pos[..., 1, :],
        "proprio/gripper_contact": env_state.obs[..., env.gripper_distance_index].squeeze(),  # TODO: this causes trouble with 1 env.
        "proprio/gripper_force": env_state.obs[..., env.gripper_force_index].squeeze(),
    }


################################################################################
# Forward Kinematics with Active + Fixed Joints
################################################################################
@jax.jit
def forward_kinematics_active(
    env_sys,
    q,
    qd,
    active_angles,  # shape (num_active,)
    fixed_values,  # shape (num_fixed,)
    active_joints,
    fixed_joints,
    ee_idx,
):
    """
    Merge the active joint angles + fixed joint angles into a full 7-D angle array,
    then do Brax forward kinematics to get (pos, rot) of eef.
    """
    full_angles = jnp.zeros(7)
    full_angles = full_angles.at[active_joints].set(active_angles)
    full_angles = full_angles.at[fixed_joints].set(fixed_values)

    q = q.at[..., 14:21].set(full_angles)
    x_new, _ = kinematics.forward(env_sys, q, qd)
    return x_new.pos[..., ee_idx, :], x_new.rot[..., ee_idx, :]


################################################################################
# Re-Parameterization for Active Joints
################################################################################
def param_map(u, min_val, max_val):
    """
    Maps unconstrained u to [min_val, max_val] by a smooth sigmoid transform,
    dimensionwise.  shape (num_active,) -> shape (num_active,).
    """
    sig = 1.0 / (1.0 + jnp.exp(-u))
    return min_val + (max_val - min_val) * sig


def inverse_param_map(theta, min_val, max_val):
    """
    Inverse of param_map: from [min_val, max_val] -> R^dim.
    shape (num_active,) -> shape (num_active,).
    """
    scaled = (theta - min_val) / (max_val - min_val)
    scaled = jnp.clip(scaled, 1e-7, 1.0 - 1e-7)
    return jnp.log(scaled / (1 - scaled))


################################################################################
# GaussNewton Residual for Reparam + Active Joints
################################################################################
def make_residual_fn(env_sys):
    @jax.jit
    def residual_fn_reparam(
        u,
        q_init,
        qd_init,
        target_pos,
        target_rot,
        min_val,
        max_val,
        fixed_values,
        active_joints,
        fixed_joints,
        ee_idx,
    ):
        # now env_sys is just used inside as a constant
        active_angles = param_map(u, min_val, max_val)
        ee_pos, ee_rot = forward_kinematics_active(
            env_sys,
            q_init,
            qd_init,
            active_angles,
            fixed_values,
            active_joints,
            fixed_joints,
            ee_idx,
        )
        pos_residual = ee_pos - target_pos
        rot_residual = ee_rot - target_rot
        return jnp.concatenate([pos_residual, rot_residual], axis=-1)

    return residual_fn_reparam


################################################################################
# IK Solver: Only Solve for Active Joints
################################################################################
@partial(jax.jit, static_argnums=0)
def inverse_kinematics(
    solver,
    env_sys,
    q_init,
    qd_init,
    target_pos,
    target_rot,
    ee_idx,
    min_active,
    max_active,
    fixed_values,
    active_joints,
    fixed_joints,
):
    """
    1) Gather the environment's *full* 7 angles from q_init[...,14:21]
    2) Slice out the active angles
    3) Reparam -> unconstrained => GaussNewton => reparam => angles
    4) Merge w/ fixed joints, return final 7-D angles
    """
    full_angle_guess = q_init[..., 14:21]  # shape (num_envs, 7)
    angle_guess = full_angle_guess[..., active_joints]  # shape (num_envs, #active)
    qd_init_zeros = jnp.zeros_like(qd_init)

    def solve_env(angle_guess_i, q_i, qd_i, tpos_i, trot_i):
        # A) Convert guess -> unconstrained u
        u_init = inverse_param_map(angle_guess_i, min_active, max_active)

        # B) Solve GaussNewton in active dims
        res = solver.run(
            init_params=u_init,
            q_init=q_i,
            qd_init=qd_i,
            target_pos=tpos_i,
            target_rot=trot_i,
            min_val=min_active,
            max_val=max_active,
            fixed_values=fixed_values,
            active_joints=active_joints,
            fixed_joints=fixed_joints,
            ee_idx=ee_idx,
        )
        u_sol = res.params

        # C) final angles
        angle_sol = param_map(u_sol, min_active, max_active)
        # fallback if nan
        is_nan = jnp.any(jnp.isnan(angle_sol))
        safe_sol = jax.lax.cond(is_nan, lambda: angle_guess_i, lambda: angle_sol)
        return safe_sol

    angle_sol_batch = jax.vmap(solve_env)(angle_guess, q_init, qd_init_zeros, target_pos, target_rot)

    # Build back the full 7 angles if needed
    def fill_full(active_sol):
        full = jnp.zeros(7)
        full = full.at[active_joints].set(active_sol)
        full = full.at[fixed_joints].set(fixed_values)
        return full

    full_sol = jax.vmap(fill_full)(angle_sol_batch)
    # jax.debug.print("full_sol {}", full_sol)
    return full_sol


@jax.jit
def scale_to_actuator_range(full_angles):
    offset = (FULL_MIN + FULL_MAX) / 2.0
    multiplier = (FULL_MAX - FULL_MIN) / 2.0

    scaled = (full_angles - offset) / jnp.where(multiplier == 0.0, 1.0, multiplier)
    scaled = scaled.at[FIXED_JOINTS].set(0.0)  # explicitly zero them out
    return scaled


################################################################################
# The main rollout logic with per-env termination + full pipeline viz
################################################################################


def main(_):
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.save_path), exist_ok=True)

    num_envs = FLAGS.num_envs

    environment = create_brax_env(env_name=FLAGS.env_name)
    wrap_for_training = envs.training.wrap
    env = environment
    env = TrajectoryIdWrapper(env)
    env = wrap_for_training(env, episode_length=FLAGS.max_episode_steps, action_repeat=1)

    # reset_fn = jax.jit(env.reset)(env_keys)
    key = jax.random.PRNGKey(FLAGS.seed)
    dataset = []

    key, env_key = jax.random.split(key, 2)
    env_keys = jax.random.split(env_key, num_envs)
    env_state = env.reset(env_keys)

    # Load MJCF for Kinematics
    env_mjcf = mjcf.load(files(assets).joinpath("panda_binpick_easy.xml"))
    # solver = GaussNewton(residual_fun=residual_fn_reparam, maxiter=1)
    residual_fn = make_residual_fn(env_mjcf)
    solver = GaussNewton(residual_fun=residual_fn, maxiter=1)

    def rollout_episode(env_state, oracle_states, oracle_params, key, max_steps: int):
        """
        JAX-parallel rollout.
        1) per-environment termination is tracked in `terminated` (shape = (num_envs,)).
        2) we store pipeline_state each step in `frames` to enable full visualization.
        """

        def step_fn(carry, step_idx):
            (env_state, oracle_states, key, terminated, frames) = carry
            key, noise_key = jax.random.split(key)

            # Oracle: select action (pos, gripper). shape (num_envs, 5)
            infos = extract_info(env_state, env)

            def broadcast_params(params, batch_size):
                return jax.tree_util.tree_map(
                    lambda x: jnp.broadcast_to(jnp.asarray(x), (batch_size,) + jnp.shape(x)),
                    params,
                )

            batch_size = oracle_states.gripper_state.shape[0]
            oracle_params_batched = broadcast_params(oracle_params, batch_size)

            actions, smooth_orientations, new_oracle_states = jax.vmap(select_action)(infos, oracle_states, oracle_params_batched)

            # Add noise
            noise = jax.random.normal(noise_key, actions.shape) * FLAGS.noise
            actions = actions + noise

            # Inverse kinematics to get joint angles ( shape (num_envs, 7) )
            q_init = env_state.pipeline_state.q
            qd_init = env_state.pipeline_state.qd
            eef_idx = 8

            current_eef_pos = env_state.pipeline_state.x.pos[..., eef_idx, :]
            target_pos = actions[..., :3] + current_eef_pos
            finger = actions[..., 4]

            # IK
            # Solve for full 7 angles, ignoring fixed_joints
            arm_action = inverse_kinematics(
                solver,
                env_mjcf,
                q_init,
                qd_init,
                target_pos,
                smooth_orientations,
                eef_idx,
                MIN_ACTIVE,
                MAX_ACTIVE,
                FIXED_VALUES,
                ACTIVE_JOINTS,
                FIXED_JOINTS,
            )

            scaled_action = scale_to_actuator_range(arm_action)
            full_action = jnp.concatenate([scaled_action, finger[..., None]], axis=-1)
            full_action = jnp.clip(full_action, -0.999, 0.999)

            # Step environment, but freeze done envs
            next_env_state = env.step(env_state, full_action)

            # For per-environment termination
            terminate = next_env_state.done == 1
            terminated = jnp.logical_or(terminated, terminate)

            extra_fields = ("truncation", "traj_id")
            state_extras = {x: next_env_state.info[x] for x in extra_fields}

            # Log pipeline_state for full visualization
            frame = env_state.pipeline_state  # shape (num_envs, ...)
            frames = jax.tree_map(
                lambda arr_store, arr_val: arr_store.at[step_idx].set(arr_val),
                frames,
                frame,
            )

            # Construct transition info (still batched)
            transition = {
                "observation": env_state.obs,
                "next_observation": next_env_state.obs,  # ADDED next_obs
                "action": full_action,
                "reward": next_env_state.reward,
                "terminate": terminate,
                **state_extras,  # [truncation, traj_id]
            }

            return (
                next_env_state,
                new_oracle_states,
                key,
                terminated,
                frames,
            ), transition

        def init_frames(pipeline_state, max_steps):
            """Create a pytree of zero arrays with shape (max_steps, *leaf_shape)."""

            def make_array(leaf):
                # leaf has shape (num_envs, ...) at runtime
                return jnp.zeros((max_steps,) + leaf.shape, dtype=leaf.dtype)

            return jax.tree_map(make_array, pipeline_state)

        frames_init = init_frames(env_state.pipeline_state, max_steps)

        (
            final_env_state,
            final_oracle_states,
            _,
            final_terminated,
            frames,
        ), traj = jax.lax.scan(
            step_fn,
            (
                env_state,
                oracle_states,
                key,
                jnp.zeros((env_state.obs.shape[0],), dtype=bool),
                frames_init,
            ),
            jnp.arange(max_steps),
        )

        return traj, final_env_state, frames, final_terminated

    for ep in trange(FLAGS.num_episodes):
        # Reset
        key, subkey = jax.random.split(key)
        env_keys = jax.random.split(subkey, num_envs)
        env_state = env.reset(env_keys)
        oracle_states = batched_initial_state(num_envs)
        oracle_params = OracleParams()

        # Rollout
        key, rollout_key = jax.random.split(key)
        traj, final_env_state, frames, final_terminated = rollout_episode(
            env_state,
            oracle_states,
            oracle_params,
            rollout_key,
            FLAGS.max_episode_steps,
        )
        # Convert from jax to numpy
        traj_np = jax.tree_map(lambda x: np.array(x), traj)
        frames_np = jax.tree_map(lambda x: np.array(x), frames)
        # done_np = np.array(final_terminated)

        # Save transitions
        dataset.append(traj_np)  # shape = (max_steps,) with batch dimension inside

        # If we want to visualize environment 0's entire episode:
        if FLAGS.html_viz and ep % FLAGS.save_interval == 0:
            # frames_np is a pytree with shape [max_steps, num_envs, ...]
            def slice_env0(pstate_array):
                return pstate_array[:, 0, ...]

            frames_env0 = jax.tree_map(slice_env0, frames_np)

            # Build the list of pipeline states
            def gather_pipeline_state(frames_env0, t):
                return jax.tree_map(lambda x: x[t], frames_env0)

            frames_list = []
            for t in range(FLAGS.max_episode_steps):
                pstate_t = gather_pipeline_state(frames_env0, t)
                frames_list.append(pstate_t)

            url = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), frames_list, height=480)
            with open(os.path.join(EXP_DIR, f"rollout_{ep}_env0.html"), "w") as f:
                f.write(url)

        # save dataset
        if ep % FLAGS.save_interval == 0 or ep == FLAGS.num_episodes - 1:
            with open(FLAGS.save_path, "wb") as f:
                pickle.dump(dataset, f)
            load_and_flatten_dataset(FLAGS.save_path)

            print(f"[Saved {len(dataset)} rollouts to {FLAGS.save_path}]")


def load_and_flatten_dataset(pickle_path):
    """Flatten the saved rollouts into per-transition arrays and write them to <pickle_path without .pkl>.npz."""
    with open(pickle_path, "rb") as f:
        dataset = pickle.load(f)  # list of episode dicts

    transitions = []
    traj_counter = 1  # start counting traj_ids from 1

    for ep_idx, ep_traj_dict in enumerate(dataset):
        obs_array = ep_traj_dict["observation"]  # shape (max_steps, num_envs, …)
        next_obs_array = ep_traj_dict["next_observation"]
        act_array = ep_traj_dict["action"]
        rew_array = ep_traj_dict["reward"]
        terminate_array = ep_traj_dict["terminate"]
        truncation_array = ep_traj_dict["truncation"]

        max_steps = obs_array.shape[0]
        num_envs = obs_array.shape[1]

        for e in range(num_envs):
            for t in range(max_steps):
                o = obs_array[t, e]
                no = next_obs_array[t, e]
                a = act_array[t, e]
                r = rew_array[t, e]
                terminate = terminate_array[t, e]
                truncation = truncation_array[t, e]

                single_step_dict = {
                    "observation": o,
                    "next_observation": no,
                    "action": a,
                    "reward": np.array([r], dtype=np.float32),
                    "terminate": np.array([terminate], dtype=np.float32),
                    "truncation": np.array([truncation], dtype=np.float32),
                    "traj_id": np.array([traj_counter], dtype=np.int64),
                }

                transitions.append(single_step_dict)
                if terminate:
                    break

            traj_counter += 1  # Increment once per env per episode

    # One stacked array per key; loads much faster and with far less memory than a pickled list of dicts.
    npz_path = os.path.splitext(pickle_path)[0] + ".npz"
    np.savez(npz_path, **{key: np.stack([t[key] for t in transitions]) for key in transitions[0]})
    print(f"[Saved {len(transitions)} transitions to {npz_path}]")


if __name__ == "__main__":
    app.run(main)
