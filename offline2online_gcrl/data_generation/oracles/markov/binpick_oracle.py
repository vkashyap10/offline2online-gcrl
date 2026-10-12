from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import lax, tree_util

###############################################################################
# Quaternion Utilities
###############################################################################


@jax.jit
def quaternion_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return jnp.array([w, x, y, z])


@jax.jit
def quaternion_inverse(q: jnp.ndarray) -> jnp.ndarray:
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


@jax.jit
def quaternion_angle_difference(q1: jnp.ndarray, q2: jnp.ndarray) -> float:
    dot = jnp.clip(jnp.dot(q1 / jnp.linalg.norm(q1), q2 / jnp.linalg.norm(q2)), -1.0, 1.0)
    angle = 2.0 * jnp.arccos(jnp.abs(dot))
    return angle


@jax.jit
def smooth_orientation_update(current: jnp.ndarray, desired: jnp.ndarray, gain=0.1) -> jnp.ndarray:
    cur = current / jnp.linalg.norm(current)
    des = desired / jnp.linalg.norm(desired)
    q_diff = quaternion_multiply(des, quaternion_inverse(cur))
    angle_diff = 2.0 * jnp.arccos(jnp.clip(q_diff[0], -1.0, 1.0))
    step_angle = gain * angle_diff

    sin_half = jnp.sin(step_angle / 2.0)
    axis = q_diff[1:]
    axis_norm = jnp.linalg.norm(axis)
    axis = jnp.where(axis_norm > 1e-8, axis / axis_norm, jnp.array([0.0, 0.0, 1.0]))
    incremental_rotation = jnp.concatenate([jnp.array([jnp.cos(step_angle / 2.0)]), axis * sin_half])
    new_orientation = quaternion_multiply(incremental_rotation, cur)
    return new_orientation / jnp.linalg.norm(new_orientation)


def update_orientation_state(current: jnp.ndarray, target_quat: jnp.ndarray, gain: float):
    # desired = target_quat / jnp.linalg.norm(target_quat)
    desired = compute_eef_orientation_from_target(target_quat)

    angle = quaternion_angle_difference(current, desired)
    smooth = smooth_orientation_update(current, desired, gain=gain)
    return desired, angle, smooth


@jax.jit
def adaptive_gain(diff: jnp.ndarray, min_gain=0.1, max_gain=0.3, scale=4.0) -> float:
    norm = jnp.linalg.norm(diff)
    return min_gain + (max_gain - min_gain) * (1 - jnp.exp(-scale * norm))


# @jax.jit
def _euler_to_quaternion(axis: str, deg: float) -> jnp.ndarray:
    """
    Converts a single-axis Euler rotation in degrees into a quaternion [w, x, y, z].
    We assume the rotation is about one of the principal axes x, y, or z.
    """
    angle = jnp.deg2rad(deg)
    half = angle * 0.5
    c = jnp.cos(half)
    s = jnp.sin(half)
    if axis == "x":
        return jnp.array([c, s, 0.0, 0.0])
    elif axis == "y":
        return jnp.array([c, 0.0, s, 0.0])
    elif axis == "z":
        return jnp.array([c, 0.0, 0.0, s])
    else:
        raise ValueError(f"Unsupported axis '{axis}'. Choose from ['x','y','z'].")


@jax.jit
def compute_eef_orientation_from_target(target_quat: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the desired end-effector orientation that aligns local EEF axes
    with the given target orientation in world frame, followed by:
      - A fixed 185° rotation about X.
      - An additional -45° rotation about Z.

    Inputs:
      target_quat: jnp.ndarray [w, x, y, z], the target's quaternion.
    Output:
      q_ee: jnp.ndarray [w, x, y, z], the desired EEF quaternion in the same format.
    """
    target_normed = target_quat / jnp.linalg.norm(target_quat)

    # Define the same fixed rotations you had in SciPy:
    q_align_x185 = _euler_to_quaternion("x", 185.0)  # ~185 deg about X
    q_align_zm45 = _euler_to_quaternion("z", -45.0)  # -45 deg about Z

    # Multiply them in wxyz format:
    #   q_ee = target * q_align_x185 * q_align_zm45
    # The function quaternion_multiply is your standard Hamilton product in wxyz ordering.
    tmp = quaternion_multiply(target_normed, q_align_x185)
    q_ee = quaternion_multiply(tmp, q_align_zm45)

    # Return normalized just in case
    return q_ee / jnp.linalg.norm(q_ee)


###############################################################################
# Oracle State and Params
###############################################################################


@tree_util.register_pytree_node_class
@dataclass
class OracleState:
    """
    A PyTree dataclass representing the Oracle's internal state.
    Once registered, JAX transformations (tree_map, vmap, etc.)
    will treat each field as a separate leaf.
    """

    gripper_state: jnp.ndarray
    above_once: jnp.ndarray
    orientation_alligned: jnp.ndarray
    target_orientation_alligned: jnp.ndarray
    grasping: jnp.ndarray
    step: jnp.ndarray
    done: jnp.ndarray
    final_pos: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.gripper_state,
            self.above_once,
            self.orientation_alligned,
            self.target_orientation_alligned,
            self.grasping,
            self.step,
            self.done,
            self.final_pos,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass
class OracleParams:
    finger_length: float = 0.2
    cube_half_height: float = 0.03
    gain_pos: float = 0.1
    gain_yaw: float = 0.3
    max_step: int = 200
    block_above_offset: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.33]))
    above_threshold: float = 0.33

    def tree_flatten(self):
        children = (
            self.finger_length,
            self.cube_half_height,
            self.gain_pos,
            self.gain_yaw,
            self.max_step,
            self.block_above_offset,
            self.above_threshold,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


###############################################################################
# Phase Functions
###############################################################################
#
# Each phase function has the same signature:
# (eff_pos, eff_rot, block_pos, block_quat, target_pos, state, params)
# and it returns (action, new_state, smooth_orientation).
#
# No more pack/unpack of state is needed.
###############################################################################


def phase1_move_above(
    eff_pos: jnp.ndarray,
    eff_rot: jnp.ndarray,
    block_pos: jnp.ndarray,
    block_quat: jnp.ndarray,
    target_pos: jnp.ndarray,  # unused in this phase
    state: OracleState,
    params: OracleParams,
) -> Tuple[jnp.ndarray, OracleState, jnp.ndarray]:
    desired_pos = block_pos + params.block_above_offset
    pos_diff = desired_pos - eff_pos
    gain = adaptive_gain(pos_diff)
    action_xyz = pos_diff * gain

    _, angle_diff, smooth_orientation = update_orientation_state(eff_rot, block_quat, gain=params.gain_yaw)
    orientation_alligned = angle_diff < jnp.deg2rad(5.0)

    new_gripper = state.gripper_state + 0.1 * (-1.0 - state.gripper_state)

    new_state = OracleState(
        gripper_state=new_gripper,
        above_once=state.above_once,
        orientation_alligned=orientation_alligned,
        target_orientation_alligned=state.target_orientation_alligned,
        grasping=state.grasping,
        step=state.step + 1,
        done=False,
        final_pos=state.final_pos,
    )
    action = jnp.concatenate([action_xyz, jnp.array([new_gripper])])
    return action, new_state, smooth_orientation


def phase2_move_to_block(
    eff_pos: jnp.ndarray,
    eff_rot: jnp.ndarray,
    block_pos: jnp.ndarray,
    block_quat: jnp.ndarray,
    target_pos: jnp.ndarray,  # unused in this phase
    state: OracleState,
    params: OracleParams,
) -> Tuple[jnp.ndarray, OracleState, jnp.ndarray]:
    desired_pos = block_pos + jnp.array([0.0, 0.0, params.finger_length + params.cube_half_height])
    pos_diff = desired_pos - eff_pos
    gain = adaptive_gain(pos_diff)
    action_xyz = pos_diff * gain

    _, _, smooth_orientation = update_orientation_state(eff_rot, block_quat, gain=params.gain_yaw)
    new_gripper = state.gripper_state + 0.1 * (-1.0 - state.gripper_state)

    new_state = OracleState(
        gripper_state=new_gripper,
        above_once=state.above_once,
        orientation_alligned=state.orientation_alligned,
        target_orientation_alligned=state.target_orientation_alligned,
        grasping=state.grasping,
        step=state.step + 1,
        done=False,
        final_pos=state.final_pos,
    )
    action = jnp.concatenate([action_xyz, jnp.array([new_gripper])])
    return action, new_state, smooth_orientation


def phase3_grasp(
    eff_pos: jnp.ndarray,
    eff_rot: jnp.ndarray,
    block_pos: jnp.ndarray,
    block_quat: jnp.ndarray,
    target_pos: jnp.ndarray,  # unused in this phase
    state: OracleState,
    params: OracleParams,
) -> Tuple[jnp.ndarray, OracleState, jnp.ndarray]:
    desired_pos = block_pos + jnp.array([0.0, 0.0, params.finger_length + params.cube_half_height])
    pos_diff = desired_pos - eff_pos
    gain = adaptive_gain(pos_diff)
    action_xyz = pos_diff * gain

    _, _, smooth_orientation = update_orientation_state(eff_rot, block_quat, gain=params.gain_yaw)
    new_gripper = state.gripper_state + 0.1 * (1.0 - state.gripper_state)

    # Simple grasp heuristic:
    grasping = jnp.linalg.norm(block_pos[:2] - eff_pos[:2]) <= 0.04

    new_state = OracleState(
        gripper_state=new_gripper,
        above_once=state.above_once,
        orientation_alligned=state.orientation_alligned,
        target_orientation_alligned=state.target_orientation_alligned,
        grasping=grasping,
        step=state.step + 1,
        done=False,
        final_pos=state.final_pos,
    )
    action = jnp.concatenate([action_xyz, jnp.array([new_gripper])])
    return action, new_state, smooth_orientation


def phase4_lift_block(
    eff_pos: jnp.ndarray,
    eff_rot: jnp.ndarray,
    block_pos: jnp.ndarray,
    block_quat: jnp.ndarray,  # unused in this phase
    target_pos: jnp.ndarray,  # unused in this phase
    state: OracleState,
    params: OracleParams,
) -> Tuple[jnp.ndarray, OracleState, jnp.ndarray]:
    desired_pos = block_pos.at[2].set(params.block_above_offset[2] * 1.2)
    pos_diff = desired_pos - eff_pos
    gain = adaptive_gain(pos_diff)
    action_xyz = pos_diff * gain

    # Lift with a neutral orientation (example: [1, 0, 0, 0])
    _, _, smooth_orientation = update_orientation_state(eff_rot, jnp.array([1.0, 0.0, 0.0, 0.0]), gain=params.gain_yaw)
    new_gripper = state.gripper_state + 0.1 * (1.0 - state.gripper_state)
    above_once = eff_pos[2] > params.above_threshold

    new_state = OracleState(
        gripper_state=new_gripper,
        above_once=above_once,
        orientation_alligned=state.orientation_alligned,
        target_orientation_alligned=state.target_orientation_alligned,
        grasping=state.grasping,
        step=state.step + 1,
        done=False,
        final_pos=state.final_pos,
    )
    action = jnp.concatenate([action_xyz, jnp.array([new_gripper])])
    return action, new_state, smooth_orientation


def phase5_move_above_target(
    eff_pos: jnp.ndarray,
    eff_rot: jnp.ndarray,
    block_pos: jnp.ndarray,  # unused in this phase
    block_quat: jnp.ndarray,  # unused in this phase
    target_pos: jnp.ndarray,
    state: OracleState,
    params: OracleParams,
) -> Tuple[jnp.ndarray, OracleState, jnp.ndarray]:
    desired_pos = target_pos + params.block_above_offset
    pos_diff = desired_pos - eff_pos
    gain = adaptive_gain(pos_diff)
    action_xyz = pos_diff * gain

    _, angle_diff, smooth_orientation = update_orientation_state(eff_rot, jnp.array([1.0, 0.0, 0.0, 0.0]), gain=params.gain_yaw)
    target_orientation_alligned = angle_diff < jnp.deg2rad(5.0)
    new_gripper = state.gripper_state + 0.1 * (1.0 - state.gripper_state)

    new_state = OracleState(
        gripper_state=new_gripper,
        above_once=state.above_once,
        orientation_alligned=state.orientation_alligned,
        target_orientation_alligned=target_orientation_alligned,
        grasping=state.grasping,
        step=state.step + 1,
        done=False,
        final_pos=state.final_pos,
    )
    action = jnp.concatenate([action_xyz, jnp.array([new_gripper])])
    return action, new_state, smooth_orientation


...


def phase6_move_to_target(
    eff_pos: jnp.ndarray,
    eff_rot: jnp.ndarray,
    block_pos: jnp.ndarray,
    block_quat: jnp.ndarray,
    target_pos: jnp.ndarray,
    state: OracleState,
    params: OracleParams,
) -> Tuple[jnp.ndarray, OracleState, jnp.ndarray]:
    pos_diff = target_pos - eff_pos
    gain = adaptive_gain(pos_diff)
    action_xyz = pos_diff * gain

    _, _, smooth_orientation = update_orientation_state(eff_rot, jnp.array([1.0, 0.0, 0.0, 0.0]), gain=params.gain_yaw)
    new_gripper = state.gripper_state + 0.1 * (1.0 - state.gripper_state)

    new_state = OracleState(
        gripper_state=new_gripper,
        above_once=state.above_once,
        orientation_alligned=state.orientation_alligned,
        target_orientation_alligned=state.target_orientation_alligned,
        grasping=state.grasping,
        step=state.step + 1,
        done=False,
        final_pos=state.final_pos,
    )
    action = jnp.concatenate([action_xyz, jnp.array([new_gripper])])
    return action, new_state, smooth_orientation


def phase7_release(
    eff_pos: jnp.ndarray,
    eff_rot: jnp.ndarray,
    block_pos: jnp.ndarray,
    block_quat: jnp.ndarray,
    target_pos: jnp.ndarray,
    state: OracleState,
    params: OracleParams,
) -> Tuple[jnp.ndarray, OracleState, jnp.ndarray]:
    pos_diff = target_pos - eff_pos
    gain = adaptive_gain(pos_diff)
    action_xyz = pos_diff * gain

    _, _, smooth_orientation = update_orientation_state(eff_rot, jnp.array([1.0, 0.0, 0.0, 0.0]), gain=params.gain_yaw)
    new_gripper = -1.0

    new_state = OracleState(
        gripper_state=new_gripper,
        above_once=state.above_once,
        orientation_alligned=state.orientation_alligned,
        target_orientation_alligned=state.target_orientation_alligned,
        grasping=state.grasping,
        step=state.step + 1,
        done=False,
        final_pos=state.final_pos,
    )
    action = jnp.concatenate([action_xyz, jnp.array([new_gripper])])
    return action, new_state, smooth_orientation


def phase8_move_air(
    eff_pos: jnp.ndarray,
    eff_rot: jnp.ndarray,
    block_pos: jnp.ndarray,
    block_quat: jnp.ndarray,
    target_pos: jnp.ndarray,
    state: OracleState,
    params: OracleParams,
) -> Tuple[jnp.ndarray, OracleState, jnp.ndarray]:
    desired_pos = jnp.array([block_pos[0], block_pos[1], params.above_threshold * 2])
    pos_diff = desired_pos - eff_pos
    gain = adaptive_gain(pos_diff)
    action_xyz = pos_diff * gain

    _, _, smooth_orientation = update_orientation_state(eff_rot, jnp.array([1.0, 0.0, 0.0, 0.0]), gain=params.gain_yaw)
    new_gripper = -1.0

    new_state = OracleState(
        gripper_state=new_gripper,
        above_once=state.above_once,
        orientation_alligned=state.orientation_alligned,
        target_orientation_alligned=state.target_orientation_alligned,
        grasping=state.grasping,
        step=state.step + 1,
        done=False,
        final_pos=state.final_pos,
    )
    action = jnp.concatenate([action_xyz, jnp.array([new_gripper])])
    return action, new_state, smooth_orientation


def phase9_move_final(
    eff_pos: jnp.ndarray,
    eff_rot: jnp.ndarray,
    block_pos: jnp.ndarray,
    block_quat: jnp.ndarray,
    target_pos: jnp.ndarray,
    state: OracleState,
    params: OracleParams,
) -> Tuple[jnp.ndarray, OracleState, jnp.ndarray]:
    pos_diff = state.final_pos - eff_pos
    gain = adaptive_gain(pos_diff)
    action_xyz = pos_diff * gain

    _, _, smooth_orientation = update_orientation_state(eff_rot, jnp.array([1.0, 0.0, 0.0, 0.0]), gain=params.gain_yaw)
    new_gripper = -1.0
    done = jnp.linalg.norm(pos_diff) < 0.02

    new_state = OracleState(
        gripper_state=new_gripper,
        above_once=state.above_once,
        orientation_alligned=state.orientation_alligned,
        target_orientation_alligned=state.target_orientation_alligned,
        grasping=state.grasping,
        step=state.step + 1,
        done=done,
        final_pos=state.final_pos,
    )
    action = jnp.concatenate([action_xyz, jnp.array([new_gripper])])
    return action, new_state, smooth_orientation


###############################################################################
# Phase Selector
###############################################################################


def select_action(info: Dict[str, Any], state: OracleState, params: OracleParams) -> Tuple[jnp.ndarray, jnp.ndarray, OracleState]:
    """
    Decide which phase to use (0..4), call it with the correct arguments,
    and return (action, smooth_orientation, new_state).
    """
    eff_pos = jnp.array(info["proprio/effector_pos"])
    eff_rot = jnp.array(info["proprio/effector_rot"])

    # target_block = info['privileged/target_block']
    block_pos = jnp.array(info["privileged/block_0_pos"])
    block_quat = jnp.array(info["privileged/block_0_rot"])
    target_pos = jnp.array(info["privileged/target_block_pos"])
    gripper_contact = jnp.array(info["proprio/gripper_contact"])

    gripper_closed = gripper_contact < 0.06
    xy_aligned = jnp.linalg.norm(block_pos[:2] - eff_pos[:2]) <= 0.02
    pos_aligned = jnp.linalg.norm(block_pos - eff_pos) <= 0.23
    target_xy_aligned = jnp.linalg.norm(target_pos[:2] - block_pos[:2]) <= 0.02
    target_pos_aligned = jnp.linalg.norm(target_pos - block_pos) <= 0.02
    gripper_open = gripper_contact > 0.09
    above = eff_pos[2] > params.above_threshold
    # is_final_position_reached = jnp.linalg.norm(state.final_pos - eff_pos) <= 0.02

    not_orientation_aligned = ~state.orientation_alligned
    not_target_orientation_aligned = ~state.target_orientation_alligned
    release_condition = ~gripper_open
    move_air_condition = gripper_open & (~above)

    # Phase logic
    phase_1 = (~xy_aligned & ~state.grasping) | not_orientation_aligned
    phase_1 = phase_1 & (~gripper_closed)
    phase_2 = ~pos_aligned & (~gripper_closed)
    phase_3 = pos_aligned & (~gripper_closed)
    phase_4 = gripper_closed & (~state.above_once) & (~target_xy_aligned)
    phase_5 = gripper_closed & ((~target_xy_aligned) | not_target_orientation_aligned)
    phase_7 = release_condition
    phase_8 = move_air_condition

    # jax.debug.print("Phase conditions:")
    # jax.debug.print("  phase_1: {}", phase_1)
    # jax.debug.print("  phase_2: {}", phase_2)
    # jax.debug.print("  phase_3: {}", phase_3)
    # jax.debug.print("  phase_4: {}", phase_4)
    # jax.debug.print("  phase_4 gripper_closed: {}", gripper_closed)
    # jax.debug.print("  phase_4 state.above_once: {}", state.above_once)
    # jax.debug.print("  phase_4 target_xy_aligned: {}", target_xy_aligned)
    # jax.debug.print("  phase_5: {}", phase_5)
    # jax.debug.print("  phase_6 (fallback): {}", ~(phase_1 | phase_2 | phase_3 | phase_4 | phase_5))
    # jax.debug.print("  phase_7 (release): {}", phase_7)
    # jax.debug.print("  phase_8 (move_air): {}", phase_8)
    # jax.debug.print("  phase_9 (fallback): {}", ~(phase_7 | phase_8))
    # jax.debug.print("  target_pos_aligned: {}", target_pos_aligned)

    args = (eff_pos, eff_rot, block_pos, block_quat, target_pos, state, params)

    def wrap_phase(f):
        return lambda _: f(*args)

    action, new_state, smooth_orientation = lax.cond(
        target_pos_aligned,
        lambda _: lax.cond(
            phase_7,
            wrap_phase(phase7_release),
            lambda _: lax.cond(
                phase_8,
                wrap_phase(phase8_move_air),
                lambda _: wrap_phase(phase9_move_final)(None),
                operand=None,
            ),
            operand=None,
        ),
        lambda _: lax.cond(
            phase_1,
            wrap_phase(phase1_move_above),
            lambda _: lax.cond(
                phase_2,
                wrap_phase(phase2_move_to_block),
                lambda _: lax.cond(
                    phase_3,
                    wrap_phase(phase3_grasp),
                    lambda _: lax.cond(
                        phase_4,
                        wrap_phase(phase4_lift_block),
                        lambda _: lax.cond(
                            phase_5,
                            wrap_phase(phase5_move_above_target),
                            lambda _: wrap_phase(phase6_move_to_target)(None),
                            operand=None,
                        ),
                        operand=None,
                    ),
                    operand=None,
                ),
                operand=None,
            ),
            operand=None,
        ),
        operand=None,
    )

    return action, smooth_orientation, new_state


###############################################################################
# Test
###############################################################################
if __name__ == "__main__":
    # A single-environment test
    state = OracleState(
        gripper_state=-1.0,
        above_once=False,
        orientation_alligned=False,
        target_orientation_alligned=False,
        grasping=False,
        step=0,
        done=False,
        final_pos=jnp.array([0.5, 0.5, 0.5]),
    )

    params = OracleParams()

    info = {
        "proprio/effector_pos": [0.0, 0.0, 0.1],
        "proprio/effector_rot": [1.0, 0.0, 0.0, 0.0],
        # "privileged/target_block": 0,
        "privileged/block_0_pos": [0.1, 0.1, 0.0],
        "privileged/block_0_rot": [1.0, 0.0, 0.0, 0.0],
        "privileged/target_block_pos": [0.3, 0.3, 0.0],
        "proprio/gripper_contact": 0.1,
    }

    action, smooth_orientation, new_state = select_action({}, info, state, params)

    print("Action:", action)
    print("Smoothed Orientation:", smooth_orientation)
    print("New State:", new_state)
