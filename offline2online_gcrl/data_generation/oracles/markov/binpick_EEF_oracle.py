import numpy as np


class MarkovOracle:
    """Markovian oracle for manipulation tasks (from OGBench)."""

    def __init__(self, env, min_norm=0.4):
        """Initialize the oracle.

        Args:
            env: Environment.
            min_norm: Minimum norm for the relative position. Setting it to a non-zero value can help the agent to learn
                more robust policies.
        """
        self._env = env
        self._min_norm = min_norm
        self._debug = False  # Set to True to print debug information.
        self._done = False

        if self._debug:
            np.set_printoptions(suppress=True)

    def shape_diff(self, diff):
        """Shape the difference vector to have a minimum norm."""
        diff_norm = np.linalg.norm(diff)
        if diff_norm >= self._min_norm:
            return diff
        else:
            return diff / (diff_norm + 1e-6) * self._min_norm

    def shortest_yaw(self, eff_yaw, obj_yaw, n=4):
        """Find the symmetry-aware shortest yaw angle to the object."""
        symmetries = np.array([i * 2 * np.pi / n + obj_yaw for i in range(-n, n + 1)])
        d = np.argmin(np.abs(eff_yaw - symmetries))
        return symmetries[d]

    def print_phase(self, phase):
        """Print the current phase."""
        if self._debug:
            print(f"Phase {phase:50}", end=" ")

    @property
    def done(self):
        return self._done

    def reset(self, ob, info):
        pass

    def select_action(self, ob, info):
        pass


class BinpickEEFMarkovOracle(MarkovOracle):
    """Scripted pick-and-place controller for arm_binpick_easy_EEF.

    Moves above the cube, grasps it, carries it above the target, places it and releases, then moves the gripper to a
    random final position. Actions are [dx, dy, dz, yaw, gripper]; the environment ignores yaw.
    """

    def __init__(self, max_step=201, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_step = max_step
        self.finger_length = 0.08
        self.cube_half_height = 0.03
        self.gain_pos = 1
        self.gain_yaw = 3

    def reset(self, ob, info):
        self._done = False
        self._step = 0
        self._final_pos = np.random.uniform(*self._env.unwrapped._arm_sampling_bounds)
        self._final_yaw = np.random.uniform(-np.pi, np.pi)

    def select_action(self, ob, info):
        effector_pos = info["proprio/effector_pos"]
        effector_yaw = info["proprio/effector_yaw"][0]

        target_block = info["privileged/target_block"]
        block_pos = info[f"privileged/block_{target_block}_pos"]
        block_yaw = self.shortest_yaw(effector_yaw, info[f"privileged/block_{target_block}_yaw"][0])
        target_pos = info["privileged/target_block_pos"]
        target_yaw = self.shortest_yaw(effector_yaw, info["privileged/target_block_yaw"][0])

        block_above_offset = np.array([0, 0, self.finger_length + self.cube_half_height + 0.1])
        above_threshold = 0.16
        gripper_closed = info["proprio/gripper_contact"] < 0.06
        gripper_open = info["proprio/gripper_contact"] > 0.09
        above = effector_pos[2] > above_threshold
        xy_aligned = np.linalg.norm(block_pos[:2] - effector_pos[:2]) <= 0.04
        pos_aligned = np.linalg.norm(block_pos - effector_pos) <= 0.12
        target_xy_aligned = np.linalg.norm(target_pos[:2] - block_pos[:2]) <= 0.02
        target_pos_aligned = np.linalg.norm(target_pos - block_pos) <= 0.02
        final_pos_aligned = np.linalg.norm(self._final_pos - effector_pos) <= 0.04

        action = np.zeros(5)
        if not target_pos_aligned:
            if not xy_aligned:
                self.print_phase("1: Move above the block")
                diff = self.shape_diff(block_pos + block_above_offset - effector_pos)
                action[:3] = diff[:3] * self.gain_pos
                action[3] = (block_yaw - effector_yaw) * self.gain_yaw
                action[4] = -1
            elif not pos_aligned:
                self.print_phase("2: Move to the block")
                diff = self.shape_diff(block_pos - effector_pos)
                action[:3] = diff[:3] * self.gain_pos
                action[3] = (block_yaw - effector_yaw) * self.gain_yaw
                action[4] = -1
            elif pos_aligned and not gripper_closed:
                self.print_phase("3: Grasp")
                diff = self.shape_diff(block_pos + np.array([0, 0, self.finger_length + self.cube_half_height]) - effector_pos)
                action[:3] = diff[:3] * self.gain_pos
                action[3] = (block_yaw - effector_yaw) * self.gain_yaw
                action[4] = 1
            elif pos_aligned and gripper_closed and not above and not target_xy_aligned:
                self.print_phase("4: Move in the air")
                diff = self.shape_diff(np.array([block_pos[0], block_pos[1], block_above_offset[2] * 2]) - effector_pos)
                action[:3] = diff[:3] * self.gain_pos
                action[3] = (target_yaw - block_yaw) * self.gain_yaw
                action[4] = 1
            elif pos_aligned and gripper_closed and above and not target_xy_aligned:
                self.print_phase("5: Move above the target")
                diff = self.shape_diff(target_pos + block_above_offset - effector_pos)
                action[:3] = diff[:3] * self.gain_pos
                action[3] = (target_yaw - block_yaw) * self.gain_yaw
                action[4] = 1
            else:
                self.print_phase("6: Move to the target")
                diff = self.shape_diff(target_pos - effector_pos)
                action[:3] = diff[:3] * self.gain_pos
                action[3] = (target_yaw - block_yaw) * self.gain_yaw
                action[4] = 1
        else:
            if not gripper_open:
                self.print_phase("7: Release")
                diff = self.shape_diff(target_pos - effector_pos)
                action[:3] = diff[:3] * self.gain_pos
                action[3] = (target_yaw - block_yaw) * self.gain_yaw
                action[4] = -1
            elif gripper_open and not above:
                self.print_phase("8: Move in the air")
                diff = self.shape_diff(np.array([block_pos[0], block_pos[1], above_threshold * 2]) - effector_pos)
                action[:3] = diff[:3] * self.gain_pos
                action[3] = (self._final_yaw - effector_yaw) * self.gain_yaw
                action[4] = -1
            else:
                self.print_phase("9: Move to the final position")
                diff = self.shape_diff(self._final_pos - effector_pos)
                action[:3] = diff[:3] * self.gain_pos
                action[3] = (self._final_yaw - effector_yaw) * self.gain_yaw
                action[4] = -1

            if final_pos_aligned:
                self._done = True

        action = np.clip(action, -1, 1)
        if self._debug:
            print(action)

        self._step += 1
        if self._step == self._max_step:
            self._done = True

        return action
