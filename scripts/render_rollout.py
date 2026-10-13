"""Render a successful rollout of a trained policy to a GIF.

Runs the policy (deterministic mean action) for up to --max_attempts episodes and saves the first one that reaches the
goal. Uses MuJoCo's offscreen renderer, so run with MUJOCO_GL=egl on a headless machine.

Usage:
    MUJOCO_GL=egl PYTHONPATH=offline2online_gcrl/impls:. python scripts/render_rollout.py \
        --train_config_path=configs/train_config.yaml --agent=offline2online_gcrl/impls/agents/gcivl_jax.py \
        --checkpoint=path/to/final.pkl --out=docs/figures/rollout.gif
"""

import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from absl import app, flags
from brax.io.model import load_params
from ml_collections import config_flags
from mujoco import mjx

from offline2online_gcrl.impls.utils.agent_setup import create_agent
from offline2online_gcrl.impls.utils.buffer_setup import create_dummy_transition
from offline2online_gcrl.impls.utils.config import load_config
from offline2online_gcrl.impls.utils.env_utils import create_wrapped_env

FLAGS = flags.FLAGS
flags.DEFINE_string("train_config_path", "./configs/train_config.yaml", "Training config used for the checkpoint.")
flags.DEFINE_string("checkpoint", None, "Path to a checkpoint saved during training (e.g. final.pkl).", required=True)
flags.DEFINE_string("out", "docs/figures/rollout.gif", "Output GIF path.")
flags.DEFINE_integer("seed", 0, "Random seed for the episode's start and goal.")
flags.DEFINE_integer("max_attempts", 20, "Episodes to try before giving up on finding a successful one.")
flags.DEFINE_integer("hold_frames", 15, "Extra frames to repeat the final state, so the result is visible.")
flags.DEFINE_integer("frame_skip", 1, "Render every n-th step.")
flags.DEFINE_integer("width", 480, "Frame width.")
flags.DEFINE_integer("height", 360, "Frame height.")
config_flags.DEFINE_config_file("agent", "offline2online_gcrl/impls/agents/gcivl_jax.py", lock_config=False)


def main(_):
    train_config = load_config(FLAGS.train_config_path)
    wrapped_env, env = create_wrapped_env(train_config.env_name, train_config.episode_length, train_config.action_repeat)
    agent = create_agent(
        config=FLAGS.agent,
        rng=jax.random.PRNGKey(0),
        env=wrapped_env,
        dummy_transition=create_dummy_transition(wrapped_env),
        goal_conditioning=train_config.goal_conditioning,
    )
    params = load_params(FLAGS.checkpoint)

    @jax.jit
    def act(obs):
        dist = agent.network.select("actor")(obs[None, : env.state_dim], obs[None, env.state_dim :], params=params, temperature=0.0)
        return jnp.clip(dist.mode()[0], -1, 1)

    reset, step = jax.jit(env.reset), jax.jit(env.step)
    key = jax.random.PRNGKey(FLAGS.seed)
    for attempt in range(FLAGS.max_attempts):
        key, reset_key = jax.random.split(key)
        state = reset(reset_key)
        states = [state.pipeline_state]
        for _ in range(train_config.episode_length):
            state = step(state, act(state.obs))
            states.append(state.pipeline_state)
            if state.done:
                break
        if float(state.reward) > 0:
            print(f"Success on attempt {attempt + 1} after {len(states) - 1} steps")
            break
    else:
        print(f"No successful episode in {FLAGS.max_attempts} attempts; saving the last one")

    model = env.sys.mj_model
    model.vis.headlight.ambient[:] = 0.5  # the scene only has one top light
    model.vis.headlight.diffuse[:] = 0.5
    renderer = mujoco.Renderer(model, height=FLAGS.height, width=FLAGS.width)
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 1  # the table (group 4) and fingers (group 3) are hidden by default
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance, camera.azimuth, camera.elevation = 0.75, 90.0, -40.0
    camera.lookat[:] = [0.0, 0.6, 0.05]  # midway between the cube's start and the goal
    frames = []
    for pipeline_state in states[:: FLAGS.frame_skip]:
        data = mjx.get_data(model, pipeline_state)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        frames.append(renderer.render())
    frames += [frames[-1]] * FLAGS.hold_frames
    imageio.mimsave(FLAGS.out, [np.asarray(f) for f in frames], duration=80, loop=0)  # ms per frame
    print(f"Wrote {len(frames)} frames to {FLAGS.out}")


if __name__ == "__main__":
    app.run(main)
