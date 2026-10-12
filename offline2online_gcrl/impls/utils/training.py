import time

import jax
import jax.numpy as jnp
from absl import logging
from brax.io.model import save_params
from brax.training import pmap
from utils.agent_setup import create_agent
from utils.buffer_setup import create_dummy_transition, create_replay_buffer, prefill_replay_buffer_pmap, prepare_transitions_for_training, setup_offline_replay_buffer
from utils.env_utils import create_wrapped_env

from offline2online_gcrl.impls.utils.env_utils import get_experience
from offline2online_gcrl.impls.utils.evaluator import create_evaluator

_PMAP_AXIS_NAME = "i"


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def setup_rng(seed, num_envs, num_local_devices_to_use, process_id):
    rng = jax.random.PRNGKey(seed)
    agent_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)  # not sure if it makes sense to split and fold local_key here
    local_key, rb_key, offline_rb_key, env_key, eval_key = jax.random.split(local_key, 5)
    env_keys = jax.random.split(env_key, num_envs // jax.process_count())
    env_keys = jnp.reshape(env_keys, (num_local_devices_to_use, -1) + env_keys.shape[1:])
    return agent_key, local_key, rb_key, offline_rb_key, env_keys, eval_key


def train_fn(train_config, offline_agent_config, online_agent_config, progress_fn):
    agent_config = offline_agent_config

    # Reproducibility preparation for (optional) multi-GPU training
    process_id = jax.process_index()
    num_local_devices_to_use = jax.local_device_count()
    device_count = num_local_devices_to_use * jax.process_count()
    logging.info(
        "local_device_count: %s; total_device_count: %s",
        num_local_devices_to_use,
        device_count,
    )

    # sampled batch size should divisble by the batch size expected by agent.
    if ((train_config.episode_length - 1) * train_config.num_envs) % agent_config["batch_size"] != 0:
        raise ValueError("(episode_length - 1) * num_envs must be divisible by batch_size")
    assert train_config.num_envs % device_count == 0

    # Setup
    ## setup training variables
    num_env_steps_per_experience = train_config.num_envs * train_config.unroll_length
    num_env_steps_per_epoch = train_config.num_training_iterations_per_epoch * num_env_steps_per_experience
    num_prefill_experiences = train_config.seed_steps // num_env_steps_per_experience + 1
    offline_training_epochs = train_config.offline_train_steps // num_env_steps_per_epoch + 1
    online_training_epochs = train_config.online_train_steps // num_env_steps_per_epoch + 1
    logging.info("Seed steps: %s", train_config.seed_steps)
    logging.info("offline training epochs: %s", offline_training_epochs)
    logging.info("online training epochs: %s", online_training_epochs)
    logging.info("Num env steps per epoch: %s", num_env_steps_per_epoch)

    ## setup rng
    agent_key, local_key, online_rb_key, offline_rb_key, env_keys, eval_key = setup_rng(train_config.seed, train_config.num_envs, num_local_devices_to_use, process_id)

    ## setup env and initialise state
    # training env
    env, _ = create_wrapped_env(env_name=train_config.env_name, episode_length=train_config.episode_length, action_repeat=train_config.action_repeat)
    env_state = jax.pmap(env.reset)(env_keys)
    dummy_transition = create_dummy_transition(env)

    # setup offline buffer
    offline_replay_buffer, offline_buffer_state = setup_offline_replay_buffer(train_config, agent_config, dummy_transition, offline_rb_key, num_local_devices_to_use, device_count)

    # setup online replay buffer
    online_replay_buffer = create_replay_buffer(
        sample_batch_size=agent_config["batch_size"],
        replay_buffer_size=train_config.online_replay_buffer_size,
        num_envs=train_config.num_envs,
        device_count=device_count,
        dummy_transition=dummy_transition,
        episode_length=train_config.episode_length,
    )
    online_buffer_state = jax.pmap(online_replay_buffer.init)(jax.random.split(online_rb_key, num_local_devices_to_use))

    ## Create agent, init training state and replicate training state across devices
    agent = create_agent(config=agent_config, rng=agent_key, env=env, dummy_transition=dummy_transition, goal_conditioning=train_config.goal_conditioning, pretrained_agent=None)
    agent = jax.device_put_replicated(agent, jax.local_devices()[:num_local_devices_to_use])
    del agent_key

    def update_agent(carry, transitions):
        """
        Perform a single sgd step using a batch of transitions.
        """
        agent, key = carry
        key, key_update = jax.random.split(key, 2)
        agent, metrics = agent.update(transitions, key, pmap_axis_name=_PMAP_AXIS_NAME)
        return (agent, key), metrics

    def data_collection_step(agent, env_state, online_buffer_state, key):
        """
        Collect an experience in the environment and store in the replay buffer.
        """
        # Collect experience
        experience_key, training_key = jax.random.split(key)
        env_state, online_buffer_state = get_experience(
            agent,
            env,
            env_state,
            online_replay_buffer,
            online_buffer_state,
            experience_key,
            train_config.action_sample_temperature,
            train_config.goal_conditioning,
            train_config.episode_length,
            train_config.unroll_length,
            success_only=train_config.insert_only_successful_episodes,
        )

        agent = agent.replace(network=agent.network.replace(env_steps=agent.network.env_steps + num_env_steps_per_experience))
        return agent, env_state, online_buffer_state

    def training_step(agent, online_buffer_state, offline_buffer_state, key, mode):
        """
        For each training training_iteration:
            Sample from replay buffer.
            Update the networks episode_length * num_envs / batch_size times.
        """

        # Sample, process, shuffle, then train
        experience_key, training_key = jax.random.split(key, 2)

        def sample_mixed(states):
            offline_buffer_state, online_buffer_state = states
            offline_buffer_state, transitions_1 = offline_replay_buffer.sample(offline_buffer_state)
            online_buffer_state, transitions_2 = online_replay_buffer.sample(online_buffer_state)
            return offline_buffer_state, online_buffer_state, transitions_1, transitions_2

        def sample_offline(states):
            offline_buffer_state, online_buffer_state = states
            offline_buffer_state, transitions_1 = offline_replay_buffer.sample(offline_buffer_state)
            offline_buffer_state, transitions_2 = offline_replay_buffer.sample(offline_buffer_state)
            return offline_buffer_state, online_buffer_state, transitions_1, transitions_2

        # sample from buffers conditioned on training mode
        offline_buffer_state, online_buffer_state, transitions_1, transitions_2 = jax.lax.cond(mode == 1, sample_mixed, sample_offline, operand=(offline_buffer_state, online_buffer_state))

        transitions = prepare_transitions_for_training(train_config, agent_config, env, transitions_1, transitions_2, experience_key)

        ## Train
        # TODO: verify agent.network.step is updated.
        (agent, _), metrics = jax.lax.scan(update_agent, (agent, training_key), transitions)
        return agent, online_buffer_state, offline_buffer_state, metrics

    def scan_training_step(n, ts, bs, offline_bs, update_key, mode):
        def body(carry, unsued_t):
            ts, bs, offline_bs, update_key, mode = carry
            new_key, update_key = jax.random.split(update_key)
            ts, bs, offline_bs, metrics = training_step(ts, bs, offline_bs, update_key, mode)
            return (ts, bs, offline_bs, new_key, mode), metrics

        return jax.lax.scan(body, (ts, bs, offline_bs, update_key, mode), (), length=n)

    def training_epoch(agent, env_state, online_buffer_state, offline_buffer_state, key, mode):
        """
        Execute a set number of training iterations per epoch.
        A new episode is added to the buffer per training iteration and the networks are updated multiple number of times as controlled by num_training_steps_per_iteration.
        """

        def f(carry, unused_t):
            ts, es, bs, offline_bs, k, mode = carry
            k, new_key, update_key = jax.random.split(k, 3)
            # Conditionally execute data collection step only when mode == 1 (training online)
            ts, es, bs = jax.lax.cond(mode == 1, lambda _: data_collection_step(ts, es, bs, k), lambda _: (ts, es, bs), operand=None)
            # ts, es, bs = data_collection_step(ts, es, bs, k) # has num_env_steps_per_experience
            (ts, bs, offline_bs, update_key, mode), metrics = scan_training_step(
                train_config.num_training_steps_per_iteration, ts, bs, offline_bs, update_key, mode
            )  # has  num_training_steps_per_iteration * episode_length * num_envs / batch_size
            return (ts, es, bs, offline_bs, new_key, mode), metrics

        (agent, env_state, online_buffer_state, offline_buffer_state, key, mode), metrics = jax.lax.scan(
            f, (agent, env_state, online_buffer_state, offline_buffer_state, key, mode), (), length=train_config.num_training_iterations_per_epoch
        )

        # update metrics
        metrics["online_buffer_size"] = online_replay_buffer.size(online_buffer_state) * train_config.num_envs
        metrics["offline_buffer_size"] = offline_replay_buffer.size(offline_buffer_state) * train_config.num_envs
        metrics["num_gradient_updates"] = agent.network.step  # TODO: needs unpmap if using more than 1 device?
        metrics["num_environment_steps"] = agent.network.env_steps
        metrics["is_online"] = mode

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return agent, env_state, online_buffer_state, offline_buffer_state, metrics

    # execute a training epoch on each device. Data synchronization happens implicitly when operations like jax.lax.pmean are used.
    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    training_walltime = 0

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(agent, env_state, online_buffer_state, offline_buffer_state, key, mode):
        """
        Execute a training epoch and time it.
        """
        nonlocal training_walltime
        t = time.time()
        (agent, env_state, online_buffer_state, offline_buffer_state, metrics) = training_epoch(agent, env_state, online_buffer_state, offline_buffer_state, key, mode)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        steps_per_second = num_env_steps_per_epoch / epoch_training_time
        metrics = {
            "training/sps": steps_per_second,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return (agent, env_state, online_buffer_state, offline_buffer_state, metrics)

    ## Eval init
    eval_env, unwrapped_eval_env = create_wrapped_env(env_name=train_config.env_name, episode_length=train_config.episode_length, action_repeat=train_config.action_repeat)
    evaluator, make_policy = create_evaluator(train_config, agent, eval_env, eval_key)

    # Run initial eval
    # logging.info("execute initial eval...")
    # metrics = {}
    # if process_id == 0:
    #     metrics = evaluator.run_evaluation(_unpmap(agent.network.params), training_metrics={})
    #     logging.info(metrics)
    #     progress_fn(1, metrics, make_policy, _unpmap(agent.network.params), unwrapped_eval_env)

    # Collect/train/eval loop
    logging.info("execute training loop...")
    current_step = int(_unpmap(agent.network.env_steps))
    for epoch_num in range(online_training_epochs + offline_training_epochs):
        logging.info("env step %s", current_step)

        # Collect data and train
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, num_local_devices_to_use)

        # 1 if training online
        train_online = jnp.full((num_local_devices_to_use,), epoch_num >= offline_training_epochs, dtype=jnp.int32)

        # seed replay buffer when training online for the first time.
        if epoch_num == offline_training_epochs:
            logging.info("seeding online buffer")
            # prefill replay buffer
            t = time.time()
            prefill_key, local_key = jax.random.split(local_key)
            prefill_keys = jax.random.split(prefill_key, num_local_devices_to_use)
            agent, env_state, online_buffer_state, _ = prefill_replay_buffer_pmap(
                agent,
                env,
                env_state,
                online_replay_buffer,
                online_buffer_state,
                prefill_keys,
                num_env_steps_per_experience,
                num_prefill_experiences,
                train_config.action_sample_temperature,
                train_config.goal_conditioning,
                train_config.episode_length,
                train_config.unroll_length,
            )
            replay_size = jnp.sum(jax.vmap(online_replay_buffer.size)(online_buffer_state)) * jax.process_count()
            logging.info("replay size after prefill %s", replay_size)
            logging.info("env steps after prefill %s", int(_unpmap(agent.network.env_steps)))
            assert replay_size * train_config.num_envs >= train_config.seed_steps
            seed_fill_time = time.time() - t
            training_walltime += seed_fill_time

        (agent, env_state, online_buffer_state, offline_buffer_state, training_metrics) = training_epoch_with_timing(
            agent, env_state, online_buffer_state, offline_buffer_state, epoch_keys, mode=train_online
        )
        current_step = int(_unpmap(agent.network.env_steps))

        # Logging and evals
        if process_id == 0:
            ## Save policy and critic params
            if train_config.save_dir and ((epoch_num % train_config.save_interval) == 0):
                params = _unpmap(agent.network.params)
                path = f"{train_config.save_dir}/step_{current_step}.pkl"
                save_params(path, params)

            ## Run evals and log
            if (epoch_num % train_config.eval_interval) == 0:
                metrics = evaluator.run_evaluation(_unpmap(agent.network.params), training_metrics)
                logging.info(metrics)
                do_render = train_config.visualization_interval > 0 and (epoch_num % train_config.visualization_interval) == 0
                progress_fn(epoch_num + 2, metrics, make_policy, _unpmap(agent.network.params), unwrapped_eval_env, do_render)

    # Final validity checks
    # Verify number of steps is sufficient
    total_steps = current_step
    logging.info("total steps: %s", total_steps)
    assert total_steps >= train_config.online_train_steps

    # ## If there were no mistakes the training_state should still be identical on all devices
    pmap.assert_is_replicated(agent.network)
    pmap.synchronize_hosts()
