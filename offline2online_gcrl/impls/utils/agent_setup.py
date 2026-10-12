from agents import agents

from offline2online_gcrl.impls.utils.replay_buffer import Transition


def create_agent(
    config,
    rng,
    env,
    dummy_transition: Transition,
    goal_conditioning=True,
    pretrained_agent=None,
):
    # set encoder
    if "encoder" in config and config["encoder"] == "none":
        config["encoder"] = None

    agent_class = agents[config["agent_name"]]

    if goal_conditioning:
        agent = agent_class.create(
            rng,
            ex_observations=dummy_transition.observation[: env.state_dim],
            ex_goals=dummy_transition.observation[env.state_dim :],
            ex_actions=dummy_transition.action,
            config=config,
            pretrained_agent=pretrained_agent,
        )
    else:
        agent = agent_class.create(
            rng,
            ex_observations=dummy_transition.observation[: env.state_dim],
            ex_actions=dummy_transition.action,
            config=config,
            pretrained_agent=pretrained_agent,
        )
    return agent
