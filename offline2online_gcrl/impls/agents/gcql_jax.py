import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCDiscreteCritic, GCValue


class GCQLAgentJAX(flax.struct.PyTreeNode):
    """Goal-conditioned Q-learning (GCQL) agent for the JAX offline-to-online training loop.

    Actor-critic with a twin Q critic and no value network. Supports AWR (actor_loss='awr') and DDPG (actor_loss='ddpg')
    for the actor loss. Differences from the GCQL agent this was ported from:
      - The TD target uses min(q1, q2) of the target critic; the original used the (q1, q2) pair directly, which broadcast the
        target to shape (2, batch).
      - The AWR baseline is Q(s, g, pi(s, g)); the original used the target critic at the dataset action, which is close to
        Q(s, g, a) itself and gives a near-zero advantage.
    """

    network: Any
    config: Any = nonpytree_field()

    def _q(self, q1, q2):
        # Q Optimism, can be required for some envs, e.g. antmaze
        # Philip J Ball, Laura Smith, Ilya Kostrikov, and Sergey Levine. Efficient online reinforcement learning
        # with offline data. arXiv preprint arXiv:2302.02948, 2023
        return q1 if self.config["q_optimism"] else jnp.minimum(q1, q2)

    def critic_loss(self, batch, grad_params, rng):
        """Compute the Q critic loss."""
        next_dist = self.network.select("actor")(batch.extras["next_states"], batch.extras["value_goals"])
        next_actions = next_dist.sample(seed=rng)

        next_q1, next_q2 = self.network.select("target_critic")(batch.extras["next_states"], batch.extras["value_goals"], next_actions)
        next_q = jnp.minimum(next_q1, next_q2)
        q = batch.extras["rewards"] + self.config["discount"] * batch.extras["masks"] * next_q

        q1, q2 = self.network.select("critic")(
            batch.extras["states"],
            batch.extras["value_goals"],
            batch.action,
            params=grad_params,
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR or DDPG)."""
        if self.config["actor_loss"] == "awr":
            # AWR loss with baseline V(s, g) ~ Q(s, g, pi(s, g)).
            q1, q2 = self.network.select("critic")(batch.extras["states"], batch.extras["actor_goals"], batch.action)
            q = self._q(q1, q2)
            pi_actions = jnp.clip(self.network.select("actor")(batch.extras["states"], batch.extras["actor_goals"]).mode(), -1, 1)
            v1, v2 = self.network.select("critic")(batch.extras["states"], batch.extras["actor_goals"], pi_actions)
            v = self._q(v1, v2)
            adv = jax.lax.stop_gradient(q - v)

            exp_a = jnp.exp(adv * self.config["alpha"])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select("actor")(batch.extras["states"], batch.extras["actor_goals"], params=grad_params)
            log_prob = dist.log_prob(batch.action)

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                "actor_loss": actor_loss,
                "adv": adv.mean(),
                "bc_log_prob": log_prob.mean(),
            }
            if not self.config["discrete"]:
                actor_info["mse"] = jnp.mean((dist.mode() - batch.action) ** 2)

            return actor_loss, actor_info
        elif self.config["actor_loss"] == "ddpg":
            # DDPG loss.
            assert not self.config["discrete"]

            dist = self.network.select("actor")(batch.extras["states"], batch.extras["actor_goals"], params=grad_params)
            if self.config["const_std"]:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select("critic")(batch.extras["states"], batch.extras["actor_goals"], q_actions)
            q = self._q(q1, q2)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch.action)

            return q_loss, {
                "actor_loss": q_loss,
                "q_mean": q.mean(),
                "q_abs_mean": jnp.abs(q).mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean((dist.mode() - batch.action) ** 2),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @partial(jax.jit, static_argnames=("networks_to_update", "pmap_axis_name"))
    def update(
        self,
        batch,
        key,
        pmap_axis_name,
        networks_to_update=frozenset({"critic", "actor"}),
    ):
        """Update the agent and return a new agent with information dictionary."""
        actor_key, critic_key = jax.random.split(key)

        def loss_fn(grad_params):
            loss, info = 0.0, {}
            if "critic" in networks_to_update:
                critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_key)
                loss += critic_loss
                for k, v in critic_info.items():
                    info[f"critic/{k}"] = v
            if "actor" in networks_to_update:
                actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_key)
                loss += actor_loss
                for k, v in actor_info.items():
                    info[f"actor/{k}"] = v
            return loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn, pmap_axis_name=pmap_axis_name)

        # Update the target network only if the critic is updated
        if "critic" in networks_to_update:
            self.target_update(new_network, "critic")

        return GCQLAgentJAX(network=new_network, config=self.config), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select("actor")(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config["discrete"]:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        rng,
        ex_observations,
        ex_goals,
        ex_actions,
        config,
        pretrained_agent=None,
    ):
        """Create a new agent.

        Args:
            rng: Random key.
            ex_observations: Example observations.
            ex_goals: Example goals (the goal dimension can differ from the observation dimension).
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
            pretrained_agent: Pretrained agent.
        """
        rng, init_rng = jax.random.split(rng, 2)

        if pretrained_agent:
            return cls(network=pretrained_agent.network, config=flax.core.FrozenDict(**config))

        if config["discrete"]:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = GCEncoder(concat_encoder=encoder_module())
            encoders["actor"] = GCEncoder(concat_encoder=encoder_module())

        # Define actor and critic networks.
        if config["discrete"]:
            critic_def = GCDiscreteCritic(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                ensemble=True,
                gc_encoder=encoders.get("critic"),
                action_dim=action_dim,
            )
        else:
            critic_def = GCValue(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                ensemble=True,
                gc_encoder=encoders.get("critic"),
            )

        if config["discrete"]:
            actor_def = GCDiscreteActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                gc_encoder=encoders.get("actor"),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config["const_std"],
                gc_encoder=encoders.get("actor"),
            )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_critic"] = params["modules_critic"]

        return cls(network=network, config=flax.core.FrozenDict(**config))


def get_config():
    # Goals are relabelled by the replay buffer (TrajectoryUniformSamplingQueue.flatten_gcrl_fn), so OGBench's
    # GCDataset goal-sampling options do not apply here.
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="gcql_jax",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=200,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Critic network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            actor_loss="ddpg",  # Actor loss type ('awr' or 'ddpg').
            alpha=0.3,  # Temperature in AWR (unused by DDPG).
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            q_optimism=False,  # Use q1 instead of min(q1, q2) in the actor loss.
            metrics=[
                "training/critic/critic_loss",  # metrics to record using wandb
                "training/critic/q_mean",
                "training/actor/actor_loss",
                "training/actor/q_mean",
                "training/actor/adv",
                "training/actor/bc_log_prob",
                "training/actor/mse",
            ],
        )
    )
    return config
