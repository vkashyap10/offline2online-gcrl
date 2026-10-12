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
from utils.networks import GCActor, GCDiscreteActor, GCValue


class GCIVLAgentJAX(flax.struct.PyTreeNode):
    """Goal-conditioned implicit V-learning (GCIVL) agent.

    This is a variant of GCIQL that only uses a V function, without Q functions.
    """

    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (next_v1_t, next_v2_t) = self.network.select("target_value")(batch.extras["next_states"], batch.extras["value_goals"])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch.extras["rewards"] + self.config["discount"] * batch.extras["masks"] * next_v_t

        (v1_t, v2_t) = self.network.select("target_value")(batch.extras["states"], batch.extras["value_goals"])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch.extras["rewards"] + self.config["discount"] * batch.extras["masks"] * next_v1_t
        q2 = batch.extras["rewards"] + self.config["discount"] * batch.extras["masks"] * next_v2_t
        (v1, v2) = self.network.select("value")(batch.extras["states"], batch.extras["value_goals"], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config["expectile"]).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config["expectile"]).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the AWR actor loss."""
        v1, v2 = self.network.select("value")(batch.extras["states"], batch.extras["actor_goals"])
        nv1, nv2 = self.network.select("value")(batch.extras["next_states"], batch.extras["actor_goals"])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

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
            actor_info.update(
                {
                    "mse": jnp.mean((dist.mode() - batch.action) ** 2),
                    # 'std': jnp.mean(dist.scale_diag), # modified: the dist object does not have scale_diag dfined when state_dependent_std is set to True
                }
            )

        return actor_loss, actor_info

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
        networks_to_update=frozenset({"value", "actor"}),
    ):
        """Update the agent and return a new agent with information dictionary."""
        new_key, key = jax.random.split(key)

        def loss_fn(grad_params):
            loss, info = 0.0, {}
            if "value" in networks_to_update:
                value_loss, value_info = self.value_loss(batch, grad_params)
                loss += value_loss
                for k, v in value_info.items():
                    info[f"value/{k}"] = v
            if "actor" in networks_to_update:
                actor_loss, actor_info = self.actor_loss(batch, grad_params, new_key)
                loss += actor_loss
                for k, v in actor_info.items():
                    info[f"actor/{k}"] = v
            return loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn, pmap_axis_name=pmap_axis_name)

        # Update the target network only if value network is updated
        if "value" in networks_to_update:
            self.target_update(new_network, "value")

        # return a new agent
        return GCIVLAgentJAX(network=new_network, config=self.config), info

    # TODO: add with pmap_axis_name
    # @jax.jit
    # def update_high_utd(
    #     self,
    #     batch
    # ) -> Tuple["GCIVLAgentJAX", dict]:
    #     """
    #     Fast JITted high-UTD version of `.update`.

    #     Splits the batch into minibatches, performs `utd_ratio` critic updates,
    #     and then one actor update.

    #     Batch dimension must be divisible by `utd_ratio`.
    #     """
    #     batch_size = batch["rewards"].shape[0]
    #     assert (
    #         batch_size % self.config['utd'] == 0
    #     ), f"Batch size {batch_size} must be divisible by UTD ratio {self.config['utd']}"
    #     minibatch_size = batch_size // self.config["utd"]

    #     def scan_body(carry: Tuple[GCIVLAgentJAX], data: Tuple[dict]):
    #         (agent,) = carry
    #         (minibatch,) = data
    #         agent, info = agent.update(
    #             minibatch,
    #             networks_to_update=frozenset({"value"}),  # Updating value network only
    #         )
    #         return (agent,), info

    #     # Reshape the batch for UTD updates
    #     def make_minibatch(data: jnp.ndarray):
    #         return jnp.reshape(data, (self.config["utd"], minibatch_size) + data.shape[1:])

    #     minibatches = jax.tree_map(make_minibatch, batch)

    #     # Perform UTD updates for value network
    #     (agent,), value_infos = jax.lax.scan(scan_body, (self,), (minibatches,))

    #     # Aggregate information from value updates
    #     value_infos = jax.tree_map(lambda x: jnp.mean(x, axis=0), value_infos)

    #     # Perform a single actor update after all value updates
    #     agent, actor_infos = agent.update(
    #         batch,
    #         networks_to_update=frozenset({"actor"}),
    #     )

    #     # Merge the two info dictionaries
    #     infos = {**value_infos, **actor_infos}

    #     return agent, infos

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
            actions = jnp.clip(actions, -0.9999, 0.9999)  # modified to work with tanh_squash
        return actions

    @classmethod
    def create(
        cls,
        rng,
        ex_observations,
        ex_goals,  # remove assumption that goal dim == observation dim
        ex_actions,
        config,
        pretrained_agent=None,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
            pretrained_agent: Pretrained agent.
        """
        rng, init_rng = jax.random.split(rng, 2)

        if not pretrained_agent:
            # ex_goals = ex_observations
            if config["discrete"]:
                action_dim = ex_actions.max() + 1
            else:
                action_dim = ex_actions.shape[-1]

            # Define encoders.
            encoders = dict()
            if config["encoder"] is not None:
                encoder_module = encoder_modules[config["encoder"]]
                # concat_encoder used for goal conditioned policy and value function.
                encoders["value"] = GCEncoder(concat_encoder=encoder_module())
                encoders["actor"] = GCEncoder(concat_encoder=encoder_module())

            # Define value and actor networks.
            value_def = GCValue(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                ensemble=True,
                gc_encoder=encoders.get("value"),
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
                    state_dependent_std=True,  # modified
                    const_std=config["const_std"],
                    gc_encoder=encoders.get("actor"),
                    tanh_squash=True,  # modified: constrain actions to (-1,1)
                )

            network_info = dict(
                value=(value_def, (ex_observations, ex_goals)),
                target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
                actor=(actor_def, (ex_observations, ex_goals)),
            )
            networks = {k: v[0] for k, v in network_info.items()}
            network_args = {k: v[1] for k, v in network_info.items()}

            network_def = ModuleDict(networks)
            network_tx = optax.adam(learning_rate=config["lr"])
            network_params = network_def.init(init_rng, **network_args)["params"]
            network = TrainState.create(network_def, network_params, tx=network_tx)

            params = network_params
            params["modules_target_value"] = params["modules_value"]

            return cls(network=network, config=flax.core.FrozenDict(**config))
        else:
            return cls(network=pretrained_agent.network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="gcivl_jax",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=200,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            alpha=10.0,  # AWR temperature.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class="GCDataset",  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
            utd=4,  # update-to-data ratio of the critic while training online
            metrics=[
                "training/value/value_loss",  # metrics to record using wandb
                "training/value/v_mean",
                "training/value/v_max",
                "training/value/v_min",
                "training/actor/actor_loss",
                "training/actor/adv",
                "training/actor/bc_log_prob",
                "training/actor/mse",
            ],
        )
    )
    return config
