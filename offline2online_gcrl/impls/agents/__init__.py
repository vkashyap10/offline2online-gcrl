# Agents compatible with the JAX offline-to-online training loop (utils/training.py).
# Ported from OGBench (https://github.com/seohongpark/ogbench) to read replay-buffer Transitions and pmap.
from offline2online_gcrl.impls.agents.gciql_jax import GCIQLAgentJAX
from offline2online_gcrl.impls.agents.gcivl_jax import GCIVLAgentJAX
from offline2online_gcrl.impls.agents.gcql_jax import GCQLAgentJAX
from offline2online_gcrl.impls.agents.iql import IQLAgent

agents = dict(gcivl_jax=GCIVLAgentJAX, gciql_jax=GCIQLAgentJAX, gcql_jax=GCQLAgentJAX, iql=IQLAgent)
