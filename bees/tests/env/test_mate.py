import hypothesis.strategies as st
from hypothesis import given

from bees.tests import strategies as bst

@given(st.data())
def test_mate_makes_num_agents_nondecreasing(data: st.DataObject) -> None:
    """ Makes sure ``len(agents)`` is nondecreasing. """
    env = data.draw(bst.envs())
    env.reset()
    old_num_agents = len(env.agents)
    tuple_action_dict = data.draw(bst.tuple_action_dicts(env=env))
    env._mate(tuple_action_dict)
    assert old_num_agents <= len(env.agents)


@given(st.data())
def test_mate_adds_children_to_agents(data: st.DataObject) -> None:
    """ Makes sure child ids get added to ``env.agents``. """
    env = data.draw(bst.envs())
    env.reset()
    old_agent_memory_addresses = [id(agent) for agent in env.agents.values()]
    tuple_action_dict = data.draw(bst.tuple_action_dicts(env=env))
    child_ids = env._mate(tuple_action_dict)
    for child_id in child_ids:
        assert child_id in env.agents


@given(st.data())
def test_mate_children_are_new(data: st.DataObject) -> None:
    """ Makes sure children are new. """
    env = data.draw(bst.envs())
    env.reset()
    old_agent_memory_addresses = [id(agent) for agent in env.agents.values()]
    tuple_action_dict = data.draw(bst.tuple_action_dicts(env=env))
    child_ids = env._mate(tuple_action_dict)
    for child_id in child_ids:
        assert id(env.agents[child_id]) not in old_agent_memory_addresses
