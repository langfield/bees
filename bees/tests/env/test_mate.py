from typing import Tuple
import hypothesis.strategies as st
from hypothesis import given, assume

from bees.tests import strategies as bst
from bees.agent import Agent

# pylint: disable=no-value-for-parameter


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


@given(st.data())
def test_mate_executes_action(data: st.DataObject) -> None:
    """ Tests children are created when they're suppsed to. """
    env = data.draw(bst.envs())

    assume(env.height * env.width >= 3)

    # Generate two adjacent positions.
    mom_pos = data.draw(bst.positions(env=env))
    open_positions = env._get_adj_positions(mom_pos)
    dad_pos = data.draw(st.sampled_from(open_positions))

    # Create a mom and dad.
    mom = Agent(
        config=env.config, num_actions=env.num_actions, pos=mom_pos, initial_health=1.0,
    )
    dad = Agent(
        config=env.config, num_actions=env.num_actions, pos=dad_pos, initial_health=1.0,
    )
    mom.is_mature = True
    dad.is_mature = True
    mom.mating_cooldown = 0
    dad.mating_cooldown = 0
    mom_id = env._new_agent_id()
    dad_id = env._new_agent_id()
    env.agents[mom_id] = mom
    env.agents[dad_id] = dad
    env._place(env.obj_type_ids["agent"], mom_pos, mom_id)
    env._place(env.obj_type_ids["agent"], dad_pos, dad_id)

    # Construct subactions.
    mom_move = data.draw(bst.moves(env=env))
    dad_move = data.draw(bst.moves(env=env))
    mom_consumption = data.draw(bst.consumptions(env=env))
    dad_consumption = data.draw(bst.consumptions(env=env))
    mom_action = (mom_move, mom_consumption, env.MATE)
    dad_action = (dad_move, dad_consumption, env.MATE)

    action_dict = {mom_id: mom_action, dad_id: dad_action}
    child_ids = env._mate(action_dict)
    assert len(child_ids) == 1
    child = env.agents[child_ids.pop()]

    def adjacent(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """ Decide whether or not two positions are orthogonally adjacent. """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    assert len(env.agents) == 3
    assert adjacent(child.pos, mom.pos) or adjacent(child.pos, dad.pos)
