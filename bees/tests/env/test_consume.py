import hypothesis.strategies as st
from hypothesis import given

from bees.tests import strategies as bst

@given(st.data())
def test_consume_removes_food_when_appropriate(data: st.DataObject) -> None:
    """ If the action is ``EAT`` and there's food, should disappear. """
    env = data.draw(bst.envs())
    env.reset()
    food_obj_type_id = env.obj_type_ids["food"]
    agent_food_positions: Dict[int, Tuple[int, int]] = {}
    for agent_id, agent in env.agents.items():
        if env.grid[agent.pos + (food_obj_type_id,)] == 1:
            agent_food_positions[agent_id] = agent.pos
    tuple_action_dict = data.draw(bst.tuple_action_dicts(env=env))

    eating_positions: List[Tuple[int, int]] = []
    for agent_id, pos in agent_food_positions.items():
        if tuple_action_dict[agent_id][1] == env.EAT:
            eating_positions.append(pos)

    env._consume(tuple_action_dict)
    
    for pos in eating_positions:
        assert env.grid[pos + (food_obj_type_id,)] == 0


# TODO: finish.
def test_consume_removes_nothing_else(data: st.DataObject) -> None:
    """ Otherwise, food should remain. """
    env = data.draw(bst.envs())
    env.reset()
    food_obj_type_id = env.obj_type_ids["food"]
    agent_food_positions: Dict[int, Tuple[int, int]] = {}
    for agent_id, agent in env.agents.items():
        if env.grid[agent.pos + (food_obj_type_id,)] == 1:
            agent_food_positions[agent_id] = agent.pos
    tuple_action_dict = data.draw(bst.tuple_action_dicts(env=env))

    eating_positions: List[Tuple[int, int]] = []
    for agent_id, pos in agent_food_positions.items():
        if tuple_action_dict[agent_id][1] == env.EAT:
            eating_positions.append(pos)

    env._consume(tuple_action_dict)
    
    for pos in eating_positions:
        assert env.grid[pos + (food_obj_type_id,)] == 0
    raise NotImplementedError
