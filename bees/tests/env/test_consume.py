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


@given(st.data())
def test_consume_removes_nothing_else(data: st.DataObject) -> None:
    """ Otherwise, food should remain in place. """
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
    
    food_positions: Set[Tuple[int, int]] = set()
    for x in range(env.width):
        for y in range(env.height):
            if env.grid[(x, y) + (food_obj_type_id,)] == 1:
                food_positions.add((x, y))
    persistent_food_positions = food_positions - set(eating_positions)

    env._consume(tuple_action_dict)
    
    for pos in persistent_food_positions:
        assert env.grid[pos + (food_obj_type_id,)] == 1


@given(st.data())
def test_consume_decreases_num_foods_correctly(data: st.DataObject) -> None:
    """ The ``env.num_foods`` attribute is decremented properly. """
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

    old_num_foods = env.num_foods
    env._consume(tuple_action_dict)
    assert old_num_foods - env.num_foods == len(eating_positions)


@given(st.data())
def test_consume_makes_agent_health_nondecreasing(data: st.DataObject) -> None:
    """ Tests that agent.health in the correct direction. """
    env = data.draw(bst.envs())
    env.reset()
    tuple_action_dict = data.draw(bst.tuple_action_dicts(env=env))

    old_healths: Dict[int, float] = {}
    for agent_id, agent in env.agents.items():
        old_healths[agent_id] = agent.health

    env._consume(tuple_action_dict)

    for agent_id, agent in env.agents.items():
        assert old_healths[agent_id] <= agent.health
