""" Test that ``Env.fill()`` works correctly. """
import itertools

from hypothesis import given

from bees.env import Env
from bees.tests import strategies


@given(strategies.envs())
def test_env_fill_places_correct_number_of_agents(env: Env) -> None:
    """ Tests that we find a place for each agent in ``self.agents``. """
    env.fill()
    num_grid_agents = 0
    for row in env.grid:
        for square in row:
            if square[env.obj_type_ids["agent"]] == 1:
                num_grid_agents += 1
    assert num_grid_agents == len(env.agents)


@given(strategies.envs())
def test_env_fill_sets_all_agent_positions_correctly(env: Env) -> None:
    """ Tests that ``agent.pos`` is set correctly. """
    env.fill()
    agent_positions = [agent.pos for agent in env.agents.values()]
    for pos in itertools.product(range(env.width), range(env.height)):
        if pos in agent_positions:
            assert env.grid[pos][env.obj_type_ids["agent"]] == 1
        if env.grid[pos][env.obj_type_ids["agent"]] == 1:
            assert pos in agent_positions


@given(strategies.envs())
def test_env_fill_places_correct_num_foods(env: Env) -> None:
    """ Tests that we get exactly ``self.initial_num_foods`` in the grid. """
    env.fill()
    num_grid_foods = 0
    for pos in itertools.product(range(env.width), range(env.height)):
        if env.grid[pos][env.obj_type_ids["food"]] == 1:
            num_grid_foods += 1
    assert num_grid_foods == env.initial_num_foods


@given(strategies.envs())
def test_env_fill_generates_id_map_positions_correctly(env: Env) -> None:
    """ Tests that ``self.id_map`` is correct after ``self.fill`` is called. """
    env.fill()
    for x, y in itertools.product(range(env.width), range(env.height)):
        object_map = env.id_map[x][y]
        for obj_type_id, obj_ids in object_map.items():
            if len(obj_ids) > 0:
                assert env.grid[x][y][obj_type_id] == 1


@given(strategies.envs())
def test_env_fill_generates_id_map_ids_correctly(env: Env) -> None:
    """ Tests that ``self.id_map`` is correct after ``self.fill`` is called. """
    env.fill()
    for x, y in itertools.product(range(env.width), range(env.height)):
        object_map = env.id_map[x][y]
        for obj_type_id, obj_ids in object_map.items():
            if obj_type_id == env.obj_type_ids["agent"]:
                for obj_id in obj_ids:
                    assert obj_id in env.agents
