""" Test that ``Env.reset()`` works correctly. """

from itertools import product

from hypothesis import given
from bees.env import Env
from bees.tests import strategies


# TODO: Everything.
@given(strategies.envs())
def test_env_reset_sees_correct_number_of_objects(env: Env) -> None:
    """ Tests that each observation has the corret number of each object type. """

    obs = env.reset()
    for agent_id, agent_obs in obs.items():

        # Calculate correct number of each object type.
        correct_obj_nums = {obj_type: 0 for obj_type in env.obj_type_ids.values()}
        for dx, dy in product(range(-env.sight_len, env.sight_len + 1), repeat=2):
            x = env.agents[agent_id].pos[0] + dx
            y = env.agents[agent_id].pos[0] + dy
            if (x, y) not in product(range(env.width), range(env.height)):
                continue
            for obj_type in env.obj_type_ids.values():
                correct_obj_nums[obj_type]
                env.id_map[x]
                env.id_map[x][y]
                env.id_map[x][y][obj_type]
                correct_obj_nums[obj_type] += len(env.id_map[x][y][obj_type])

        # Calculate number of each object type in returned observations.
        observed_obj_nums = {obj_type: 0 for obj_type in env.obj_type_ids.values()}
        for dx, dy in product(range(-env.sight_len, env.sight_len + 1), repeat=2):
            for obj_type in env.obj_type_ids.values():
                observed_obj_nums[obj_type] += agent_obs[dx][dy][obj_type]

        print("Correct: %s" % correct_obj_nums)
        print("Observed: %s" % observed_obj_nums)
        assert correct_obj_nums == observed_obj_nums
