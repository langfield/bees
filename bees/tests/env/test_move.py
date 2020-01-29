import hypothesis.strategies as st
from hypothesis import given
from bees.tests import strategies as bst

# pylint: disable=no-value-for-parameter

@given(st.data())
def test_move_moves_or_does_nothing(data: st.DataObject) -> None:
    """ Makes sure they actually move or STAY. """
    # TODO: Handle out-of-bounds errors.
    # TODO: Consider making the environment toroidal.
    env = data.draw(bst.envs())
    env.reset()
    tuple_action_dict = data.draw(bst.tuple_action_dicts(env=env))
    executed_dict = env._move(tuple_action_dict)

    raise NotImplementedError


@given(st.data())
def test_move_holds_other_actions_invariant(data: st.DataObject) -> None:
    """ Makes sure the returned action dict only modifies move subaction space. """
    env = data.draw(bst.envs())
    env.reset()

    tuple_action_dict = data.draw(bst.tuple_action_dicts(env=env))
    executed_dict = env._move(tuple_action_dict)

    pairs = zip(list(tuple_action_dict.values()), list(executed_dict.values()))
    for attempted_action, executed_action in pairs:
        assert attempted_action[1:] == executed_action[1:]


@given(st.data())
def test_move_only_changes_to_stay(data: st.DataObject) -> None:
    """ Makes sure the returned action dict only changes to STAY if at all. """
    env = data.draw(bst.envs())
    env.reset()

    tuple_action_dict = data.draw(bst.tuple_action_dicts(env=env))
    executed_dict = env._move(tuple_action_dict)

    pairs = zip(list(tuple_action_dict.values()), list(executed_dict.values()))
    for attempted_action, executed_action in pairs:
        if attempted_action[0] != executed_action[0]:
            assert executed_action[0] == env.STAY
