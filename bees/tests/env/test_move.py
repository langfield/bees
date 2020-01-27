import hypothesis.strategies as st
from hypothesis import given


@given(st.data())
def test_move_removes_agents_from_prev_positions(data: st.DataObject) -> None:
    """ Makes sure they actually move. """
    # TODO: Handle out-of-bounds errors.
    # TODO: Consider making the environment toroidal.
    raise NotImplementedError
