#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests the ``utils.flat_action_to_tuple()`` function. """
import functools
from typing import List

from hypothesis import given
import hypothesis.strategies as st

from bees.utils import flat_action_to_tuple


@given(st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=5))
def test_flat_action_to_tuple_is_bijection(subaction_sizes: List[int]) -> None:
    """ Make sure this function generates a bijection. """

    results = []
    flat_action_size = functools.reduce(lambda a, b: a * b, subaction_sizes)
    for flat_action in range(flat_action_size):
        results.append(flat_action_to_tuple(flat_action, subaction_sizes))

    assert len(set(results)) == flat_action_size


@given(st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=5))
def test_flat_action_to_tuple_generates_valid_tuples(
    subaction_sizes: List[int],
) -> None:
    """ Make sure this function generates tuples inside of the given action space. """

    flat_action_size = functools.reduce(lambda a, b: a * b, subaction_sizes)
    for flat_action in range(flat_action_size):
        result = flat_action_to_tuple(flat_action, subaction_sizes)

        for i, subaction in enumerate(result):
            assert isinstance(subaction, int)
            assert 0 <= subaction < subaction_sizes[i]
