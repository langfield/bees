""" Test that the ``Config`` class loads dictionaries correctly. """
from typing import Dict, Any
import hypothesis.strategies as st
from hypothesis import given
from bees.config import Config


@given(st.dictionaries(keys=st.from_regex(r"[a-zA-Z_-]+"), values=strategies.config_values()))
def test_config_adds_all_keys_from_nested_dicts(settings: Dict[str, Any]) -> None:
    """ Test that all keys are added when the dictionary is nested. """
    pass

@given(st.dictionaries(keys=st.from_regex(r"[a-zA-Z_-]+"), values=strategies.config_values()))
def test_config_attributes_get_set(settings: Dict[str, Any]) -> None:
    """ Test that all keys are set as attributes. """
    pass

@given(st.dictionaries(keys=st.from_regex(r"[a-zA-Z_-]+"), values=strategies.config_values()))
def test_config_settings_passed_by_value(settings: Dict[str, Any]) -> None:
    """ Test that modifying ``Config.settings`` doesn't change the argument dict. """
    pass

@given(st.dictionaries(keys=st.from_regex(r"[a-zA-Z_-]+"), values=strategies.config_values()))
def test_config_repr_prints_everything(settings: Dict[str, Any]) -> None:
    """ Test that every key appears in the string representation. """
    pass

