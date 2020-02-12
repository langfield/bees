#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test that the ``Config`` class loads dictionaries correctly. """
from typing import Dict, Any
from hypothesis import given
from bees.tests import strategies
from bees.config import Config

# pylint: disable=no-value-for-parameter


@given(strategies.settings_dicts())
def test_config_adds_all_keys_from_nested_dicts(settings: Dict[str, Any]) -> None:
    """ Test that all keys are added when the dictionary is nested. """
    config = Config(settings)

    def check_keys(mapping: Dict[str, Any], config: Config) -> None:
        """ Recursively add all keys from a nested dictionary. """
        for key, value in mapping.items():
            if isinstance(value, dict):
                check_keys(value, getattr(config, key))
            assert key in config.keys()

    check_keys(settings, config)


@given(strategies.settings_dicts())
def test_config_attributes_get_set(settings: Dict[str, Any]) -> None:
    """ Test that all keys are set as attributes. """
    config = Config(settings)

    def check_attributes(mapping: Dict[str, Any], config: Config) -> None:
        """ Recursively add all keys from a nested dictionary. """
        for key, value in mapping.items():
            if isinstance(value, dict):
                check_attributes(value, getattr(config, key))
            assert hasattr(config, key)

    check_attributes(settings, config)


def test_config_settings_passed_by_value() -> None:
    """ Test that modifying ``Config.settings`` doesn't change the argument dict. """
    settings = {"key": {"subkey": [1, 2]}}
    config = Config(settings)
    settings["key"]["subkey"].append(3)
    assert 3 not in config.key.subkey


@given(strategies.settings_dicts())
def test_config_repr_prints_everything(settings: Dict[str, Any]) -> None:
    """ Test that every key appears in the string representation. """
    config_repr = repr(Config(settings))
    for key in settings:
        assert repr(key) in config_repr
