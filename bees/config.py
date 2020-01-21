""" Converts settings dictionary into config object. """
# NOTE: https://stackoverflow.com/questions/42272335/how-to-make-a-class-which-has-getattr-properly-pickable
import copy
from pprint import pformat
from typing import List, Dict, Any


class Config:
    """
    Configuration object.

    Parameters
    ----------
    settings : ``Dict[str, Any]``.
    """

    def __init__(self, settings: Dict[str, Any]):

        self.keys: List[str] = []
        self.settings: Dict[str, Any] = copy.deepcopy(settings)

        # Add all key-value pairs.
        for key, value in self.settings.items():
            if isinstance(value, dict):
                value = Config(value)
                self.settings[key] = value
            setattr(self, key, value)
            self.keys.append(key)

    def __getattr__(self, item: str) -> Any:
        """ Override to make mypy happy. """
        try:
            settings = super().__getattribute__("settings")
            return settings[item]
        except KeyError:
            return super().__getattribute__(item)

    def __repr__(self) -> str:
        """ Return string representation of object. """

        # Try to use ``sort_dicts`` option, only available in Python 3.8.
        try:
            # pylint: disable=unexpected-keyword-arg
            formatted = pformat(self.settings, sort_dicts=False)  # type: ignore
        except TypeError:
            formatted = pformat(self.settings)
        return formatted
