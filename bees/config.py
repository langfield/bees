""" Converts settings dictionary into config object. """
from typing import List, Dict, Any

from pprint import pformat


class Config:
    """ Configuration object. """

    # TODO: Needs testing.

    def __init__(self, settings: Dict[str, Any]) -> None:
        """ __init__ function for Config class. """

        self.keys: List[str] = []

        # TODO: Does this need to be deep?
        self.settings: Dict[str, Any] = settings.copy()
        for key, value in settings.items():
            if isinstance(value, dict):
                value = Config(value)
                self.settings[key] = value
            setattr(self, key, value)
            self.keys.append(key)

    def __getattr__(self, name: str) -> Any:
        """ Override to make mypy happy. """
        return self.settings[name]

    def __repr__(self) -> str:
        """ Return string representation of object. """

        # Try to use ``sort_dicts`` option, only available in Python 3.8.
        try:
            # pylint: disable=unexpected-keyword-arg
            formatted = pformat(self.settings, sort_dicts=False)  # type: ignore
        except TypeError:
            formatted = pformat(self.settings)
        return formatted
