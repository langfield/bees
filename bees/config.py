""" Converts settings dictionary into config object. """
import sys
import json
from typing import List, Dict, Any

from pprint import pprint, pformat


class Config:
    """ Configuration object. """

    def __init__(self, settings: Dict[str, Any]) -> None:
        """ __init__ function for Config class. """

        self.keys: List[str] = []
        self.settings: Dict[str, Any] = settings
        for key, value in settings.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)
            self.keys.append(key)

    def __repr__(self) -> str:
        """ Return string representation of object. """

        # Try to use ``sort_dicts`` option, only available in Python 3.8.
        try:
            # pylint: disable=unexpected-keyword-arg
            formatted = pformat(self.settings, sort_dicts=False)
        except TypeError:
            formatted = pformat(self.settings)
        return formatted
