""" Converts settings dictionary into config object. """
import sys
import json
from typing import List, Dict, Any

from pprint import pprint


# pylint: disable=no-member
class Config:
    """ Configuration object. """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.keys: List[str] = []
        self.settings: Dict[str, Any] = settings
        for key, value in settings.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)
            self.keys.append(key)


def main() -> None:
    """ Temporarily see if the class works. """
    settings_path = sys.argv[1]
    with open(settings_path, "r") as settings_file:
        settings = json.load(settings_file)
    config = Config(settings)
    print(config.env.height)
    print(config.keys)


if __name__ == "__main__":
    main()
