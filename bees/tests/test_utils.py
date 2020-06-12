#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from typing import Any, Dict

import hypothesis

# Set hypothesis settings.
hypothesis.settings.register_profile("test_settings", deadline=None)
hypothesis.settings.load_profile("test_settings")

# Read test settings file.
TEST_SETTINGS_PATH = "settings/test_settings.json"
with open(TEST_SETTINGS_PATH, "r") as test_settings_file:
    default_settings = json.load(test_settings_file)


def get_default_settings() -> Dict[str, Any]:
    """ Returns a copy of the default settings. """
    return dict(default_settings)
