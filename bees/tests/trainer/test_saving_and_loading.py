""" Test saving and loading of the trainer. """
import os
import json
import shutil
import tempfile
import argparse
from typing import Dict, Any

import hypothesis.strategies as st
from hypothesis import given

from bees.trainer import train
from bees.tests.strategies import bees_settings

# pylint: disable=no-value-for-parameter


@given(bees_settings(), st.integers(min_value=2, max_value=1000))
def test_saving_and_loading(settings: Dict[str, Any], time_steps: int) -> None:
    """ Test saving and loading. """
    settings["time_steps"] = time_steps
    settings["save_interval"] = time_steps // 2

    # Create settings file.
    tempdir = tempfile.mkdtemp()
    settings_path = os.path.join(tempdir, "settings.json")
    with open(settings_path, "w") as settings_file:
        json.dump(settings, settings_file)

    # Create args object.
    save_path = os.path.join(tempdir, "models/save_load_test/")
    args_dict = {
        "settings": settings_path,
        "load_from": "",
        "save_path": save_path,
        "save_root": "",
    }
    args = argparse.Namespace(**args_dict)

    # Call first training round.
    train(args)

    args_dict = {"settings": settings_path, "load_from": save_path, "save_root": ""}
    args = argparse.Namespace(**args_dict)

    # Call second training round.
    train(args)

    # TODO: We need to figure out some way to get references to objects inside
    # training loop and environment at the end or during training, so we can
    # assert they have the appropriate values.
    shutil.rmtree(tempdir)
