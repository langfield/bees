""" Test saving and loading of the trainer. """
import os
import json
import shutil
import argparse
import datetime
import tempfile
from typing import Any, Dict

import torch
import hypothesis.strategies as st
from hypothesis import given
from hypothesis import settings as hsettings

from bees.trainer import train
from bees.tests.strategies import bees_settings

# pylint: disable=no-value-for-parameter


@given(bees_settings(), st.integers(min_value=2, max_value=1000))
@hsettings(max_examples=100, deadline=datetime.timedelta(milliseconds=200))
def saving_and_loading(settings: Dict[str, Any], time_steps: int) -> None:
    """
    Test saving and loading.

    Note: This test case doesn't run (it doesn't start with "test", so pytest doesn't
    call it) on purpose because we decided to ditch it for now. Testing saving and
    loading is annoying because our trainer is monolithic, and we don't want to take the
    time to do it right now. We should come back and do this later.
    """

    settings["time_steps"] = time_steps
    settings["save_interval"] = time_steps // 2

    # NOTE: If this is set too high with multiprocessing on, cc will CRASH!
    settings["num_agents"] = min(5, settings["num_agents"])

    # DEBUG to figure out why hypothesis is throwing an OSError.
    settings["mp"] = False

    # CUDA check.
    if not torch.cuda.is_available():
        settings["cuda"] = False
    exit()

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
    with open("test_settings.json", "a+") as settings_file:
        json.dump(settings, settings_file, indent=4)

    # Call first training round.
    train(args)

    save_path = os.path.join(tempdir, "models/save_load_test/")
    save_root = os.path.join(tempdir, "models/")

    args_dict = {
        "settings": settings_path,
        "load_from": save_path,
        "save_path": "",
        "save_root": save_root,
    }
    args = argparse.Namespace(**args_dict)

    settings["time_steps"] *= 2
    with open(settings_path, "w") as settings_file:
        json.dump(settings, settings_file)

    # Call second training round.
    # train(args)

    # TODO: We need to figure out some way to get references to objects inside
    # training loop and environment at the end or during training, so we can
    # assert they have the appropriate values.
    shutil.rmtree(tempdir)


if __name__ == "__main__":
    test_saving_and_loading()
