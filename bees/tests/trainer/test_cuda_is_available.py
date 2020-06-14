import os
import sys
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
@given(bees_settings(), st.integers(min_value=2, max_value=1000))
@hsettings(max_examples=100, deadline=datetime.timedelta(milliseconds=200))
def test_cuda(settings: Dict[str, Any], time_steps: int) -> None:
    pass
    a = torch.cuda.is_available()
    print(a)
