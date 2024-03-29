#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Run to animate an environment log. """
from typing import List
import argparse
import time


def main(args: argparse.Namespace) -> None:
    """ Plays a log file of environment states as an animation. """

    # Read in log file
    with open(args.log_path, "r") as f:
        log_lines = f.readlines()

    # Parse log file into separate timesteps
    timesteps = []
    timestep: List[str] = []
    for line in log_lines:
        if line == ",":
            timesteps.append(list(timestep))
            timestep = []
        else:
            timestep.append(line)

    # Output each timestep
    for timestep in timesteps:
        for line in timestep:
            print(line)
        time.sleep(1)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("log_path", type=str, help="Path of log to play.")
    ARGS = PARSER.parse_args()
    main(ARGS)
