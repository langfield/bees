import os
import argparse

import torch


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument("--load-from", default="", help="Saved directory to load from.")
    parser.add_argument("--settings", default="", help="Settings file to use.")
    parser.add_argument(
        "--algo", default="ppo", help="algorithm to use: a2c | ppo | acktr"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="RMSprop optimizer alpha (default: 0.99)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--use-gae",
        action="store_true",
        default=False,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=1.0,
        help="gae lambda parameter (default: 1.0)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.0,
        help="entropy term coefficient (default: 0.0)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=1.0,
        help="value loss coefficient (default: 1.0)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--cuda-deterministic",
        action="store_true",
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="how many training CPU processes to use (default: 16)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=32,
        help="number of forward steps in A2C (default: 32)",
    )
    parser.add_argument(
        "--ppo-epoch", type=int, default=4, help="number of ppo epochs (default: 4)"
    )
    parser.add_argument(
        "--num-mini-batch",
        type=int,
        default=1,
        help="number of batches for ppo (default: 1)",
    )
    parser.add_argument(
        "--clip-param",
        type=float,
        default=0.3,
        help="ppo clip parameter (default: 0.3)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="log interval, one log per n updates (default: 10)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="save interval, one save per n updates (default: 100)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="eval interval, one eval per n updates (default: None)",
    )
    parser.add_argument(
        "--num-env-steps",
        type=int,
        default=10e6,
        help="number of environment steps to train (default: 10e6)",
    )
    parser.add_argument(
        "--env-name",
        default="PongNoFrameskip-v4",
        help="environment to train on (default: PongNoFrameskip-v4)",
    )
    parser.add_argument(
        "--log-dir",
        default="/tmp/gym/",
        help="directory to save agent logs (default: /tmp/gym)",
    )
    parser.add_argument(
        "--save-root",
        default="./models/",
        help="directory to save agent logs (default: ./models/)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--use-proper-time-limits",
        action="store_true",
        default=False,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--recurrent-policy",
        action="store_true",
        default=False,
        help="use a recurrent policy",
    )
    parser.add_argument(
        "--use-linear-lr-decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ["a2c", "ppo", "acktr"]
    if args.recurrent_policy:
        assert args.algo in [
            "a2c",
            "ppo",
        ], "Recurrent policy is not implemented for ACKTR"

    validate_args(args)

    return args

def validate_args(args: argparse.Namespace) -> None:
    """ Validates ``args``. Will raise ValueError if invalid arguments are given. """

    # Check for settings file or loading path.
    if not args.settings and not args.load_from:
        raise ValueError("Must either provide argument --settings or --load-from.")

    # Validate paths.
    if args.load_from and not os.path.isdir(args.load_from):
        raise ValueError("Invalid load directory for argument --load-from: '%s'." % args.load_from)
    if args.settings and not os.path.isfile(args.settings):
        raise ValueError("Invalid settings file for argument --settings: '%s'." % args.settings)

    # Check for missing --settings argument.
    if args.load_from and not args.settings:
        print("Warning: Argument --settings not provided, loading from '%s'." % args.load_from)
