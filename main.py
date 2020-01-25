""" Run the environment. """
import argparse
from bees.trainer import train


def main() -> None:
    """ Main function to run bees environment. """

    # Get and validate args.
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-from", default="", help="Saved directory to load from.")
    parser.add_argument("--settings", default="", help="Settings file to use.")
    parser.add_argument(
        "--save-root",
        default="./models/",
        help="directory to save agent logs (default: ./models/)",
    )

    args = parser.parse_args()

    # Run training.
    train(args)


if __name__ == "__main__":
    main()
