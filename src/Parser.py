import argparse
from rich_argparse import RichHelpFormatter


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process some integers.',
        formatter_class=RichHelpFormatter
    )
    parser.add_argument(
        "--epe",
        default=100,
        type=int,
        help="The number of episodes per epoch.",
    )
    parser.add_argument(
        "--gamma",
        default=0.99,
        type=float,
        help="The discount factor.",
    )
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="The learning rate.",
    )
    parser.add_argument(
        "--actor",
        default="",
        type=str,
        help="The path to the actor checkpoint.",
    )
    parser.add_argument(
        "--critic",
        default="",
        type=str,
        help="The path to the critic checkpoint.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit("This is not a script.")
