from src.Parser import parse_args
from src.Train import train
from src.Evaluate import evaluate


def _main_():
    args = parse_args()
    if args.train or not args.evaluate:
        train(
            episode_per_epoch=args.epe,
            gamma=args.gamma,
            lr=args.lr,
            actor_checkpoint=args.actor,
            critic_checkpoint=args.critic,
        )
    if args.evaluate:
        if not args.actor:
            raise ValueError("No actor checkpoint provided.")
        evaluate(
            actor_checkpoint=args.actor,
        )


if __name__ == "__main__":
    _main_()
