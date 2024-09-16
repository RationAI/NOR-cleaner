import argparse
import logging

import scripts.evaluate
import scripts.predict
import scripts.prepare
import scripts.train_model
from lib import LOG_CONFIG_KWARGS

# Set up logging
logging.basicConfig(level=logging.INFO, **LOG_CONFIG_KWARGS)  # type: ignore

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool for LPZ-NOR Decision System"
    )

    # Add commands `prepare`, `train`, and `predict`
    subparsers = parser.add_subparsers(dest="command")

    # Add `prepare` command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare the data")

    # Add `train` command
    train_parser = subparsers.add_parser("train", help="Train the model")

    # Add `predict` command
    predict_parser = subparsers.add_parser(
        "predict", help="Predict the target variable"
    )

    # Add `evaluate` command
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate the model performance"
    )

    args = parser.parse_args()

    if args.command == "prepare":
        logger.info("Preparing the data...")
        scripts.prepare.main()
    elif args.command == "train":
        logger.info("Training the model...")
        scripts.train_model.main()
    elif args.command == "predict":
        logger.info("Predicting the target variable...")
        scripts.predict.main()
    elif args.command == "evaluate":
        logger.info("Evaluating the model performance...")
        scripts.evaluate.main()
    else:
        # Print help message
        parser.print_help()

    # Add empty line for better readability
    logger.info("")


if __name__ == "__main__":
    main()
