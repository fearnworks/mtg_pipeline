# training_util.py
import argparse
from training.util import extract_and_save_frames
from training.predict_video import predict_video
from training.train import train
from pathlib import Path
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="MTG Card Detector")
    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument("--output_dir", type=str, help="Directory to save extracted frames")
    parser.add_argument("--video_path", type=str, help="Path to the video file")
    parser.add_argument("--rotate", action="store_true", help="Rotate the video by 90 degrees")
    parser.add_argument("--skip_rate", type=int, default=1, help="Number of seconds to skip")
    parser.add_argument("--setup_scaffolding", action="store_true", help="Setup directory structure for training")
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", required=True, help="Path to model config file")
    train_parser.add_argument("--data", required=True, help="Path to dataset config file")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")

    args = parser.parse_args()

    if args.command == "train":
        train(args.config, args.data, args.epochs)
    elif args.command == "predict":
        predict_video(args.video, args.model)

    video_path = args.video_path
    output_dir = Path(args.output_dir)
    skip_rate = args.skip_rate

    logger.info(f"Extracting frames from {video_path}")
    extract_and_save_frames(video_path, output_dir, skip_rate, args.rotate)
    logger.info("Frame extraction complete.")

if __name__ == "__main__":
    main()