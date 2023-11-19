import argparse
from training.util import extract_and_save_frames
from training.predict_video import predict_video
from training.train import train
from pathlib import Path
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="MTG Card Detector")
    parser.add_argument("-r","--rotate", action="store_true", help="Rotate the video by 90 degrees")

    subparsers = parser.add_subparsers(dest="command")

    # Subparser for the train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", required=True, help="Path to model config file")
    train_parser.add_argument("--data", required=True, help="Path to dataset config file")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")

    # Subparser for the predict command
    predict_parser = subparsers.add_parser("predict", help="Predict using a trained model")
    predict_parser.add_argument("-o", "--output_dir", required=True, help="Path where the output video will be saved")
    predict_parser.add_argument("-v","--video", required=True, help="Path to the video file")
    predict_parser.add_argument("--model", required=True, help="Path to the trained model")

    # Subparser for extract_and_save_frames command
    extract_parser = subparsers.add_parser("extract_frames", help="Extract frames from a video")
    extract_parser.add_argument("-v","--video_path", type=str, required=True, help="Path to the video file")
    extract_parser.add_argument("-o","--output_dir", type=str, required=True, help="Directory to save extracted frames")
    extract_parser.add_argument("--skip_rate", type=int, default=1, help="Number of seconds to skip")

    args = parser.parse_args()

    if args.command == "train":
        train(args.config, args.data, args.epochs)
    elif args.command == "predict":
        predict_video(args.video, args.model, args.output_dir, rotate=args.rotate)
    elif args.command == "extract_frames":
        video_path = args.video_path
        output_dir = Path(args.output_dir)
        skip_rate = args.skip_rate

        logger.info(f"Extracting frames from {video_path}")
        extract_and_save_frames(video_path, output_dir, skip_rate, args.rotate)
        logger.info("Frame extraction complete.")

if __name__ == "__main__":
    main()
