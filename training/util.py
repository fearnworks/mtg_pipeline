#util.py
import cv2
from card_detector import extract_frames
from pathlib import Path
from loguru import logger

def crop_frame(frame, crop_size):
        # Crop from the bottom left corner
        height, width = frame.shape[:2]
        if width >= crop_size and height >= crop_size:
            frame = frame[height - crop_size:height, 0:crop_size]
        else:
            logger.warning("The frame is smaller than the cropping size, skipping crop.")
        
        # Resize frame to the desired size
        frame = cv2.resize(frame, (crop_size, crop_size))
        return frame

def save_frame(frame, frame_number, output_dir):
    """Saves a single frame to the specified directory."""
    output_path = output_dir / f"frame_{frame_number}.jpg"
    cv2.imwrite(str(output_path), frame)
    logger.info(f"Frame {frame_number} saved to {output_path}")

def extract_and_save_frames(video_path, output_directory, skip_rate=1, rotate=False, size=640):
    """
    Extracts frames from a video, resizes, optionally rotates them, and saves them to an output directory.
    
    Args:
        video_path (str): Path to the video file.
        output_directory (Path): Directory to save the frames.
        skip_rate (int): Number of frames to skip (default is 1).
        rotate (bool): Whether to rotate frames by 90 degrees clockwise.
        size (int): Size to which the images will be resized. Default is 640 for YOLO.
    """
    output_directory.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist
    for frame_count, frame in enumerate(extract_frames(video_path, skip_rate)):  #
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = crop_frame(frame, crop_size=size)
            # Save the frame
            save_frame(frame, frame_count, output_directory)
            frame_count += 1


def setup_scaffolding(output_dir):
    """Sets up the directory structure for a training run."""
    base_dir = Path(output_dir)
    for subdir in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Training scaffolding set up at {base_dir}")
