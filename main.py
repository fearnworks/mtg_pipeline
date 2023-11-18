# main.py
from loguru import logger
import cv2
import argparse
from card_detector import extract_frames, detect_cards_in_frame, show_anns, extract_card_names

def process_video(video_path, rotate=False):
    logger.info("Processing video")
    frame_count = 0
    for frame in extract_frames(video_path):
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        detected_cards = detect_cards_in_frame(frame)
        logger.info(detected_cards)
        card_names = extract_card_names(frame, detected_cards)

        # Visualize the annotations on the frame
        frame_with_anns = show_anns(frame, detected_cards)

        # Save the frame
        cv2.imwrite(f"output/test/frame_{frame_count}.jpg", frame_with_anns)
        frame_count += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video and detect cards.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--rotate", action="store_true", help="Rotate the video by 90 degrees")
    args = parser.parse_args()
    
    if args.video_path:
        video_path = args.video_path
    process_video(args.video_path, args.rotate)