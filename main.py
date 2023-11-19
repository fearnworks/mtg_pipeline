# main.py
from loguru import logger
import cv2
import argparse
from card_detector import extract_frames, detect_cards_in_frame, show_anns
from extract import extract_card_info  # Assuming the new functions are in this module

def process_video(video_path, rotate=False):
    logger.info("Processing video")
    frame_count = 0
    all_cards_info = []  # To store information of all cards detected in video

    for frame in extract_frames(video_path):
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        card_segment = detect_cards_in_frame(frame)
        # logger.info(f"Detected cards: {detected_cards}")

        # Extract card information using the new function
        card_info_list, bounded_frame = extract_card_info(frame, draw_boxes=True)
        
        logger.info(f"Processing Frame {frame_count}: {card_info_list}")
        all_cards_info.append(card_info_list)  # Add to our list of all card info

        # Visualize the annotations on the frame
        frame_with_anns = show_anns(frame, card_segment)

        # Save the frame
        cv2.imwrite(f"output/test/frame_{frame_count}_seg.jpg", frame_with_anns)
        cv2.imwrite(f"output/test/frame_{frame_count}_bound.jpg", bounded_frame)
        frame_count += 1

    # Log or process the extracted card information as needed
    for card_info in all_cards_info:
        logger.info(f"Set: {card_info.set}, Collector Number: {card_info.collector_number}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video and detect cards.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--rotate", action="store_true", help="Rotate the video by 90 degrees")
    args = parser.parse_args()
    
    process_video(args.video_path, args.rotate)