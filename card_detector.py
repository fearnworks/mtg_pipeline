# card_detector.py
from typing import List, Generator
import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from loguru import logger 
import pytesseract
sam_checkpoint = "sam_vit_h_4b8939.pth" 
model_type = "vit_h"  
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,  # Reduced density to favor larger objects
    pred_iou_thresh=0.90,  # Higher IOU threshold for better quality masks
    stability_score_thresh=0.95,  # Higher stability score for more reliable masks
    min_mask_region_area=5000,  # Set a higher minimum area to focus on larger masks
)
logger.info(device)



def show_anns(frame: np.ndarray, anns: List[dict]) -> np.ndarray:
    for ann in anns:
        mask = ann['segmentation']
        bbox = ann['bbox']
        color = list(np.random.random(size=3) * 256)  

        colored_mask = np.zeros_like(frame, frame.dtype)
        for i in range(3):  # Assuming frame has 3 channels (BGR)
            colored_mask[:, :, i][mask] = color[i]

        # Combine the current frame with the colored mask using addWeighted
        frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

        # Draw bounding box
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    return frame

def extract_frames(video_path: str, skip_rate: int) -> Generator[np.ndarray, None, None]:
    video = cv2.VideoCapture(video_path)
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    logger.info(f"FPS of {frames_per_second} detected")
    frame_skipping_rate = int(frames_per_second) * skip_rate # Number of frames to skip (one second's worth)

    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Only process one frame per second
        if frame_count % frame_skipping_rate == 0:
            yield frame
        frame_count += 1

    video.release()

def detect_cards_in_frame(frame: np.ndarray) -> List[dict]:
    # Convert the color space from BGR (OpenCV default) to RGB as expected by SAM
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use the SAM model to generate masks for the current frame
    masks = mask_generator.generate(rgb_frame)

    # Find the largest mask by area
    largest_mask = max(masks, key=lambda mask: mask['bbox'][2] * mask['bbox'][3], default=None)

    # Return a list with only the largest mask if any masks were detected
    return [largest_mask] if largest_mask else []