# card_detector.py

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

def show_anns(frame, anns):
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

def extract_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    logger.info(f"FPS of {frames_per_second} detected")
    frame_skipping_rate = int(frames_per_second)  # Number of frames to skip (one second's worth)

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

def detect_cards_in_frame(frame):
    # Convert the color space from BGR (OpenCV default) to RGB as expected by SAM
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use the SAM model to generate masks for the current frame
    masks = mask_generator.generate(rgb_frame)
    detected_cards = [mask for mask in masks]  # Extract bounding boxes from masks
    return detected_cards

def extract_card_names(frame, masks):
    card_names = []
    for mask in masks:
        # Assuming 'bbox' is in the format (x, y, width, height)
        x, y, w, h = mask['bbox']
        
        # Extract the ROI for the card's name
        # You may need to adjust the slicing depending on the exact location of the name on the card
        name_roi = frame[y:y+h, x:x+w]

        # Preprocess the ROI and perform OCR
        card_name = preprocess_and_ocr(name_roi)
        
        # Add the card's name to the array if it's not empty
        if card_name:
            card_names.append(card_name)
        else:
            card_names.append("No card")
    
    return card_names
def preprocess_and_ocr(text_zone):
    # Resize it to be bigger (so less pixelized)
    H = 50
    img_scale = H / text_zone.shape[0]
    new_size = (int(text_zone.shape[1] * img_scale), H)
    newimg = cv2.resize(text_zone, new_size)

    # Binarize it
    gray = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

    # Erode it
    kernel = np.ones((1, 1), np.uint8)
    erosion = cv2.erode(img_thresh, kernel, iterations=1)

    # OCR using Tesseract
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(erosion, config=custom_config)
    return text.strip()