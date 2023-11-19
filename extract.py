import cv2
import pytesseract
from dataclasses import dataclass

@dataclass
class CardExtractInfo:
    set: str
    collector_number: str
    
def preprocess_for_ocr(roi):
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    processed_roi = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return processed_roi

def extract_text_by_area(frame, x, y, w, h, color=(0, 255, 0), draw_box=False):
    # Crop the area where the text is located
    roi = frame[y:y+h, x:x+w]
    # Preprocess the ROI for OCR
    processed_roi = preprocess_for_ocr(roi)
    # OCR with Tesseract
    custom_config = r'--oem 3 --psm 7'
    text = pytesseract.image_to_string(processed_roi, config=custom_config)
    
    if draw_box:
        # Draw a rectangle around the text area
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    return text.strip(), frame

def extract_card_info(frame, draw_boxes=False):
    # The exact height of the frame and the height of the card within the frame is needed to adjust these values accurately
    # These are example coordinates and need to be fine-tuned:
    card_height_from_bottom = 240  # The height of the area from the bottom of the frame to consider as the card's location
    card_width_from_edge = 100  # The width of the area from the left edge of the frame to consider as the card's location
    set_area_width = 250  # The width of the area to capture the set code
    collector_number_area_width = 250  # The width of the area to capture the collector number
    
    frame_height, frame_width = frame.shape[:2]
    set_area_x = card_width_from_edge  # Starting X coordinate, a bit of padding from the left edge of the frame
    collector_number_area_x = set_area_x  # Same X coordinate for the collector number area
    
    bounding_box_height = 50  # The height of the bounding box
    set_area_y = frame_height - card_height_from_bottom
    collector_number_area_y = set_area_y - bounding_box_height  # Above the set box

    set_area_coords = (set_area_x, set_area_y, set_area_width, bounding_box_height)
    collector_number_coords = (collector_number_area_x, collector_number_area_y, collector_number_area_width, bounding_box_height)

    # Extract text and optionally draw bounding boxes
    set_text, frame_with_set_box = extract_text_by_area(frame, *set_area_coords, color=(0, 0, 255), draw_box=draw_boxes)  # Red color for set
    collector_number_text, frame_with_collector_number_box = extract_text_by_area(frame_with_set_box, *collector_number_coords, color=(128, 0, 128), draw_box=draw_boxes)  # Purple color for collector_number

    card_info = CardExtractInfo(
        set=set_text,
        collector_number=collector_number_text
    )

    return card_info, frame_with_collector_number_box