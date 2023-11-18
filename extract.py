import os

import cv2
import pytesseract
from matplotlib import pyplot as plt

def extract_text_from_image(img_path):
    # Load the image using OpenCV
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    # Resize the image
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 15, 4)  # Adjust block size and constant
    
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    
    # Visualize the images
    plt.figure(figsize=(15, 15))
    plt.subplot(141), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(142), plt.imshow(gray, cmap='gray'), plt.title('Grayscale Image')
    plt.subplot(143), plt.imshow(thresh, cmap='gray'), plt.title('Thresholded Image')
    plt.subplot(144), plt.imshow(denoised, cmap='gray'), plt.title('Denoised Image')
    plt.show()

    # Use pytesseract to extract text
    extracted_text = pytesseract.image_to_string(denoised)
    print("Extracted Text:", extracted_text)
    
    # Use pytesseract to extract textQ
    extracted_text = pytesseract.image_to_string(gray)
    print("Extracted Text:", extracted_text)

    return extracted_text.strip()

path = os.getcwd() + '/input/frame.png'
print(path)
text = extract_text_from_image(path)
print(text)

