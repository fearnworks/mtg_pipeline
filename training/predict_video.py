import os
import cv2
from ultralytics import YOLO

def predict_video(video_path, model_path, output_path, threshold=0.2, rotate=False):
    video_path_out = f'{output_path}.mp4'

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if rotate:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    # Load a model
    model = YOLO(model_path)

    while ret:
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
