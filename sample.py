import math
import time

import cv2
import cvzone
from ultralytics import YOLO

confidence = 0.6

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(3, 640)
cap.set(4, 480)

model = YOLO("yolov8n.pt")

classNames = ["real", "fake"]

def analyze_spoof(imgFace):
    if imgFace is None or imgFace.size == 0:
        return False, ""
    
    gray = cv2.cvtColor(imgFace, cv2.COLOR_BGR2GRAY)
    
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    glare_ratio = cv2.countNonZero(thresh) / (imgFace.shape[0] * imgFace.shape[1] + 1e-6)
    
    # Hackathon heuristic logic
    if glare_ratio > 0.04:  # Threshold for bright glare/reflection
        return True, "Replay Attack"
    if blur_var < 50:       # Low variance generally means a flat/blurry printed image
        return True, "Printed Image"
        
    return False, ""

prev_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        print("Warning: Failed to read frame.")
        break

    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            if conf > confidence:
                if cls == 0: # person
                    # Crop face carefully keeping bounds in check
                    imgFace = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
                    is_fake, attack_type = analyze_spoof(imgFace)
                    
                    if is_fake:
                        label = f"FAKE - {attack_type}"
                        color = (0, 0, 255)
                    else:
                        label = "REAL"
                        color = (0, 255, 0)
                elif cls == 67: # cell phone
                    label = "FAKE - Phone Spoof"
                    color = (0, 0, 255)
                elif cls in [62, 63]: # TV or Laptop
                    label = "FAKE - Replay Attack"
                    color = (0, 0, 255)
                elif cls in [73, 76]: # book / paper
                    label = "FAKE - Printed Image"
                    color = (0, 0, 255)
                else:
                    continue

                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(
                    img,
                    f'{label} {int(conf * 100)}%',
                    (max(0, x1), max(35, y1)),
                    scale=1.5, thickness=2, colorR=color, colorB=color
                )

    # Avoid division by zero on first frame
    if prev_frame_time > 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

    prev_frame_time = new_frame_time

    cv2.imshow("Anti-Spoofing Detection", img)

    # Press 'q' to quit gracefully
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()