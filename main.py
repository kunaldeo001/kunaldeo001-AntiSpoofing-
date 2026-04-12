from ultralytics import YOLO
import cv2
import cvzone
import math
import time

confidence = 0.6

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(3, 640)
cap.set(4, 480)

model = YOLO("yolov8n.pt")

classNames = ["fake", "real"]

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
            if conf > confidence and cls < len(classNames):
                if classNames[cls] == 'real':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(
                    img,
                    f'{classNames[cls].upper()} {int(conf * 100)}%',
                    (max(0, x1), max(35, y1)),
                    scale=2, thickness=4, colorR=color, colorB=color
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