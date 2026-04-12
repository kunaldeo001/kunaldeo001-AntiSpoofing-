from time import time

import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

classId = 0  # Real = 1, Fake = 0
output = 'Dataset/Datacollect'
confidence = 0.8
save = True
BlurThreshold = 35

debug = False
offsetWPercent = 10
offsetHPercent = 20
camWidth, camHeight = 640, 480
floatingPoint = 6

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(3, camWidth)
cap.set(4, camHeight)

import os
os.makedirs(output, exist_ok=True)

detector = FaceDetector()

while True:
    success, img = cap.read()
    if not success:
        print("Warning: Failed to read frame.")
        break

    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []
    listInfo = []

    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]

            if score > confidence:
                offsetW = int((offsetWPercent / 100) * w)
                offsetH = int((offsetHPercent / 100) * h)

                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)

                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                imgFace = img[y:y + h, x:x + w]
                if imgFace.size == 0:
                    continue

                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

                listBlur.append(blurValue > BlurThreshold)

                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2
                xcn = round(min(xc / iw, 1.0), floatingPoint)
                ycn = round(min(yc / ih, 1.0), floatingPoint)
                wn = round(min(w / iw, 1.0), floatingPoint)
                hn = round(min(h / ih, 1.0), floatingPoint)

                listInfo.append(f"{classId} {xcn} {ycn} {wn} {hn}\n")

                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(
                    imgOut,
                    f'Score: {int(score * 100)}% Blur: {blurValue}',
                    (x, max(y - 10, 20)),
                    scale=2, thickness=3
                )

        if save:
            if all(listBlur) and listBlur:
                timeNow = str(time()).replace('.', '')
                img_path = f"{output}/{timeNow}.jpg"
                label_path = f"{output}/{timeNow}.txt"

                cv2.imwrite(img_path, img)

                # Use context manager to prevent file handle leaks
                with open(label_path, 'a') as f:
                    for info in listInfo:
                        f.write(info)

    cv2.imshow("Data Collection", imgOut)

    # Press 'q' to quit gracefully
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
