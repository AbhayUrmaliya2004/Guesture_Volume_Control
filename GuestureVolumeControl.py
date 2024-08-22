import cv2 
import numpy as np
import time
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL 
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

prevTime = 0

vol = 0
volBar = 400
while True: 
    success, img = cap.read()

#    img = cv2.resize(img, (640, 480))  # Reduce image size for faster processing
    img = detector.findHands(img)
    lmList = detector.findPositions(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip
        
        cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        # Map the length to the volume range
        # Need to fix the pct and volPct and volBar 
        vol = np.interp(length, [15, 120], [minVol, maxVol])
        volBar = np.interp(length, [15, 120], [400, 150])
        volPct = np.interp(length, [15, 120], [0, 100])

        # Check if the length is valid and set the volume
        if length is not None:
            volume.SetMasterVolumeLevel(vol, None)
        print(length)

        # volume bar 
        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, f"{int(volPct)}%", (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)

        if length < 20:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 