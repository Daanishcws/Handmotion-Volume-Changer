########
# Author: Daanish Ali
# Date: 11/12/23
# Title: Handmotion volume changer
########




import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# solution APIs
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Volume Control Library Usage
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol, volBar, volPer = volRange[0], volRange[1], 400, 0

# Constants for volume control
INITIAL_HAND_DISTANCE = None
SPEED_FACTOR = 1000  # Adjust this factor to control volume sensitivity (increase for more sensitivity)
MIN_DISTANCE = 20    # Minimum distance between thumb and index finger for volume control
DEADZONE_WIDTH = 200  # Width of the deadzone on the right side

# Initialize close_hands_count
close_hands_count = 0

# Webcam Setup
wCam, hCam = 640, 480
# Try different backends if MSMF has issues (e.g., cv2.CAP_DSHOW)
cam = cv2.VideoCapture(0, cv2.CAP_MSMF)
cam.set(3, wCam)
cam.set(4, hCam)

# Mediapipe Hand Landmark Model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:
        ret, image = cam.read()
        
        if not ret:
            continue  # Skip this frame and continue with the next one
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Gesture recognition: Check if hands are close together (clapping)
                landmarks = hand_landmarks.landmark
                x1, y1, z1 = landmarks[4].x, landmarks[4].y, landmarks[4].z
                x2, y2, z2 = landmarks[8].x, landmarks[8].y, landmarks[8].z
                hand_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

                if hand_distance < MIN_DISTANCE:
                    close_hands_count += 1
                else:
                    close_hands_count = 0

        # Volume control logic...
        if results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0:
            myHand = results.multi_hand_landmarks[0]
            lmList = []
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]

                # Exclude the right side of the frame (deadzone) from volume control
                if x1 < wCam - DEADZONE_WIDTH:
                    # Marking Thumb and Index finger
                    cv2.circle(image, (x1, y1), 15, (255, 255, 255))
                    cv2.circle(image, (x2, y2), 15, (255, 255, 255))
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    hand_distance = math.hypot(x2 - x1, y2 - y1)

                    if INITIAL_HAND_DISTANCE is None:
                        INITIAL_HAND_DISTANCE = hand_distance

                    # Calculate the change in hand distance and adjust the volume
                    delta_distance = hand_distance - INITIAL_HAND_DISTANCE
                    volume_delta = delta_distance / SPEED_FACTOR
                    new_volume = volume.GetMasterVolumeLevelScalar() + volume_delta

                    # Clip the volume to the valid range
                    new_volume = max(min(new_volume, 1.0), 0.0)

                    volume.SetMasterVolumeLevelScalar(new_volume, None)
                    volBar = np.interp(new_volume, [0.0, 1.0], [400, 150])
                    volPer = int(new_volume * 100)

                    # Volume Bar
                    cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
                    cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
                    cv2.putText(image, f'{volPer} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                                1, (0, 0, 0), 3)

        cv2.imshow('handDetector', image)
        key = cv2.waitKey(1)
        
        if key == 27:  # Press 'Esc' key to exit the program
            break
        
        if cv2.waitKey(1) == ord('f') and cv2.waitKey(1) == ord('b'):  # Press 'F' and 'B' keys together to exit
            break

cam.release()
cv2.destroyAllWindows()

