# Handmotion-Volume-Changer
Using just your hands and your webcam you can change the volume of your computer!


Hand Gesture Volume Control
Overview
This project utilizes computer vision and hand tracking to control system volume through hand gestures. It leverages the MediaPipe library for hand tracking and the pycaw library for audio control. The program captures webcam feed, detects hand landmarks, and adjusts the system volume based on the distance between the thumb and index finger.

**Features**
- Hand Tracking: Utilizes the MediaPipe library to detect and track hand landmarks in real-time from the webcam feed.

- Gesture Recognition: Recognizes a clapping gesture by monitoring the distance between the thumb and index finger, triggering a volume control action.

- Volume Control: Adjusts system volume dynamically based on the hand gesture. The closer the fingers, the lower the volume, and vice versa.

- User Feedback: Displays visual feedback on the screen, including hand landmarks, hand connection lines, and a volume control bar.

**Dependencies**
- OpenCV: Used for capturing webcam feed and image processing.
- MediaPipe: Employs the Hand module for hand tracking.
- pycaw: Interfaces with the Windows Audio API to control system volume.
  
**Usage**
1. Install the required libraries: pip install mediapipe opencv-python pycaw.

2. Run the script: python hand_gesture_volume_control.py.

3. Place your hand in front of the webcam, and adjust the volume by bringing your thumb and index finger closer or farther apart.

4. Optionally, perform a clapping gesture to trigger volume changes.

**Configuration**
- Sensitivity: Adjust the SPEED_FACTOR constant in the code to control the volume sensitivity.

- Deadzone: Modify the DEADZONE_WIDTH constant to set the width of the deadzone on the right side of the frame, excluding it from volume control.

Notes
- Ensure that your system has a webcam and the required dependencies installed.

- Tested on Windows operating systems.
