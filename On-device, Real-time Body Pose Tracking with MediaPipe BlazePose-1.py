import os
# Disable TensorFlow oneDNN optimizations for reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open video source.")
    exit()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Warning: Empty frame.")
            continue

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Real-time Body Pose Tracking with BlazePose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
