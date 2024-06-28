import cv2
import torch
import numpy as np
import pyautogui
import subprocess

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Video capture object
camera = cv2.VideoCapture(0)

# Define confidence threshold
confidence_threshold = 0.5

# Define the path to ffmpeg executable
ffmpeg_path = 'C:/Users/ACER/anaconda3/pkgs/ffmpeg-4.3.1-ha925a31_0/Library/bin/ffmpeg.exe'

# Function to start screen recording
def start_screen_recording(output_path):
    # Define the screen recording parameters
    screen_width, screen_height = pyautogui.size()
    codec = 'libx264'
    fps = 30.0

    # Define the ffmpeg command to start screen recording
    command = [
        ffmpeg_path,
        '-f', 'gdigrab',
        '-framerate', str(fps),
        '-video_size', f'{screen_width}x{screen_height}',
        '-i', 'desktop',
        '-codec:v', codec,
        '-pix_fmt', 'yuv420p',  # Add this line to specify pixel format
        output_path
    ]

    # Start the screen recording process
    recording_process = subprocess.Popen(command)

    # Optionally return the process object if you want to manage it later
    return recording_process

# Function to stop screen recording
def stop_screen_recording(recording_process):
    # Terminate the recording process
    recording_process.terminate()
    print("Screen recording stopped")

# Function to take screenshot
def take_screenshot():
    pyautogui.screenshot("D:/Drone Screenshot/screenshot.png")
    print("Screenshot taken")

# Function to draw buttons on the frame
def draw_buttons(frame, recording):
    cv2.putText(frame, "Start Recording (Press 'r')", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Take Screenshot (Press 's')", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if recording:
        cv2.putText(frame, "Stop Recording & Save (Press 't')", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

recording_process = None
recording = False  # State variable to track recording status

while True:
    ret, frame = camera.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Perform inference with YOLOv5 on the frame
    results = model(frame)

    # Process the detection results
    detections = results.xyxy[0]

    # Filter detections for 'person' class only
    # Filter detections for 'car' class only (assuming 'car' is class index 2)
    car_detections = detections[detections[:, 5] == 2]

    # Draw rectangle boxes around detected cars with blue color
    for detection in car_detections:
        x1, y1, x2, y2, conf, _ = detection  # Extract coordinates and confidence
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert coordinates to integers
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle with blue color

    # Filter detections for 'person' class only (assuming 'person' is class index 0)
    person_detections = detections[detections[:, 5] == 0]

    # Draw rectangle boxes around detected persons with red color (already in the code)
    for detection in person_detections:
        x1, y1, x2, y2, conf, _ = detection  # Extract coordinates and confidence
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert coordinates to integers
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw rectangle with red color

    # Count the number of detected persons
    count = len(person_detections)

    # Display the count on the screen
    cv2.putText(frame, f'Persons: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw buttons on the frame
    frame = draw_buttons(frame, recording)

    # Display the frame
    cv2.imshow('Camera', frame)

    # Check for key press events
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        output_path = "D:/Drone Recording/recording.mp4"
        recording_process = start_screen_recording(output_path)
        recording = True  # Set recording status to True
        print("Screen recording started")
    elif key == ord('s'):
        take_screenshot()
    elif key == ord('t') and recording:
        stop_screen_recording(recording_process)
        recording = False  # Set recording status to False
        break  # Exit the loop after stopping recording and saving

# Release the camera resource when done
camera.release()
cv2.destroyAllWindows()

remove car detection from this and make seprate code for car detection
