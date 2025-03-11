import cv2
import time
import os
import requests
import threading
import tkinter as tk
from tkinter import messagebox
from threading import Thread
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# ตั้งค่า LINE Notify
LINE_NOTIFY_TOKEN = "wDWc4yjjTH90dosYSXKva3tfhzW5Qz38NHgGHgt8HfL"
LINE_NOTIFY_URL = "https://notify-api.line.me/api/notify"



# Function to send LINE Notify message with image
def send_line_notify(message, image_path=None):
    headers = {"Authorization": f"Bearer {LINE_NOTIFY_TOKEN}"}
    data = {"message": message}

    if image_path:
        files = {"imageFile": open(image_path, "rb")}
        requests.post(LINE_NOTIFY_URL, headers=headers, data=data, files=files)
        files["imageFile"].close()
    else:
        requests.post(LINE_NOTIFY_URL, headers=headers, data=data)

# Function to show pop-up alert and stop further notifications
def show_popup(message, stop_alert_event):
    root = tk.Tk()
    root.withdraw()
    response = messagebox.askyesno("🔥 Fire Alert!", message)

    if response:
        print("User confirmed fire alert!")
        stop_alert_event.set()  # Stop further alerts
    else:
        print("User ignored fire alert!")
    root.destroy()

# Load trained YOLO model
model = YOLO(r"runs/detect/train/weights/best.pt")

# Open video file
video_path = "test_video/2110972-uhd_3840_2160_30fps.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Failed to load video!")
    exit()

fire_detected = False
last_alert_time = 0
fire_detected_time = 0
last_fire_location = None

# Create an event to stop alerts
stop_alert_event = threading.Event()

# Get frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit when the video ends

    # Run YOLO fire detection
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = result.names[int(box.cls[0])]

            # If fire is detected with confidence > 50%
            if "fire" in label.lower() and conf > 0.50:
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                box_size = (x2 - x1) * (y2 - y1)  # Calculate bounding box size

                # **Determine horizontal position**
                if center_x < frame_width * 0.33:
                    horizontal_pos = "Left side"
                elif center_x > frame_width * 0.66:
                    horizontal_pos = "Right side"
                else:
                    horizontal_pos = "Center"

                # **Determine vertical position**
                if center_y < frame_height * 0.33:
                    vertical_pos = "Top"
                elif center_y > frame_height * 0.66:
                    vertical_pos = "Bottom"
                else:
                    vertical_pos = "Middle"

                # **Determine depth (Near-Far)**
                if box_size > (frame_width * frame_height * 0.05):  # If bounding box is large
                    depth_pos = "Near the camera"
                elif box_size < (frame_width * frame_height * 0.02):  # If bounding box is small
                    depth_pos = "Far from the camera"
                else:
                    depth_pos = "Mid-range"

                fire_location = f"{horizontal_pos}, {vertical_pos}, {depth_pos}"
                print(f"🚨 Fire detected at: {fire_location}")

                # Draw bounding box and display position
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{fire_location}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Save the frame with fire detection
                img_filename = "fire_detected_frame.jpg"
                cv2.imwrite(img_filename, frame)

                # Check if fire has been detected in the same location for more than 30 seconds
                current_time = time.time()
                if fire_location != last_fire_location:
                    fire_detected_time = current_time  # Reset timer when fire is detected in a new location
                    last_fire_location = fire_location

                if current_time - fire_detected_time < 30:  # If fire has been detected for less than 30 seconds
                    if current_time - last_alert_time > 10:  # Alert every 10 seconds
                        if not stop_alert_event.is_set():  # Only alert if stop signal is not set
                            # Send alerts to LINE Notify with the image attached
                            alert_message = f"🔥 Fire detected at: {fire_location}"

                            # Show pop-up alert in a separate thread
                            Thread(target=show_popup, args=(alert_message, stop_alert_event)).start()

                            # Send alerts to LINE Notify with the image attached
                            Thread(target=send_line_notify, args=(alert_message, img_filename)).start()

                            last_alert_time = current_time

                else:
                    # If fire has been detected for more than 30 seconds, stop sending alerts
                    print("Fire alert stopped for this location.")

    # Resize frame for display
    frame_resized = cv2.resize(frame, (1280, 720))
    cv2.imshow("🔥 Fire Detection System", frame_resized)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Clean up by deleting the saved image file
if os.path.exists(img_filename):
    os.remove(img_filename)
