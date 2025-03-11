import cv2
import time
import os
import requests
import threading
import tkinter as tk
from tkinter import messagebox
from threading import Thread
from ultralytics import YOLO

# ตั้งค่า LINE Notify
LINE_NOTIFY_TOKEN = "wDWc4yjjTH90dosYSXKva3tfhzW5Qz38NHgGHgt8HfL"
LINE_NOTIFY_URL = "https://notify-api.line.me/api/notify"

# ฟังก์ชันส่งข้อความไปยัง LINE Notify
def send_line_notify(message, image_path=None):
    headers = {"Authorization": f"Bearer {LINE_NOTIFY_TOKEN}"}
    data = {"message": message}

    if image_path:
        files = {"imageFile": open(image_path, "rb")}
        requests.post(LINE_NOTIFY_URL, headers=headers, data=data, files=files)
        files["imageFile"].close()
    else:
        requests.post(LINE_NOTIFY_URL, headers=headers, data=data)

# ฟังก์ชันแสดง pop-up แจ้งเตือน
def show_popup(message, stop_alert_event):
    root = tk.Tk()
    root.withdraw()
    response = messagebox.askyesno("🔥 Fire Alert!", message)

    if response:
        print("User confirmed fire alert!")
        stop_alert_event.set()  # หยุดการแจ้งเตือนเพิ่มเติม
    else:
        print("User ignored fire alert!")
    root.destroy()

# โหลดโมเดล YOLO
model = YOLO(r"runs/detect/train2/weights/best.pt")

# เปิดวิดีโอไฟล์
video_path = "test_video/2110972-uhd_3840_2160_30fps.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Failed to load video!")
    exit()

# สร้างตัวแปรเก็บสถานะ
fire_detected = False
last_alert_time = 0
fire_detected_time = 0
last_fire_location = None
stop_alert_event = threading.Event()

# ขนาดเฟรมวิดีโอ
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ค่าทางสถิติสำหรับ Precision และ Recall
true_positives = 0
false_positives = 0
false_negatives = 0

# ฟังก์ชันคำนวณ IoU (Intersection over Union)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # จบเมื่อวิดีโอสิ้นสุด

    # ใช้ YOLO ตรวจจับไฟ
    results = model(frame)
    detected_fire = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = result.names[int(box.cls[0])]

            # ตรวจจับวัตถุ "fire" ด้วยความมั่นใจ > 50%
            if "fire" in label.lower() and conf > 0.50:
                detected_fire.append((x1, y1, x2, y2))

    # เปรียบเทียบกับค่าความจริง (Ground Truth)
    ground_truth_fire = []  # ต้องใส่ข้อมูล bounding box ของไฟจริง ๆ

    # คำนวณ True Positive (TP) และ False Negative (FN)
    for gt_box in ground_truth_fire:
        matched = False
        for detected_box in detected_fire:
            iou = calculate_iou(gt_box, detected_box)
            if iou > 0.5:
                true_positives += 1
                matched = True
                break
        if not matched:
            false_negatives += 1

    # คำนวณ False Positive (FP)
    for detected_box in detected_fire:
        matched = False
        for gt_box in ground_truth_fire:
            iou = calculate_iou(gt_box, detected_box)
            if iou > 0.5:
                matched = True
                break
        if not matched:
            false_positives += 1

    # คำนวณ Precision และ Recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # แสดงค่า Precision และ Recall บนหน้าจอ
    cv2.putText(frame, f"Precision: {precision:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Recall: {recall:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # แสดงผลลัพธ์วิดีโอ
    frame_resized = cv2.resize(frame, (1280, 720))
    cv2.imshow("🔥 Fire Detection System", frame_resized)

    # บันทึกเฟรมที่ตรวจพบไฟ
    if detected_fire:
        img_filename = "fire_detected_frame.jpg"
        cv2.imwrite(img_filename, frame)

        # แจ้งเตือนผ่าน LINE Notify ทุก 10 วินาที
        current_time = time.time()
        if current_time - last_alert_time > 10 and not stop_alert_event.is_set():
            alert_message = "🔥 Fire detected! Precision: {:.2f}, Recall: {:.2f}".format(precision, recall)

            # แสดง pop-up แจ้งเตือน
            Thread(target=show_popup, args=(alert_message, stop_alert_event)).start()

            # ส่ง LINE Notify พร้อมรูปภาพ
            Thread(target=send_line_notify, args=(alert_message, img_filename)).start()

            last_alert_time = current_time

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ปิดโปรแกรม
cap.release()
cv2.destroyAllWindows()

# ลบรูปภาพที่สร้างขึ้น
if os.path.exists(img_filename):
    os.remove(img_filename)
