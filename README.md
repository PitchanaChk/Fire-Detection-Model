### README.md  

# Fire Detection System using YOLOv8  

## Overview  
This project is a fire detection system using **YOLOv8**, designed for real-time fire detection in various environments. The system can integrate with **CCTV** for continuous monitoring and send alerts via **Line Notify** when fire is detected.  

## Features  
- Real-time fire detection using **YOLOv8**  
- Fine-tuned model for **high accuracy** and **low false alarms**  
- Supports **CCTV integration**  
- **Instant notifications** via **Line Notify**  
- **Robust detection** under different lighting conditions  

## Dataset  
The model is trained on the **Fire Detection** dataset from Roboflow:  
👉 [Fire Detection Dataset](https://universe.roboflow.com/dave-absmb/fire-detection-ijcxg-tyrsm/dataset/7)  

The dataset is organized as follows:  
```
fire_dataset/
│-- train/     # Training images
│-- test/      # Testing images
│-- valid/     # Validation images
```

## Project Structure  
```
|-- fire_dataset/      # Training dataset
|-- runs/detect/       # Detection results
|-- test_video/        # Test videos 
|-- README.md          # Project documentation
|-- data.yaml          # Dataset configuration file
|-- train_model.ipynb  # Model training script
|-- test_alarm.py      # Fire detection test script
|-- requirements.txt   # Required dependencies
|-- yolo11n.pt         # Trained YOLOv11 model (if applicable)
|-- yolov8n.pt         # Trained YOLOv8 model
```

---

## Installation Guide  

### **1. Download & Extract the ZIP File**  
1. Download the project ZIP file and extract it to your preferred directory.  

```bash
unzip fire-detection-model.zip
cd fire-detection-model
```

### **2. Create a Virtual Environment (Optional but Recommended)**  
To prevent dependency conflicts, create a virtual environment:  

```bash
python -m venv fire_env
source fire_env/bin/activate  # macOS/Linux
fire_env\Scripts\activate     # Windows
```

### **3. Install Dependencies**  
Run the following command to install all required libraries:  

```bash
pip install -r requirements.txt
```

### **4. Download & Place the Trained Model**  
Ensure that `yolov8n.pt` (or `yolo11n.pt`) is in the root directory of the project. If missing, download it from the YOLOv8 repository:  

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

---

## Running the System  

### **1. Train the Model (Optional)**  
If you want to train the model from scratch, open **`train_model.ipynb`** in Jupyter Notebook:  

```bash
jupyter notebook train_model.ipynb
```

Modify `data.yaml` if necessary to match your dataset path, then run all cells.

---

### **2. Run Fire Detection on a Test Video**  
To test fire detection on sample videos, run:  

```bash
python test_alarm.py --model yolov8n.pt --video test_video/fire_test1.mp4
```

This will process the video and save the output in the `runs/detect/` folder.

---

### **3. Enable Fire Alerts via Line Notify**  
To receive alerts when fire is detected:  

1. Open **`alarm.ipynb`**  
2. Follow the instructions to set up **Line Notify API**  
3. Run the notebook to start receiving notifications.

---

## Contributors  
- **PitchanaChk**  
- **PattamapornKaru**  
