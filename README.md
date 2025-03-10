# Fire Detection System using YOLOv8

## Overview
This project focuses on developing a fire detection and classification system using YOLOv8, a high-accuracy deep learning model. The system aims to enhance real-time fire detection capabilities while minimizing false negatives. By leveraging object detection and classification techniques, this system can efficiently identify fire incidents in various indoor environments and send real-time alerts via Line Notify or smart home integrations.

## Features
- Real-time fire detection using YOLOv8
- Fine-tuned model for high accuracy and reduced false negatives
- Supports CCTV integration for continuous monitoring
- Instant alert notifications via Line Notify
- Optimized for various lighting conditions and obstructions

## Dataset
The model has been trained using the **Fire Detection** dataset from Roboflow: [Fire Detection Dataset](https://universe.roboflow.com/dave-absmb/fire-detection-ijcxg-tyrsm/dataset/7). The dataset includes images and videos under different conditions, ensuring robustness across various scenarios. The dataset is stored in the `fire_dataset` folder.

## Project Structure
```
|-- fire_dataset/      # Training dataset
|-- runs/detect/       # Detection results
|-- test_video/        # Test videos for validation
|-- README.md          # Project documentation
|-- alarm.ipynb        # Notebook for alert system integration
|-- data.yaml          # Dataset configuration file
|-- train_model.ipynb  # Model training script
|-- yolo11n.pt         # Trained YOLOv11 model (if applicable)
|-- yolov8n.pt         # Trained YOLOv8 model
```

## Installation
### Prerequisites
- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Line Notify API (for alert notifications)

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/fire-detection.git
   cd fire-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the trained YOLOv8 model and place it in the project directory.

### Alert System
For real-time notifications using Line Notify, use `alarm.ipynb` to configure API integration.

## Contributors
- **PitchanaChk** 
- **PattamapornKaru** 


