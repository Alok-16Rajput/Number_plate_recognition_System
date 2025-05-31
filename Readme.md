# Number Plate Recognition System

# Overview

# his project is a Number Plate Recognition System that utilizes OpenCV, Keras, and TensorFlow to detect, extract, and recognize vehicle license plates. The system first detects the number plate using Haar Cascade Classifier, processes the plate image, and then classifies individual characters using a Convolutional Neural Network (CNN).

# Technologies Used
## Python
## OpenCV (for image processing and plate detection)
## Keras & TensorFlow (for building and training the CNN model)
## NumPy & Pandas (for data handling)
## Matplotlib (for visualization)
## scikit-learn (for model evaluation)

# Dataset
## The dataset used for training the Number Plate Recognition System consists of individual character images.

### 1 Character Dataset:
#### Contains individual characters (A-Z, 0-9) extracted from number plates.
#### Each character is stored as a grayscale image of size 28x28 pixels.
#### Used for training the CNN model to recognize letters and digits.

# Project Structure
📦 Number-Plate-Recognition-System
├── 📁 data
│   ├── 📁 train          # Training dataset
│   └── 📁 val            # Validation dataset
├── 📁 models             # Contains trained models
├── 📁 haarcascades       # Haar cascade files for plate detection
├── 📄 Number_Plate_Recognition.ipynb   # Main detection script
├── 📄 trained_model.h5   # Saved trained model
├── 📄 test.ipynb         # Testing script
└── 📄 README.md          # Project documentation 

# Workflow
## 1.Load Image & Preprocess

### Read the image using OpenCV.
### Convert it to grayscale.
### Apply Haar Cascade to detect the number plate.
### Crop and preprocess the detected plate.

# 2.Character Segmentation

### Apply thresholding, erosion, and dilation to clean the image.
### Find contours of characters.
### Extract individual character images.

# 3.Character Recognition

### Resize characters to 28x28.
### Feed them into a trained CNN model.
### Predict and decode the characters into a readable format.

# 4.Model Training

### CNN architecture with Conv2D, MaxPooling, Dropout, and Dense layers.
### Uses categorical cross-entropy loss and Adam optimizer.
### Trained using ImageDataGenerator for data augmentation.

# 5.Evaluation & Performance Metrics

### Evaluate accuracy using a validation dataset.
### Display confusion matrix and classification report.

# Flowchart

    A[Input Image] -->|Convert to Grayscale| B[Preprocessing]
    B -->|Apply Haar Cascade| C[Detect Number Plate]
    C -->|Crop & Process Image| D[Thresholding & Filtering]
    D -->|Find Contours| E[Character Segmentation]
    E -->|Resize & Normalize| F[CNN Model Prediction]
    F -->|Decode Characters| G[Extracted Plate Number]
    G -->|Display Output| H[Final Recognized Number]

# How to Run

## 1.Clone the repository:
## git clone https://github.com/Alok-16Rajput/Number_plate_recognition_system_project.git
## cd Number-Plate-Recognition-System

## 2.Install dependencies:
## pip install -r requirements.txt

## 3.Run the code:
### python Number_plate_recognition.ipynb