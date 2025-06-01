# Number Plate Recognition System 🚗
---
## 📌 Overview
---
 This project presents an automated Number Plate Recognition (NPR) system leveraging OpenCV, Keras, and TensorFlow to accurately detect, extract, and recognize vehicle license plates. The system employs a Haar Cascade Classifier for initial plate detection, followed by image preprocessing and segmentation. A Convolutional Neural Network (CNN) is then utilized to classify and recognize individual alphanumeric characters on the plate.
 
---
## 🛠 Technologies Used
- **Python**
- **OpenCV (for image processing and plate detection)**
- **Keras & TensorFlow (for building and training the CNN model)**
- **NumPy & Pandas (for data handling)**
- **Matplotlib (for visualization)**
- **scikit-learn (for model evaluation)**
---

## 📊 Dataset
The dataset used to train the Number Plate Recognition System comprises individual character images extracted from license plates, enabling accurate classification of alphanumeric characters.

## 1 Character Dataset:
- **Consists of individual alphanumeric characters (A–Z, 0–9) extracted from vehicle license plates.**
- **Each character is represented as a 28×28 pixel grayscale image.**
- **Primarily used to train the Convolutional Neural Network (CNN) for accurate recognition of letters and digits.**

## 📂 Project Structure
'''bash
📦 Number-Plate-Recognition-System
├── 📁 data
│   ├── 📁 train          
│   └── 📁 val            
├── 📁 models             
├── 📁 haarcascades       
├── 📄 Number_Plate_Recognition.ipynb   
├── 📄 trained_model.h5  
├── 📄 test.ipynb         
└── 📄 README.md           
'''

---
# 🔄 Workflow
## 1.Load Image & Preprocess

- **Read the image using OpenCV.**
- **Convert it to grayscale.**
- **Apply Haar Cascade to detect the number plate.**
- **Crop and preprocess the detected plate.**

## 2.Character Segmentation

- **Apply thresholding, erosion, and dilation to clean the image.**
- **Find contours of characters.**
- **Extract individual character images.**

## 3.Character Recognition

#### >Resize characters to 28x28.
- **Feed them into a trained CNN model.**
- **Predict and decode the characters into a readable format.**

## 4.Model Training

- **Uses categorical cross-entropy loss and Adam optimizer.**
- **Trained using ImageDataGenerator for data augmentation.**

## 5.Evaluation & Performance Metrics

- **Evaluate accuracy using a validation dataset.**
- **Display confusion matrix and classification report.**
---
## 🚀 How to Run

### 1.Clone the repository:

 - **git clone 'https://github.com/Alok-16Rajput/Number_plate_recognition_System.git'**
 - **cd Number-Plate-Recognition-System**



## 2.Install dependencies:

 pip install -r requirements.txt

## 3.Run the code:

python Number_plate_recognition.ipynb
---

## 📈 Result
- **Achieved high recognition accuracy using the CNN-based classification model.**
- **Successfully detects and recognizes Indian vehicle license plates in real-world scenarios.**
- **Demonstrates robustness to varying lighting conditions and viewing angles.**

- **The model achieved an accuracy of 97%. Future enhancements may include expanding the training dataset, refining the character segmentation process, and optimizing the model architecture to improve generalization and overall performance.**
---

## 🏆 Future Improvements
- **Improve Optical Character Recognition (OCR) accuracy by incorporating advanced deep learning models such as CRNN or transformer-based architectures.**
- **Integrate real-time detection capabilities using live camera feeds for on-the-fly number plate recognition.**
- **Extend support for multilingual license plates and diverse number plate formats to enhance system versatility across regions.**
---

## 🏆 Future Uses and Applications
-  **1.Real-time Traffic Management:** Integration with live traffic surveillance systems to automatically detect and analyze vehicles. This can assist authorities in monitoring traffic flow, identifying violations, issuing fines, and tracking stolen vehicles in real time.

- **2.Parking Lot Management:** Deployment in parking facilities to automate vehicle entry and exit tracking, record timestamps, calculate parking fees, and detect unauthorized or blacklisted vehicles.

- **3.Toll Collection Systems:** Implementation in toll booth systems to recognize vehicle number plates for seamless, contactless toll collection. The system can automatically deduct fees from linked user accounts, improving efficiency and reducing congestion.
---

## 📞 contact
📧 **Email:** alok1602.kumar@gmail.com


