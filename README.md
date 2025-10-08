# 🌊 AI-Powered Oil Spill Detection and Segmentation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)

This project is a complete end-to-end deep learning solution for detecting and segmenting oil spills in satellite imagery. It uses a U-Net model built with PyTorch to perform semantic segmentation, classifying each pixel as either "oil spill" or "no spill". The trained model is deployed in a real-time, interactive web application using Streamlit.

## ✨ Key Features

- **Accurate Segmentation:** Utilizes a U-Net architecture, achieving a high Intersection over Union (IoU) score of **0.88** on the test set.
- **Binary Classification:** Classifies images as either "Oil Spill Detected" or "No Spill Detected" based on the segmentation output.
- **Interactive Web App:** A user-friendly Streamlit application allows users to upload their own images and receive instant predictions.
- **Visual Overlay:** The predicted spill mask is displayed as a semi-transparent, colored overlay on the original image for clear and intuitive visualization.
- **Portable Model:** The final model is exported to the ONNX format, making it ready for cross-platform deployment (e.g., in the Unity engine).

## 🚀 Live Demo

Live - (https://abhioilspill.streamlit.app/)

## 📸 Screenshots

Here is the model in action, correctly identifying a spill and clearing an image with no spill.

#### Positive Detection (Oil Spill)

![Oil Spill Detected](https://github.com/abhiramavarma/AI_SpillGuard_OSD-AbhiramaVarma_Nandayala/blob/main/images/oil_spill.png)
*The model correctly identifies the oil spill and overlays the segmented area in red.*

#### Negative Detection (No Spill)

![No Spill Detected](https://github.com/abhiramavarma/AI_SpillGuard_OSD-AbhiramaVarma_Nandayala/blob/main/images/No_Oill_Spill.png)
*The model correctly classifies this image as having no spill, ignoring challenging features like sun glint.*

## 💻 Technologies Used

- **AI & Deep Learning:** Python, PyTorch, U-Net, CNNs
- **Data & Image Processing:** OpenCV, Albumentations, NumPy, Pillow
- **Deployment & Web:** Streamlit, Streamlit Community Cloud
- **Development Tools:** Git & GitHub, Google Colab

## 🛠️ Setup and Local Installation

To run this project on your local machine, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

