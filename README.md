# Plant-disease-Detection
Plant disease detection using a Convolutional Neural Network (CNN) built with Keras/TensorFlow. Identifies diseases from leaf images.
 Table of Contents

* Project Overview.
* Features
* Dataset
* Model Architecture
* Technologies Used
* Setup & Installation
* How to Use
* Results
* Contributing
* License

Project Overview

This project aims to help farmers and agricultural experts in identifying plant diseases early by leveraging computer vision and deep learning. By simply taking a picture of a plant leaf, the model can predict the type of disease (or if the leaf is healthy), allowing for timely intervention and treatment.
The core of this project is a **Convolutional Neural Network (CNN)**, a class of deep neural networks well-suited for image classification tasks.

project Features
*High Accuracy:** Achieves high precision in classifying multiple plant diseases.
*Scalable Architecture:** Built using TensorFlow 2.x and Keras for robust and scalable model training.
*Simple Web Interface:** (Optional - *if you build one*) Includes a simple Streamlit/Flask app to upload an image and get an instant prediction.
*Jupyter Notebooks:** Detailed notebooks for data preprocessing, model training, and evaluation.

Tech Stack

*Backend: Python
*Deep Learning: TensorFlow, Keras
* Data Handling: Pandas, NumPy
* Image Processing: OpenCV, PIL
* Web App (Optional): Streamlit or Flask

dataset-icon Dataset

This project uses the **PlantVillage Dataset**, which contains over 54,000 images of plant leaves.
* Source: [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plant-disease-dataset-full) (or link to your specific source)
* Content: 38 classes of healthy and diseased plant leaves.
* Classes Include:
    * Apple (Scab, Black Rot, Healthy)
    * Tomato (Bacterial Spot, Early Blight, Healthy)
 
Model Architecture

The core of this project is a **Convolutional Neural Network (CNN). The architecture consists of:

1.  Convolutional Layers (Conv2D): To extract features (like edges, textures, and patterns) from the images.
2.  Activation Function (ReLU): To introduce non-linearity.
3.  Pooling Layers (MaxPooling2D): To reduce the spatial dimensions and computational load.
4.  Dropout Layers: To prevent overfitting by randomly "dropping" neurons during training.
5.  Flatten Layer: To convert the 2D feature maps into a 1D vector.
6.  Dense Layers (Fully Connected):To perform the final classification.
7.  Output Layer: A `Softmax` activation function to output probabilities for each of the 38 classes.
