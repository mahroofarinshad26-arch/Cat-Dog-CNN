#  Cat vs Dog Image Classification using CNN

## Project Overview

This project is a Deep Learning image classification project that uses a Convolutional Neural Network (CNN) to classify images into two categories:

-  Cat
-  Dog

The trained CNN model is integrated with a Streamlit web application, allowing users to upload an image and get a prediction instantly.

---

##  Objective

The main objective of this project is to build a CNN-based image classification model that can identify whether an uploaded image is a cat or a dog.

The trained model is deployed using Streamlit so that users can interact with the model through a simple web interface.

---

##  Dataset

The model was trained using a Cat and Dog image dataset containing images belonging to two classes:

- Cat
- Dog

The images were preprocessed and resized before being given to the CNN model.

---

##  Model Used

### Convolutional Neural Network (CNN)

CNN is a Deep Learning algorithm commonly used for image classification.

The CNN learns important visual features such as:

- Edges
- Shapes
- Textures
- Patterns

These features help the model distinguish between cats and dogs.

---

##  Project Workflow

```text
Dataset
   ↓
Image Preprocessing
   ↓
Image Resizing
   ↓
CNN Model
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Save Trained Model
   ↓
Streamlit Application
   ↓
Upload Image
   ↓
Cat / Dog Prediction

-----

## Technologies Used

Python
TensorFlow
Keras
NumPy
PIL
Streamlit
Git
GitHub

----

## Image Preprocessing
Before training the model, the images were preprocessed.
Main preprocessing steps include:
Loading images
Resizing images to a fixed size
Converting images into numerical arrays
Normalizing pixel values
Preparing images for CNN training

-----


## Streamlit Application

A Streamlit web application was created to make the trained model easy to use.
Users can:
Open the web application
Upload a cat or dog image
The image is processed
The trained CNN model predicts the class
The prediction is displayed on the screen

-----

## Conclusion

This project demonstrates the complete workflow of an image classification application using Deep Learning.
A CNN model was trained for Cat vs Dog classification and integrated with Streamlit for real-time image prediction. The project also demonstrates model deployment using GitHub and Streamlit Community Cloud.
