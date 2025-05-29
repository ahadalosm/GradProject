" Diabetic Retinopathy Detection using AI "

This is my graduation project, where I developed an AI-based system for the early and accurate detection of diabetic retinopathy using deep learning techniques.

" Project Overview "
Traditional diagnostic methods rely on time-consuming manual examinations by specialists. In this project, I proposed an automated solution based on Convolutional Neural Networks (CNNs) to identify diabetic retinopathy stages through retinal fundus images.

The project utilizes the public IDRiD dataset and applies multiple deep learning architectures via transfer learning to detect and grade diabetic retinopathy. The goal was to enhance diagnostic efficiency, reduce the burden on healthcare systems, and improve patient outcomes.

" Project Structure "

data_exploration.ipynb: Data cleaning, handling missing values, and data visualization (bar chart, pie chart, histogram, line graph).

model_building.ipynb: Model training and evaluation using multiple CNN architectures.

a. Training Set/ and b. Testing Set/: Folders containing retinal images used for training and testing.

IDRiD_Disease_Grading_Training Labels.xlsx: Labels for training data.

IDRiD_Disease_Grading_Testing Labels.xlsx: Labels for testing data.

" Models & Techniques "

I implemented and tested the following CNN models:

AlexNet

ResNet34

ResNet50

DenseNet121

All models were modified for binary classification: output is 1 if diabetic retinopathy is detected, 0 otherwise.

To improve training efficiency, I used Transfer Learning with pre-trained weights from the ImageNet dataset.

Each model was trained and evaluated using the same preprocessing pipeline and metrics to ensure a fair comparison.

" Results "

Model & Testing Accuracy

AlexNet: 77.67%

DenseNet121: 77.67%

ResNet50: 76.7%

ResNet34: 75.6%

Visualizations and metrics (confusion matrix, ROC curve, classification report) were used to evaluate performance.

Validation accuracy was consistently lower than training accuracy, indicating signs of overfitting.

" Technologies & Libraries "

Python

Jupyter Notebook

pandas, numpy

matplotlib, seaborn

TensorFlow & Keras

scikit-learn

" Data Preprocessing "

Checked for missing values

Dropped irrelevant columns

Used bar charts and pie charts to visualize label distribution

Normalized and preprocessed image data

Author:

Ahad Alosaimi
Computer Science Graduate – 2025

" Notes "

This project is academic and for educational purposes. For feedback or collaboration, feel free to reach out.

