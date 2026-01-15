# Edge-AI-Product-Detection-Inferencing-Workshop
This workshop covers the fundamentals of edge AI with hands-on practice of real-time inferencing. 

## Presentation Slides (before project begins)
https://github.com/hibahkhan2022-rgb/Edge-AI-Product-Detection-Inferencing-Workshop/blob/main/EdgeAIProject.pdf

## Colab Project
### Overview
This project implements an end-to-end image classification pipeline for categorizing consumer products into makeup, skincare, and scents using a lightweight convolutional neural network. The focus is on building a reproducible PyTorch system that can be deployed for real-time inference on resource-constrained devices.
### Issue
The task is formulated as a single-label, multi-class classification problem. It was divided into three categories: skincare, makeup, and scents. It's common for models to get confused due to simple minimalistic packaging. The images from the dataset were hand-picked to show similarities in boxes, lids, minimalistic designing, etc. 
### Dataset
The dataset consists of labeled product images organized in a folder-per-class structure compatible with torchvision.datasets.ImageFolder. 
1. Number of classes: 3
2. Class imbalance is present (scents is the smallest class)
3. Images vary in lighting, background, and packaging style
### Model Architecture
The model uses MobileNetV3-Small as a backbone, initialized with ImageNet pretrained weights. The final classification head is replaced to match the number of target classes. We are choosing this architecture for representational capacity and for inferencing/edge deployment
### Training Procedure
As indicated in the Colab file, the AdamW optimizer and Cross-Entropy loss were used. In transform, the data augmentation techniques included color jitter, horizontal flips, and random resized crops.
### Evaluation
Includes the ConfusionMatrix, per-class precision, recall, and F1 scores
```
Confusion matrix:
 [[25  2  3]
 [ 0 14  2]
 [ 1  0 19]]
```
Classification report:
```
              precision    recall  f1-score   support

      makeup       0.96      0.83      0.89        30
      scents       0.88      0.88      0.88        16
    skincare       0.79      0.95      0.86        20

    accuracy                           0.88        66
   macro avg       0.88      0.89      0.88        66
weighted avg       0.89      0.88      0.88        66
```
### Results
Best validation accuracy achieved: ~88%

Some key observations:
1. Makeup is predicted with high precision.
2. Skincare exhibits high recall but lower precision, acting as a “default” class.
3. Scents is the most challenging class due to visual overlap and fewer samples.

### Error Analysis
It was found misclassifications primarily arise from visual ambiguity rather than optimization failure. Products with shared packaging are often misrepresented across categories. This analysis informed decisions around data augmentation and motivates future work on representation-level diagnostics rather than treating errors as noise.

### Future Direction
This model can be exported for real-time analysis using Torchscript or ONNX
