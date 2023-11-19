# MTG Card Detector Training Guide

This guide will walk you through training a YOLO v8 model to detect the collector number and set code on Magic: The Gathering cards.

## Prerequisites

Before starting, ensure you have:

- Python 3.8+
- An appropriate GPU for deep learning (CUDA-enabled is recommended)
- Dependencies from `requirements.txt` installed in your Python environment

## Step 1: Data Collection

Collect images of MTG cards with clear visibility of the collector number and set code.

```bash
# Suggested directory structure for your datasets
datasets/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## Step 2: Data Annotation
Annotate your images using a tool like CVAT or LabelImg. Draw bounding boxes around the collector number and set code areas.

## Step 3: Dataset Preparation
Organize your dataset as follows:

- Place all training images in datasets/images/train/.
- Place all validation images in datasets/images/val/.
- Corresponding annotation files in YOLO format should be placed in datasets/labels/train/ and datasets/labels/val/.

## Step 4: Configuration
Edit the config.yaml file to specify the path to your datasets and define your classes.

```yaml
train: datasets/images/train/
val: datasets/images/val/

nc: 1  # number of classes
names: ['MTG_Card']
```

## Step 5: Training
Run the training script. You can adjust hyperparameters like epochs, batch size, etc., as needed.

```bash
python train.py --img 640 --batch 16 --epochs 100 --data config.yaml --weights yolov8n.pt
```

## Step 6: Evaluation
Assess your model using the validation set:

- Check the accuracy of bounding box predictions
- Review loss metrics to ensure the model is learning effectively

## Step 7: Testing on Unseen Data
To ensure robustness, test your model on a separate set of images not used during training or validation.

## Step 8: Iteration
Based on the testing, you may need to retrain your model with adjusted parameters or additional annotated data.

## Step 9: Deployment
Prepare your model for deployment. You may need to convert it into an efficient format suitable for your application.

## Step 10: Maintenance
Create a maintenance plan:

- Keep detailed records of configurations and training iterations
- Schedule periodic retraining with updated data to maintain model performance