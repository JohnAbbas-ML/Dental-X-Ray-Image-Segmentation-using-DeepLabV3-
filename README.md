# Dental X-Ray Image Segmentation using DeepLabV3+

## Introduction

This project focuses on dental X-ray image segmentation using a deep learning-based approach. The goal is to develop a segmentation model capable of accurately identifying different regions within dental X-ray images, which can assist in automated dental diagnostics and analysis. The dataset consists of X-ray images and corresponding segmentation masks, and the DeepLabV3+ model is employed for segmentation tasks.

The implementation includes dataset preprocessing, data augmentation, training with DeepLabV3+, and evaluation of the model’s performance using metrics such as Mean Intersection over Union (mIoU) and Pixel Accuracy (PA). The project leverages PyTorch, Albumentations, and Segmentation Models PyTorch (SMP) for efficient training and inference.

---

## Proposed Methodology

### 1. Custom Dataset Class
A `CustomSegmentationDataset` class is implemented using PyTorch’s `Dataset` class to:
- Load images and masks.
- Apply transformations such as resizing and normalization using Albumentations.
- Convert images and masks to PyTorch tensors.
- Ensure pixel values are correctly scaled.

### 2. Data Augmentation & Normalization
- Albumentations library is used for data augmentation.
- Images are resized to `(256x256)`, normalized with ImageNet mean and standard deviation.
- Masks are converted into binary format (0 or 1) for segmentation.

### 3. DataLoader Preparation
- The dataset is split into training (90%), validation (5%), and testing (5%).
- DataLoaders are created for efficient batch processing during training and evaluation.

### 4. Model Selection
- DeepLabV3+ architecture is chosen for segmentation.
- The model is initialized with two output classes (background and foreground).

### 5. Training Pipeline
The training process follows these steps:
- The model is trained using `CrossEntropyLoss` as the loss function.
- Adam optimizer is used with a learning rate of `3e-4`.
- Training runs for `10 epochs` with early stopping based on validation loss.
- Performance metrics (mIoU, PA) are computed during training.

### 6. Evaluation Metrics
- **Pixel Accuracy (PA)**: Measures how many pixels are correctly classified.
- **Mean IoU (mIoU)**: Computes the average Intersection over Union across classes.
- **Loss Curve**: Tracks training and validation loss to monitor learning progress.

---

## Testing & Results

### 1. Model Performance
The model is evaluated using the validation and test sets. The best model is saved based on validation loss improvements.
- **Best Validation Loss**: Achieved through early stopping.
- **mIoU and PA Scores**: Displayed for each epoch to track model improvement.

### 2. Visualization of Predictions
The segmentation results are visualized by plotting:
- Original X-ray image
- Ground truth segmentation mask
- Model’s predicted mask

Example results show the model’s ability to accurately segment dental structures in X-ray images.

### 3. Performance Graphs
The following learning curves are plotted:
- **IoU Learning Curve**: Shows improvement of mIoU over epochs.
- **Pixel Accuracy Curve**: Displays how PA improves with training.
- **Loss Curve**: Tracks training and validation loss across epochs.

---

## Installation & Usage

### Prerequisites
- Python 3.x
- PyTorch
- OpenCV
- Albumentations
- Segmentation Models PyTorch (SMP)
- Matplotlib
- NumPy
- tqdm

### Installation
```bash
pip install torch torchvision torchaudio
pip install albumentations opencv-python numpy matplotlib tqdm segmentation-models-pytorch
```

---

## Conclusion
This project successfully implements dental X-ray segmentation using DeepLabV3+. The model demonstrates promising results in segmenting dental structures with high pixel accuracy and IoU scores. Future work includes fine-tuning hyperparameters, experimenting with other segmentation architectures, and expanding the dataset for improved generalization.

---

## Acknowledgments
- Kaggle dataset: *A Collection of Dental X-Ray Images for Analysis*
- PyTorch & Albumentations for image processing
- Segmentation Models PyTorch for pre-built architectures

---
