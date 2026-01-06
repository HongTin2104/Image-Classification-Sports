# Sports Image Classification

A deep learning project for classifying sports images into 100 different sport categories using transfer learning with MobileNetV2.

## Overview

This project implements a convolutional neural network (CNN) using transfer learning to classify sports images. The model is built on top of MobileNetV2 (pre-trained on ImageNet) and fine-tuned for sports classification. The model achieves high accuracy in identifying various sports from images.

## Features

- **100 Sport Categories**: Classifies images into 100 different sports including:
  - Team sports (basketball, football, volleyball, etc.)
  - Individual sports (archery, swimming, tennis, etc.)
  - Extreme sports (skydiving, rock climbing, etc.)
  - Water sports (surfing, water polo, etc.)
  - And many more!

- **Transfer Learning**: Uses MobileNetV2 pre-trained on ImageNet for faster and better training
- **Data Augmentation**: Implements various augmentation techniques to improve model generalization
- **GPU Support**: Optimized for GPU training (including Mac MPS support)
- **Comprehensive Evaluation**: Includes training, validation, and test set evaluation with visualization

## Dataset

The dataset is organized in the following structure:
```
Sports_Data/
├── train/          # Training images (13,492 images)
├── valid/          # Validation images (500 images)
└── test/           # Test images (500 images)
```

Each sport category has its own subdirectory containing images.

## Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 180x180 pixels
- **Architecture**:
  - MobileNetV2 base (frozen)
  - GlobalAveragePooling2D
  - Dense layers with dropout and batch normalization
  - Final softmax layer for 100-class classification

## Requirements

### Python Version
Python 3.8 or higher

### Dependencies

Install the required packages using:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- TensorFlow >= 2.16.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- Pillow >= 10.0.0

### Mac GPU Support (Optional)

For Mac with Apple Silicon (M1/M2/M3), install tensorflow-metal for GPU acceleration:

```bash
pip install tensorflow-metal>=1.0.0
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Image-Classification-Sports
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the dataset is in the `Sports_Data/` directory with the structure mentioned above.

## Usage

### Training the Model

Open and run the Jupyter notebook `Sports_Image_Classification.ipynb`:

```bash
jupyter notebook Sports_Image_Classification.ipynb
```

The notebook includes the following steps:

1. **Import Libraries**: Set up TensorFlow and configure GPU/MPS
2. **Set Up Data Paths**: Configure paths to training, validation, and test data
3. **Set Parameters**: Define image size, batch size, and number of classes
4. **Load Training Data**: Load training data with data augmentation
5. **Load Validation Data**: Load validation data with normalization
6. **Load Test Data**: Load test data with normalization
7. **Visualize Samples**: Display sample images from the dataset
8. **Build Model**: Create CNN model with transfer learning
9. **Compile Model**: Configure optimizer, loss function, and metrics
10. **Set Up Callbacks**: Configure early stopping, model checkpoint, and learning rate reduction
11. **Train Model**: Train the model on the training dataset
12. **Visualize Results**: Plot training history (accuracy and loss)
13. **Evaluate Model**: Evaluate on test set
14. **Make Predictions**: Predict on new images
15. **Save Model**: Save the final trained model

## Model Performance

The model achieves the following performance metrics:

- **Test Accuracy**: ~86.8%
- **Test Top-5 Accuracy**: ~98.6%
- **Validation Accuracy**: ~85.6%

## Project Structure

```
Image-Classification-Sports/
├── Sports_Image_Classification.ipynb  # Main training notebook
├── sports_classification_model.h5      # Trained model (saved)
├── best_model.h5                       # Best model checkpoint
├── class_names.json                    # Class names mapping
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── LICENSE                             # License file
└── Sports_Data/                        # Dataset directory
    ├── train/                          # Training images
    ├── valid/                          # Validation images
    └── test/                           # Test images
```

## Key Features of the Implementation

1. **Data Augmentation**: 
   - Random horizontal flip
   - Random rotation (±10%)
   - Random zoom (±10%)
   - Random contrast adjustment

2. **Regularization**:
   - Dropout layers (0.2-0.3)
   - L2 regularization
   - Batch normalization

3. **Training Optimizations**:
   - Early stopping to prevent overfitting
   - Model checkpointing to save best model
   - Learning rate reduction on plateau
   - Dataset caching and prefetching for faster training

## Troubleshooting

### GPU Issues on Mac

If you encounter issues with tensorflow-metal:

```bash
pip uninstall tensorflow-metal
pip install tensorflow-metal --upgrade
```

Then restart the Jupyter kernel.

### SSL Certificate Errors

The notebook disables SSL verification for downloading pre-trained weights. This is only for development purposes. In production, ensure proper SSL certificates are configured.

## License

See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- MobileNetV2 architecture and pre-trained weights from TensorFlow/Keras
- ImageNet dataset for pre-training
- All contributors to the open-source deep learning community

## Contact

For questions or issues, please open an issue on the repository.

