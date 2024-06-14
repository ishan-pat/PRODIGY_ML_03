
# Image Classification with SVM - UI Application

This project is a user-friendly application designed to simplify the process of training an image classification model using Support Vector Machines (SVM). The application allows users to load image datasets, preprocess the images, train the SVM model, and visualize the classification results.




## Features

1. Dataset Loading: Load image datasets from ZIP files.

2. Image Preprocessing: Resize images, convert to grayscale, and flatten them.

3. Model Training: Train an SVM model on the preprocessed image dataset.

4. Results Visualization: Display classification metrics including precision, recall, and accuracy.

5. CSV Export: Export classification results to a CSV file.
## Requirements
Python 3.x

Tkinter

OpenCV

NumPy

Scikit-learn

Pandas
## Installation

1. Clone the repository

```bash
  git clone https://github.com/yourusername/image-classification-svm.git
  cd image-classification-svm

```
2.  Install Dependencies:

```bash
  pip install -r requirements.txt
```

    
## Usage
1. Run the application 
```bash
    python svm_image_classification.py

```

2. Load Dataset:

Click on the "Browse" button to select a ZIP file containing your image dataset.

3. Train Model:

Click on the "Train Model" button to preprocess the images and train the SVM model.

4. Show Results:

Click on the "Show Results" button to display the classification metrics and visualize the results.

5. Export Results:

Results are automatically saved to a CSV file in the project directory.