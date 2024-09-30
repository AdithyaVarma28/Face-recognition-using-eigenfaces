# FaceRecognition

## Overview

The FaceRecognition project is a Python application designed to recognize faces from images using machine learning techniques. It employs a simple algorithm to train on a dataset of faces and then validate new images against this trained model. This project demonstrates the fundamentals of face recognition and image processing.

## Features

- **Image Preprocessing:** Converts images to grayscale and flattens them for analysis.
- **Training Model:** Utilizes Singular Value Decomposition (SVD) to project images into a lower-dimensional space.
- **Face Recognition:** Compares new images to a training set and identifies the closest match based on Euclidean distance.
- **User-Friendly Output:** Displays the recognized name for each validation image.

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YourUsername/FaceRecognition
   ```
2. **Navigate to the Project Directory:**
   ```bash
   cd FaceRecognition
   ```
3. **Create a Virtual Environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. **Install Dependencies:**
   ```bash
   pip install numpy opencv-python
   ```
   
## Prepare Your Dataset

1. **Create the `Training Set` Directory:**
   - Inside the main project directory, create a folder named `Training Set`.
   - Within the `Training Set` folder, create subdirectories for each individual whose face you want to recognize. Name each subdirectory after the person (e.g., `person1`, `person2`).

   Example structure:
   ```
   Training Set/
   ├── person1/
   │   ├── person1_1.jpeg
   │   ├── person1_2.jpeg
   │   └── person1_3.jpeg
   └── person2/
       ├── person2_1.jpeg
       ├── person2_2.jpeg
       └── person2_3.jpeg
   ```

2. **Create the `Validation Set` Directory:**
   - In the main project directory, create a folder named `Validation Set`.
   - Place the images you want to test for recognition inside this folder. These images can include faces of individuals from the `Training Set` or new faces.

   Example structure:
   ```
   Validation Set/
   ├── test1.jpeg
   ├── test2.jpeg
   └── test3.jpeg
   ```

By organizing your dataset this way, the application can efficiently train and validate the face recognition model.

3. **Run the Application:**
   ```bash
   python FaceRecognition.py
   ```

4. **Output:**
   - The application will print the recognized name for each image in the validation set.

## Contribution Guidelines

- **Reporting Issues:** Please report any issues or bugs using the GitHub Issues tab.
- **Submitting Pull Requests:** Fork the repository, make your changes, and submit a pull request with a description of your changes.

## Related Work

For further reading, you can refer to the foundational paper that inspired this project: [Eigenfaces for Recognition](https://sites.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf).

## Directory Structure

```
.
├── FaceRecognition.py
├── README.md
├── Training Set
├── Validation Set
└── venv
    ├── bin
    ├── include
    │   └── python3.12
    ├── lib
    │   └── python3.12
    │       └── site-packages
    │           ├── cv2
    │           ├── numpy
    │           ├── numpy-2.1.1.dist-info
    │           ├── opencv_python-4.10.0.84.dist-info
    │           ├── pip
    │           └── pip-23.2.1.dist-info
    └── pyvenv.cfg
```
