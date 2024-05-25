# Object Tracking and Speed Estimation Project

This project aims to track objects (specifically cars) in video footage and estimate their speeds using computer vision techniques and machine learning models. The project is implemented using Python and utilizes various libraries such as OpenCV, PyTorch, and Keras.

## Files

### `general_concept.py`

This file contains the main code for object detection, tracking, and speed estimation. It uses the YOLOv5 model for object detection and performs speed calculation based on the detected object's movement across frames. The script also loads and processes JSON files containing information about designated areas and zones in the video footage.

### `marking.py`

This file contains code for object tracking using the Kernelized Correlation Filters (KCF) algorithm. It allows the user to select a region of interest (ROI) in the video, and the script tracks the selected object throughout the video frames.

### `speed_auto.py`

This file implements a Convolutional Neural Network (CNN) model using Keras for predicting the speed of objects in video frames. The model is trained on batches of frames, and the predicted speeds are stored in a list.

### `tracker.py`

Similar to `marking.py`, this file also performs object tracking using the KCF algorithm. However, it includes an additional CNN model for predicting the x and y coordinates of the tracked object.

## Usage

1. Ensure that you have the required dependencies installed (OpenCV, PyTorch, Keras, etc.).
2. Place the video files and corresponding JSON files in the appropriate directories specified in the code.
3. Run the desired script (e.g., `python general_concept.py`) to execute the specific functionality.

## Dependencies

The project relies on the following Python libraries:

- OpenCV
- PyTorch
- Keras
- NumPy
- JSON

Make sure to install these dependencies before running the scripts.

## Note

This project involves various computer vision and machine learning techniques for object tracking and speed estimation. The provided code serves as a starting point and may require further modifications or improvements based on your specific requirements and the complexity of the video footage.
