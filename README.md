# MNIST CNN Model Using TensorFlow and Keras (Python)

## Description

This project demonstrates how to build a convolutional neural network (CNN) to classify handwritten digits from the MNIST dataset using TensorFlow and Keras. The model is trained to identify digits from pixel values, with early stopping applied to reduce the risk of overfitting.

## Dataset

The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is 28x28 pixels in grayscale.

## Model Architecture

The model uses a simple CNN architecture with the following layers:
- **Conv2D**: Extracts patterns and features from the input image.
- **MaxPooling2D**: Reduces the size of the image, keeping the most important information.
- **Flatten**: Converts the 2D features into a 1D vector.
- **Dense**: Outputs the final classification of digits (10 classes).

## Code Walkthrough

### Data Loading and Preprocessing
![Data Preprocessing Image](https://github.com/rhb140/MNIST-CNN/blob/main/mnistNNImage2.jpg?raw=true)

Load the MNIST dataset and normalize the pixel values to the range [0, 1], which allows the model to train more efficiently. We also one-hot encode the y data, converting the integer labels into a binary matrix.

### Create the Model
![Model Creation Image](https://github.com/rhb140/MNIST-CNN/blob/main/mnistNNImage3.jpg?raw=true)

Define the model using the sequential class and add convolutional, pooling, flatten, and dense layers to build the network. The **Conv2D** and **MaxPooling2D** layers help the model find important patterns and reduce the image size. The **Flatten** layer reshapes the data into a 1D format that can be used by the **Dense** layer to make the final predictions.

### Compile and Train
![Compile and Train Image](https://github.com/rhb140/MNIST-CNN/blob/main/mnistNNImage4.jpg?raw=true)

Categorical cross-entropy loss and the Adam optimizer are used to compile the model. **Early stopping** is implemented to stop training when the validation loss stops improving, helping to prevent overfitting.

### Evaluation
![Evaluation Image](https://github.com/rhb140/MNIST-CNN/blob/main/mnistNNImage5.jpg?raw=true)

Evaluate the accuracy and loss of the model using the test dataset which is unseen data.

### Accuracy Graph
![Accuracy Graph](https://github.com/rhb140/MNIST-CNN/blob/main/mnistNNImage6.jpg?raw=true)

This graph shows both the training and validation accuracy over the epochs.

### Loss Graph
![Loss Graph](https://github.com/rhb140/MNIST-CNN/blob/main/mnistNNImage7.jpg?raw=true)

This graph shows both the training and validation loss over the epochs.


## Conclusion

The model consistently achieves around 99% accuracy after evaluation. This project demonstrates how to create and train a Convolutional Neural Network (CNN) to classify images from the MNIST dataset. With a simple architecture using convolutional layers and early stopping, the model performs well.

### Author  
Created by [rhb140](https://github.com/rhb140)

### Citation
MNIST dataset:
LeCun, Y., Cortes, C., & Burges, C. J. (1998). The MNIST Database of Handwritten Digits. AT&T Labs Research.
