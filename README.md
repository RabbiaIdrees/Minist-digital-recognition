
## MNIST Handwritten Digit Recognition

This project uses the MNIST dataset of handwritten digits to build a machine learning model capable of recognizing and classifying digits from 0 to 9. The dataset contains 60,000 training images and 10,000 test images, each of which is a 28x28 pixel grayscale image representing a handwritten digit.

### What I Did:

1. **Data Preprocessing**:

   * Loaded the MNIST dataset using Keras.
   * Normalized the pixel values to the range \[0, 1] by dividing by 255.
   * Reshaped the data to a format suitable for convolutional neural networks (CNN), i.e., (28, 28, 1) for each image.

2. **Model Architecture**:

   * Built a convolutional neural network (CNN) with two convolutional layers followed by max-pooling layers.
   * Flattened the 2D feature maps and passed them through dense layers, with the final output layer using the softmax activation to classify the digits (0-9).

3. **Model Training**:

   * Used the Adam optimizer and sparse categorical cross-entropy loss function.
   * Trained the model for 5 epochs with a validation split of 0.1 (10%).

4. **Evaluation**:

   * Evaluated the model on the test set to measure its accuracy.
   * Achieved a test accuracy of **98.47%**.

5. **Deployment**:

   * Saved the trained model as `cnn_mnist_model.h5`.
   * Created a custom image prediction function that allows the model to recognize custom handwritten digits uploaded by the user.

6. **Optional Web Interface**:

   * Integrated a Streamlit app to enable real-time drawing and recognition of handwritten digits.

### Results:

* The CNN model successfully classifies digits with a test accuracy of **98.47%**.
* The app allows users to draw digits on a canvas, and the model will predict the digit in real-time.

### Future Improvements:

* Experimenting with deeper architectures or other optimization techniques to improve the accuracy.
* Extending the app with additional features such as digit enhancement or noise reduction.

### Requirements:

* TensorFlow
* Keras
* Streamlit (for the web interface)

Feel free to fork this repository and use the model for your own digit recognition projects!

