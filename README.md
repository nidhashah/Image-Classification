# Image-Classification

Image Classification Using Convolutional Neural Network
Summary

This script implements a Convolutional Neural Network (CNN) for image classification of sports personalities. The dataset includes images of five sports icons: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli. The script performs data preprocessing, model training, evaluation, and prediction.
Data Preprocessing

The script loads images from the specified directories for each sports personality. Images are resized to (128, 128) pixels using the OpenCV and Pillow libraries. The script creates a dataset array containing the image data and a label array specifying the class labels (0 to 4) for each personality.
Train-Test Split

The dataset is split into training and testing sets using the train_test_split function from scikit-learn. Image pixel values are normalized to the range [0, 1] by dividing by 255.
Convolutional Neural Network Model

The CNN model is built using the Keras Sequential API. It consists of a convolutional layer, max-pooling layer, flatten layer, and densely connected layers. Dropout is applied for regularization to prevent overfitting. The model outputs probabilities using the softmax activation function.
Training

The model is compiled using the Adam optimizer, categorical crossentropy loss, and accuracy as the evaluation metric. Training is performed for 50 epochs with a batch size of 128, and a 10% validation split is applied.
Evaluation

Model evaluation includes computing accuracy on the test set. The script generates a classification report with precision, recall, and F1-score for each class.
Prediction

The script includes a section for predicting the class of new images. A list of image paths to predict is provided, and the model predicts the class for each image.
Critical Findings

The model achieves a certain accuracy on the test set, but further optimization may be explored for better performance. The script uses categorical crossentropy as the loss function, but labels are provided as integers. Consider using sparse categorical crossentropy or one-hot encoding labels. Data augmentation, such as random flips, rotations, and zoom, is commented out. Experimenting with data augmentation may enhance the model's generalization capability. The code lacks exception handling for potential errors during image loading or model training.
  

