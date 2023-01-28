# Fresh-Fruit-And-Rotten-Fruit-Classifier

# Introduction:
Fruits are an essential part of our daily diet, providing essential nutrients and vitamins for maintaining a healthy lifestyle. However, the quality of fruits can vary greatly, with some being fresh and ripe while others may be spoiled or rotten. In this project, we aim to classify fresh and rotten fruits using machine learning techniques. The goal of this study is to develop a model that can accurately distinguish between fresh and rotten fruits based on their visual characteristics. This model can be used to improve the efficiency and accuracy of fruit sorting in the food industry, reducing food waste and ensuring that consumers receive high-quality fruits. In this project we will explore the use of various features and techniques, such as color, texture, and shape analysis, to develop a robust and accurate model for fruit classification. The performance of the proposed model will be evaluated using a dataset of images of fresh and rotten fruits, and its performance will be compared with existing methods. Those are LeNet-5 and AlexNet.

# Literature review:
The classification of fresh and rotten fruits is a well-studied problem in the field of computer vision and machine learning. In recent years, various approaches have been proposed to address this problem, ranging from traditional image processing techniques to deep learning methods.

One of the earliest approaches for fruit classification is based on color analysis. Researchers have proposed using color features, such as hue, saturation, and intensity, to distinguish between fresh and rotten fruits. These methods have shown to be effective in some cases, but they can be sensitive to lighting conditions and may not work well for fruits with similar color characteristics.

Texture analysis has also been used as a feature for fruit classification. Methods such as Gabor filters and Local Binary Patterns (LBP) have been proposed to extract texture features from fruit images. These methods have been shown to be effective in improving the classification accuracy, especially for fruits with similar color characteristics.

More recently, deep learning techniques have been applied to the problem of fruit classification. Convolutional Neural Networks (CNNs) have been used to extract features from fruit images and classify them as fresh or rotten. These methods have shown to be highly effective and have surpassed the performance of traditional methods.

In addition to the above-mentioned methods, other researchers used multiple modalities such as depth, thermal, and multi-spectral images in combination with traditional or deep learning approaches.

In summary, the literature review suggests that various approaches have been proposed for the classification of fresh and rotten fruits. Color analysis, texture analysis and deep learning are the most commonly used methods, with deep learning methods showing the best performance. However, the combination of multiple modalities with these methods can further improve the performance. In this project we are going to use a model based on Convolutional Neural Networks (CNNs).

# Dataset:
Dataset is taken from kaggle website.

Dataset Link: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

In this dataset there are three fruits namely apple, orange and banana and each fruit is having two classes fresh and rotten so this dataset consists of six classes/folders and each folder is having approximately 2000 images and they are further classified as over 1600 images for testing and nearly 400 for training our model.

So our goal is to classify the given image among those six classes so our model is supervised machine learning model or we can also say it is a supervised deep learning model because we are using convolution and pooling layers.

# Methodology:
The methodology for fresh fruit and rotten fruit classification in this project involves using a convolutional neural network (CNN) model trained on a dataset of images of fresh and rotten fruits. The dataset is taken from the Kaggle website and is split into a training set and a validation set.

The model architecture used is a custom CNN model consisting of multiple convolutional and max pooling layers, as well as dropout layers for regularization. The model is trained using the Adam optimizer and categorical cross-entropy loss function.

The model is compared against two well-known CNN architectures: LeNet-5 and AlexNet. The performance of the custom model is evaluated by comparing its accuracy and loss on the validation set against the accuracy and loss of LeNet-5 and AlexNet.

The Following are the steps to build the model:
  1. In the code provided, the first step is to load the dataset using the function "tf.keras.preprocessing.image_dataset_from_directory()" and set the batch size, image size, and validation split. 

  2. Then the data is preprocessed by normalizing the pixel values and one hot encoding the labels. 

  3. After that, the custom CNN architecture is built using the Keras Sequential API, which consist of multiple convolutional layers, max pooling layers, batch normalization layers, dropout layers and dense layers. 

  4. Finally, the model is compiled with the Adam optimizer and categorical cross-entropy loss function and trained on the training set for 20 epochs. 

  5. The results are evaluated on the validation set.
  
# Results:
model summary for the custom CNN model:

![image](https://user-images.githubusercontent.com/72589374/215264195-fa675b2b-ac25-48e7-8ec5-b392f51e63ae.png)

And our CNN model gives an accuray of 96% on the dataset that we considered and the following graph shows the increase in training accuracy with each epoch.

![image](https://user-images.githubusercontent.com/72589374/215264388-747c5b87-8c17-47fc-9508-461886d44f8f.png)

And the following are the training results for LeNet-5 and AlexNet

![image](https://user-images.githubusercontent.com/72589374/215264447-0b431eeb-fcea-435b-9424-e2a17a61a241.png)

![image](https://user-images.githubusercontent.com/72589374/215264472-7e15b695-d0f9-465d-85f1-8780f005572a.png)

Custom Model VS LeNet-5 VS AlexNet on training data

![image](https://user-images.githubusercontent.com/72589374/215264568-3691ad70-52b5-41b8-9cd3-922be84f1384.png)

The results for the CNN model show that it has a test accuracy of 96%. The model summary shows that it is a sequential model with multiple convolutional layers, batch normalization layers, max pooling layers, and dropout layers. The total number of parameters in the model is 1,343,014 and the number of trainable parameters is 1,341,094.

![image](https://user-images.githubusercontent.com/72589374/215264587-eda921c2-09af-4dc3-b035-4ae6badb47dc.png)

When compared with the LeNet-5 model, which has an accuracy of 87% on the same dataset, the CNN model has a higher accuracy of 96%. Similarly, when compared with AlexNet, which has an accuracy of 85% on the same dataset, the CNN model has a higher accuracy of 96%. This suggests that the CNN model is a more effective model for classifying fresh and rotten fruits in the given dataset.

# Conclusion:
Based on the results, it appears that the CNN model used for fresh fruit and rotten fruit classification achieved a test accuracy of 96%, which is significantly higher than the test accuracy of LeNet-5 and AlexNet models, respectively, on the same dataset. This suggests that the CNN model is a more effective choice for this specific classification task. Additionally, the model summary indicates that the CNN model has a larger number of parameters and is more complex than the LeNet-5 and AlexNet models, which may have contributed to its improved performance. Overall, it seems that the CNN model is a strong choice for fresh fruit and rotten fruit classification, and further exploration and fine-tuning of this model may lead to even better performance.
