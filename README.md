# **Image Classification of Omnivores and Herbivores using the CIFAR-100 Dataset**

**Authors: Zach Riane I. Machacon, Josh B. Ratificar, Jahna Patricia Poticar**

## Context

This project aims to classify images into two categories: omnivores and herbivores, utilizing the CIFAR-100 dataset. The CIFAR-100 dataset contains 60,000 32x32 color images in 100 classes, with each class containing 600 images. Here, we focus on distinguishing between animals that primarily consume plants (herbivores) and those that consume both plants and animals (omnivores). Through deep learning techniques and convolutional neural networks (CNNs), we seek to build a model capable of accurate classification among diverse species.

## Prerequisites
The project uses multiple libraries used frequently in machine learning and deep learning projects.
```
pip install tensorflow
pip install matplotlib
pip install seaborn
pip install numpy
pip install pandas
```

## Model Architecture

<p align="center">
    <img src="https://i.imgur.com/NBppEb4.png" width="200" />
</p>

The neural network to be created is a Convolutional Neural Network (CNN). The model architecture comprises multiple convolutional layers with rectified linear unit (ReLU) activation functions, which extract hierarchical features from input images. These convolutional layers are followed by batch normalization layers that normalize the inputs to each layer, improving the stability and convergence of the model during training. Max-pooling layers downsample the feature maps, reducing spatial dimensions and extracting dominant features. Dropout layers are incorporated to prevent overfitting by randomly deactivating neurons during training. The model employs global average pooling to reduce spatial dimensions and compute feature representations. Fully connected layers with ReLU activation functions further process the extracted features before the final softmax layer, which produces class probabilities for multi-class classification. The Adam optimizer with gradient clipping helps stabilize the training process by limiting the magnitude of gradients.

## Transfer Learning

<p align="center">
    <img src="https://i.imgur.com/w5LRcwI.png" width="1200" />
</p>

Transfer learning was done 2 times in order to achieve the highest accuracy we possibly could. Both training phases used the same model as the original training however, with a lower learning rate for fine-tuning. A reduced learning rate is commonly favored as it promotes greater stability during the fine-tuning process. This helps maintain the model's previously acquired features with minimal disruption or significant modifications (Buhl, 2023).

## Conclusions

After 300 epochs of initial training and transfer learning, the model yielded promising results. With a training loss of `0.0002` and a training accuracy of `1.0`, it demonstrated a high level of proficiency in learning the training data. The testing phase revealed a slightly lower but still respectable testing accuracy of `0.79`, accompanied by a testing loss of `1.53`. These metrics suggest that the model was able to generalize reasonably well to unseen data, indicating the effectiveness of transfer learning in this context. 

## Recommendations

The model performed satisfactorily on testing however, overfitting still remains prevalent after transfer learning and data augmentation. These are some points that were taken note by the authors in order to improve the model:

1. **More Data Augmentation**: Expanding the dataset with more samples and varying techniques could reduce overfit and improve the generalization ability of the model.

2. **Regularization Techniques**: Implement regularization techniques like L2 regularization to prevent overfitting during training. These methods can help the model generalize better to unseen data by reducing reliance on specific features.

3. **Hyperparameter Tuning**: Perform systematic hyperparameter tuning to optimize the model's architecture and training parameters. This includes adjusting learning rates, batch sizes, and optimizer algorithms to find the combination that yields the best performance on the testing set.

4. **Model Architecture Exploration**: Experiment with different neural network architectures, such as increasing the depth or width of the network, adding more layers, or trying different activation functions. A more complex model may capture intricate patterns in the data more effectively.

5. **Transfer Learning Refinement**: Fine-tune the transfer learning process by adjusting the layers frozen during pre-training and the learning rate schedule. This allows the model to adapt more specifically to the target task while still benefiting from pre-existing knowledge.

6. **Ensemble Methods**: If allowed to use pre-trained models like `VGG-16` or `YOLOv8`, these models could be ensembled through `bagging` or `boosting` in order to derive a more accurate result.

## Feel free to reach out and send feedback
You can contact me at zachriane01@gmail.com to express your feedback about the project and how I can improve.