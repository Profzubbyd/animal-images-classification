# 1. Introduction

In the growing industry of home surveillance technology, our company, a leading provider of home security solutions, is continually striving to enhance the capabilities of our products to meet the diverse needs of our customers. As part of our commitment to innovation, we recognize the significance of safeguarding not only homes but also the cherished animals that reside within them.

Recent consumer analysis has revealed a compelling trend among our customers who frequently engage in backyard activities, such as raising chickens or spending quality time with their pets outdoors. Acknowledging the shared sentiment of our customers towards the safety of their domestic animals, we have identified an opportunity to introduce an advanced feature to our popular backyard wildlife camera – an alarm function designed to detect potentially dangerous wild animals.

In this research project, we develop a robust and efficient detection model capable of discerning wild animals that pose a threat to domestic animals. By implementing a sophisticated detection system, we aim to empower our customers with an additional layer of protection, allowing them to proactively safeguard their beloved animals from potential threats.

# 2. Methodology

## 2.1 Convolutional Neural Network

A Convolutional Neural Network (CNN) is a deep feedforward neural network that emulates the hierarchical organization observed in biological neural networks. A typical CNN architecture comprises distinct layers, namely the input layer, convolution layer, pooling layer, fully connected layer, and output layer [1]. The convolutional layer, the feature extraction layer, utilizes a set of convolution kernels to extract image features. The pooling layer, the down-sampling layer, efficiently reduces model parameters and reduces the overfitting issue.
At the network's end, the fully connected layer acts as a classifier, transforming the convolutional layer's two-dimensional output vector into a one-dimensional vector. Currently, convolutional layers, or global average pooling layers, are commonly used instead of fully connected layers [2]. Notably, the utilization of vast datasets for training has propelled deep convolutional neural network models to unprecedented levels of speed and accuracy in image recognition over recent years.

## 2.2 MobileNetV2

MobileNetV2 is a variant of the convolutional neural network with notable innovations. Introducing the inverted residual block concept as an evolution of MobileNetV1, MobileNetV2 departs from the traditional residual block paradigm. In the traditional residual block workflow, the process involves dimensionality reduction using a 1×1 convolution kernel, followed by feature filtration with a 3×3 convolution kernel, and subsequent dimensionality enhancement and ReLU activation through a combination of 1×1 convolution kernels. The output features are then added in an element-wise manner to form the input for the next layer [3].
In contrast, the inverted residual block in MobileNetV2 employs a different approach. It utilizes a 1×1 convolution kernel to increase the dimensionality of low-dimensional features (excluding ReLU), followed by another 1×1 convolution kernel to reduce feature dimensions. Additionally, MobileNetV2 adopts a linear bottleneck in place of ReLU to mitigate potential feature degradation. These modifications aim to prevent information loss, thereby enhancing the model's expressive capacity.

## 2.3 Transfer Learning

Transfer learning involves leveraging knowledge acquired in a source domain to address new challenges in a target domain, thereby offering improved solutions [4]. Methods of transfer learning include transfer sample, transfer feature, and transfer parameter [5]. Sample migration combines source and target samples when data similarity is high, adjusting source domain weights to obtain target domain weights. Feature migration identifies feature associations between domains by minimizing differences in feature reconstruction. Parameter migration entails sharing parameters between source and target domains, automatically adjusting weights for optimal results [6]. 
This approach mitigates data insufficiency issues, making transfer learning a preferred technology for projects with insufficient training data or computational resources. The schematic diagram of the transfer learning process is given in the figure below [7].

<p align="center">
  <img width="400" alt="image" src="https://github.com/Profzubbyd/animal-images-classification/assets/46527701/da80a163-a6b0-44de-99c6-bf893afc2a1d">
</p>

<p align="center">
  Figure 1: Schematic diagram of the transfer learning process
</p>


## 2.4 Bayesian Optimization

Bayesian optimization is a powerful technique employed in hyperparameter tuning. It streamlines the process of finding the most suitable hyperparameters by constructing a probabilistic model of the objective function, often representing the model's performance metric. The key objective is to strike a balance between exploration and exploitation to efficiently navigate the hyperparameter space.
Bayesian optimization iteratively selects hyperparameter configurations, evaluates their performance, and updates its probabilistic model based on these observations. By leveraging Gaussian processes or other surrogate models, Bayesian optimization makes informed decisions about where to explore and exploit in the hyperparameter space, focusing on promising regions to enhance efficiency. Bayesian optimization has proven to be particularly advantageous for optimizing complex, black-box functions where the relationship between hyperparameters and model performance is non-trivial.

# 3. Experiment

## 3.1 Dataset

The dataset comprises of 90 animals with 60 images each. For this classification project, we are to work with two categories: not dangerous (domestic animals) and dangerous (wild animals that are harmful to domestic animals). We went through the list and found 10 domestic animals (cat, cow, dog, donkey, goat, goose, hamster, horse, pig, and sheep) that made up the not dangerous class. To ensure the dataset was balanced, we selected 10 other animals (bear, boar, coyote, eagle, fox, hyena, leopard, lion, tiger, wolf) that are harmful to the domestic ones to make up the dangerous class. The final dataset consisted of 1200 images. Figure 2 shows some pictures from the final dataset.

<p align="center">
  <img width="800" alt="image" src="https://github.com/Profzubbyd/animal-images-classification/assets/46527701/0e66c202-2a84-4ce8-8409-047b84aa6dc0">
</p>

<p align="center">
  Figure 2: Final animal dataset sample
</p>



## 3.2 Experimental Process

In this experiment, the dataset was divided into 80% for training and 20% for validation, consisting of two classes. The original images were resized to 128x128 for computational efficiency. The experiment was conducted on Google Colab using the T4 GPU session runtime. A batch size of 16 and 50 epochs were selected for training the network model, with inspiration from Xuefeng Liu et al [8], who used 200 epochs. However, due to Bayesian optimization, the chosen 50 epochs proved sufficient as convergence was observed around that point. Bayesian optimization with epoch=7 yielded the best learning rate (0.0031301075118151454) for training the main network model.

Transfer learning, applied at both stages, involved modifying the feature parameters of the fully connected layer in the pre-trained MobileNetV2 network. The network was then fine-tuned on the target domain animal images, adjusting all layer parameters for improved adaptation to the specific task of animal image classification. The accuracy and loss functions of the MobileNetV2 transfer learning model showed changes during the training process, reaching a best validation accuracy of 95.86%.

<p align="center">
  <img width="800" alt="image" src="https://github.com/Profzubbyd/animal-images-classification/assets/46527701/9a1aabb5-363e-43d0-b782-8b691b9fc86e">
</p>

<p align="center">
  Figure 3: The accuracy and loss function of the MobileNetV2 transfer learning model
</p>

# 4. Results

The model evaluation used a test set of 160 images evenly distributed across two classes, employing various performance metrics including AUC, Accuracy, Precision, Recall, and F1-score. The AUC score of 0.92 indicates a 92% chance of correctly ranking positive instances higher than negative ones, suggesting good performance in binary classification. The ROC curve in Figure 4 visualizes this. The model achieved 90% accuracy, 0.88 precision, 0.925 recall, and an F1 score of 0.9024. The experiment's confusion matrix is presented in Figure 5.

<p align="center">
  <img width="500" alt="image" src="https://github.com/Profzubbyd/animal-images-classification/assets/46527701/dc2f102e-85e9-49d8-961c-a58118640881">
</p>

<p align="center">
  Figure 4: ROC curve of the network model on the test data
</p>

<p align="center">
  <img width="500" alt="image" src="https://github.com/Profzubbyd/animal-images-classification/assets/46527701/4c57f4ba-9755-45c4-ac90-6de59c56b781">
</p>

<p align="center">
  Figure 5: Confusion Matrix
</p>

# 5. Discussion

The experiment applied transfer learning to MobileNetV2 for animal image classification, with Bayesian optimization determining the optimal learning rate (0.0031301075118151454). The best validation accuracy reached 95.86%, while the overall model achieved 90% accuracy, 0.88 precision, 0.925 recall, and an F1 score of 0.9024. The study solely used MobileNetV2 and suggests potential improvements through comparing with other networks, implementing data preprocessing, and exploring data enhancement techniques in future research.

# References

[1] Keqing Zhu, Jie Tian, and Haining Huang, “Underwater object Images Classification Based on convolutional neural network,” 2018 IEEE 3rd International Conference on Signal and Image Processing (ICSIP), pp. 301-305, 2018.

[2] Yan Chunman, Wang Cheng. Development and Application of Convolutional Neural Network Model[J].Computer Science and Exploration

[3] Sandler M, Howard A, Zhu Menglong et al, “MobileNetV2: Inverted Residuals and Linear bottlenecks,” http://arxiv.org/abs/1801.04381, 2018.

[4] Xin Sun, Junyu Shi, Lipeng Liu, et al, “Transferring deep knowledge for object recognition in Low-quality underwater videos,” Neurocomputing, vol. 275, pp. 897-908, 2017.

[5] HooChang Shin, Roth, Holger Roth, Mingchen Gao, and Ronald M Summers, “Deep convolutional neural networks for computer-aided detection: CNN architectures, dataset characteristics, and transfer learning,” IEEE Transactions on Medical Imaging, vol. 35, pp. 1285-1298, 2016.

[6] Ling Shao, Fan Zhu, and Xuelong Li, “Transfer Learning for Visual Categorization: A Survey,” IEEE Transactions on Neural Networks and Learning Systems, vol. 26, pp. 1019-1034, 2015.

[7] Man Hu, and Fucheng You, “Research on animal image classification based on transfer learning,” In Proceedings of the 4th International Conference on Electronic Information Technology and Computer Engineering, Xiamen, China, 6–8 November 2020; Association for Computing Machinery: New York, NY, USA, 2020; pp. 756–761.

[8] Xurfeng Liu, Zhenqing Jia, et al, “Real-time Marine Animal Images Classification by Embedded System Based on Mobilenet and Transfer Learning,” In Proceedings of the OCEANS 2019, Marseille, France, 17–20 June 2019; pp. 1–5.
