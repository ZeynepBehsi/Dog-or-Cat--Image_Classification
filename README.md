## Dog vs Cat Image Classification with VGG16 & CNN
# Project Overview
This project aims to classify images of cats and dogs using a Convolutional Neural Network (CNN) model. Due to limitations in the dataset, the pre-trained VGG16 model, which has been trained on millions of images, was used as the base model. The VGG16 model was fine-tuned and adapted to this specific binary classification task, achieving an impressive 99% accuracy on the training set and 91% accuracy on the test set.

# Features
- **Binary Image Classification:** Classifies images into two categories - Dog or Cat.
- **Pre-trained Model:** Utilizes the VGG16 model, pre-trained on a vast dataset of images.
- **Fine-tuning:** The model is fine-tuned to adapt it to the specific task of classifying dogs and cats.
- **Overfitting Prevention:** Techniques like dropout, early stopping, and learning rate reduction were implemented to avoid overfitting.

# Technologies Used
- Python
- TensorFlow
- Keras
- CNN (Convolutional Neural Network)
- VGG16 Model

# Installation
1. Clone the repository:

```bash
git clone https://github.com/yourusername/Dog-vs-Cat-Classification.git
```

2. Navigate to the project directory:

```bash
cd Dog-vs-Cat-Classification
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Running the Project
- Ensure the dataset is properly structured within the train and validation folders.
- Run the Jupyter Notebook to start training the model:

```bash
jupyter notebook main.ipynb
```

- Follow the steps in the notebook to train the model, evaluate its performance, and make predictions.


# Results
- Training Accuracy: 99%
- Test Accuracy: 91%
- The model successfully classifies dog and cat images with high accuracy. Overfitting was mitigated using techniques such as dropout, early stopping, and learning rate reduction.

# Conclusion
This project demonstrates the effectiveness of using a pre-trained model like VGG16 for image classification tasks, especially when dealing with limited datasets. By fine-tuning the model and applying overfitting prevention techniques, high accuracy was achieved.

