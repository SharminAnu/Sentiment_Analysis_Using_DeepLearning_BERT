# Sentiment Analysis Using Traditional DeepLearning and BERT (Part of this code was from Kaggle)
The project explores the performance of different neural network architectures and compares them with BERT's approach to understand how each model processes and predicts sentiment.

**Traditional Deep Learning Models**
We implemented several traditional deep learning models, including:

**Simple RNN:** The basic form of RNN, which is good at handling sequence data but struggles with long-term dependencies.
**LSTM (Long Short-Term Memory):** An advanced form of RNN that can capture long-term dependencies better than Simple RNN.
**BiLSTM (Bidirectional LSTM):** An extension of LSTM that processes data in both forward and backward directions to better understand the context.
**CNN (Convolutional Neural Network):** Primarily known for image processing, CNN is also effective in pattern recognition within data, making it useful for text analysis.
Each model was constructed with different layers, activation functions, and dropout rates to prevent overfitting.

**BERT (Bidirectional Encoder Representations from Transformers)**
BERT is a transformer-based model that reads the entire sequence of words at once. This model is pre-trained on a large corpus of text and then fine-tuned for specific tasks like sentiment analysis.

**Data Preparation**
The data was collected from Kaggle (https://www.kaggle.com/datasets/kazanova/sentiment140) and then pre-processed to convert the tweets into a suitable format for model training. We utilized the NLTK library for tokenization and PyTorch's DataLoader for creating efficient training loops.

**Training**
The models were trained with a dataset split of 80% for training and 20% for testing. To accommodate the extensive size of the dataset, we iterated the models for 2 to 5 epochs, which took about 3 hours for traditional models and over 7 hours for BERT.

**Evaluation**
The models were evaluated based on their accuracy and the balance between true positives and negatives. The confusion matrix was used as a primary tool for this evaluation.

**Results**
BERT outperformed all traditional models with an accuracy of 87%. The results suggest that understanding both directions of the data sequence simultaneously provides a significant advantage in sentiment classification tasks.

**Environment and Libraries**
The project was developed in virtual environments using Python 3.12 for traditional models and Python 3.10 for BERT. TensorFlow was employed for traditional models, and PyTorch was used for BERT.

**Usage**
To replicate the results or use the models for your sentiment analysis tasks, follow the instructions in the Jupyter notebooks contained within this repository.

**Contribution**
Contributions to improve the models or extend the dataset are welcome. Please submit a pull request or open an issue to discuss potential changes.
