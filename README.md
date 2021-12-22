# Speech Emotion Recognition

Speech and emotions are two of the most important modes of communication among human beings, which makes Speech Emotion Recognition (SER) a key component in Human-Computer Interaction (HCI) systems. The pandemic social restrictions led to a lack of interactions and psychological distress which affected the emotional and mental health of individuals impacted by the pandemic. Thus, a need for remote emotion monitoring is felt. The main goal of this project is to explore different machine learning algorithms on the SER task which can help address this need. In particular, we compare traditional statistical approaches to more modern deep learning methods based on the evaluation metrics to learn more about the structure of the data and the complexity of the SER task.

## Dataset

We use the Ryerson Audio-Visual Database of Emotion Speech and Song (RAVDESS), an English language database commonly used to evaluate SER algorithms. The database is gender-balanced containing 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. In this project, we use audio-only modality, which contains 1440 recordings samples
We split our data samples into three sets: train, validation, and test. To prevent data leakage, we set aside actors 1-21 for training and validation, but held out actors 22-24 for a test set. 

## Feature Extraction

Humans can sense emotions from only sounds. If we convert the raw audio signals into a perceptually meaningful space, we can represent the data more closely to how humans perceive and improve audio classification. Some of these features are:
• Mel-frequency cepstral coefficients (MFCCs)
• Mel spectrogram
• Spectral contrast
• Chromagram

For local features, we divide audio signals into short time windows of length 50 ms with 50\% overlap and compute Mel frequency cepstral coefficients (MFCCs), Mel-spectrogram, chromagram, and spectral contrast. By computing these features, we convert our raw audio signals into a perceptually meaningful space, representing the data in a way that's closer to how humans perceive the audio signals. Doing so has been shown to improve audio classifcation accuracy for SER. For global features, we compute mean, standard deviation, max, min, and range of each raw audio signal. 

<img src="https://github.com/mohnikbakht/RAVDESS_Speech_Emotion_Recognition/blob/main/figs/2D%20features.svg" width="900"/>
<img src="https://github.com/mohnikbakht/RAVDESS_Speech_Emotion_Recognition/blob/main/figs/Fig5.svg" width="900"/>

## Feature Selection

We approach feature selection by implementing the backwards selection algorithm shown below. We use the validation accuracy as the evaluation score throughout this procedure. The results of our selection algorithm for all models are shown in Table below.

<img src="https://github.com/mohnikbakht/RAVDESS_Speech_Emotion_Recognition/blob/main/figs/backward.PNG" width="900"/>

<img src="https://github.com/mohnikbakht/RAVDESS_Speech_Emotion_Recognition/blob/main/figs/backward_alg.PNG" width="900"/>

## Models & Pipeline

There are two classes of algorithms that we investigate: 1) traditional statistical learning methods and 2) deep learning methods. First, we develop linear discriminative models such as logistic regression and support vector machines to establish the baseline performance of the SER classification task. These methods have been shown to be reasonably effective at audio sample classification tasks. Furthermore, we can use the test results to determine if our task is linearly separable given our extracted features, or if more complex models are required. Second, we develop three neural network architectures: 1) multi-layer perceptron (MLP), 2) 1D convolutional (CNN), and 3) 2D convolutional networks (CNN). In recent works, neural networks have been shown to achieve state-of-the-art performance in audio classification tasks. We use Scikit-Learn implementations of LR and SVM, and Pytorch to build all deep learning models. 

<p float="left">
  <img src="https://github.com/mohnikbakht/RAVDESS_Speech_Emotion_Recognition/blob/main/figs/Fig2.svg" width="200"/>
  <img src="https://github.com/mohnikbakht/RAVDESS_Speech_Emotion_Recognition/blob/main/figs/Fig4.svg" width="200"/>
  <img src="https://github.com/mohnikbakht/RAVDESS_Speech_Emotion_Recognition/blob/main/figs/Fig5.svg" width="200"/>
</p>
  
<img src="https://github.com/mohnikbakht/RAVDESS_Speech_Emotion_Recognition/blob/main/figs/Fig1.svg" width="900"/>

## Training Parameters

- Statistical Learning
  - Hyperparameter Search: Grid search to determine regularization strength.
    
- Deep Learning
  - Cross-entropy loss
  - Adam optimizer with learning rate of 1e-4
  - ReLU activation function
  - He initialization
  - Mini-batch size of 8.

## Steps taken for each model

1. Hyperparameter tuning using the train and validation set
    - Regularization strength
    - Network architecture
2. Backward selection to pick the best features
3. Retrain on the whole train + validation set
4. Test on held-out test set and record evaluation metrics
    - Accuracy, Precision, Recall, F1 Score

## Results

Evaluation results for each model are listed in the table below:

<img src="https://github.com/mohnikbakht/RAVDESS_Speech_Emotion_Recognition/blob/main/figs/results.PNG" width="900"/>

### Reports:
- final paper 
- presentation slides
