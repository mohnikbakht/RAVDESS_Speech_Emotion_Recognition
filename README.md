# StatML-Project-2021

Reports:
- final paper 
- presentation slides


# Speech Emotion Recognition

Speech and emotions are two of the most important modes of communication among human beings, which makes Speech Emotion Recognition (SER) a key component in Human-Computer Interaction (HCI) systems. The pandemic social restrictions led to a lack of interactions and psychological distress which affected the emotional and mental health of individuals impacted by the pandemic. Thus, a need for remote emotion monitoring is felt. The main goal of this project is to explore different machine learning algorithms on the SER task which can help address this need. In particular, we compare traditional statistical approaches to more modern deep learning methods based on the evaluation metrics to learn more about the structure of the data and the complexity of the SER task.

## Dataset

We use the Ryerson Audio-Visual Database of Emotion Speech and Song (RAVDESS), an English language database commonly used to evaluate SER algorithms. The database is gender-balanced containing 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. In this project, we use audio-only modality, which contains 1440 recordings samples
We split our data samples into three sets: train, validation, and test. To prevent data leakage, we set aside actors 1-21 for training and validation, but held out actors 22-24 for a test set. 

## Feature Extraction
For local features, we divide audio signals into short time windows of length 50 ms with 50\% overlap and compute Mel frequency cepstral coefficients (MFCCs), Mel-spectrogram, chromagram, and spectral contrast. By computing these features, we convert our raw audio signals into a perceptually meaningful space, representing the data in a way that's closer to how humans perceive the audio signals. Doing so has been shown to improve audio classifcation accuracy for SER. For global features, we compute mean, standard deviation, max, min, and range of each raw audio signal. 


<img src="https://github.com/mohnikbakht/RAVDESS_Speech_Emotion_Recognition/blob/main/figs/2D%20features.svg" width="700"/>
<img src="https://github.com/mohnikbakht/RAVDESS_Speech_Emotion_Recognition/blob/main/figs/Fig2.svg" width="700"/>


## Feature Selection

We approach feature selection by implementing the backwards selection algorithm shown below. We use the validation accuracy as the evaluation score throughout this procedure. The results of our selection algorithm for all models are shown in Table below.

## Models

There are two classes of algorithms that we investigate: 1) traditional statistical learning methods and 2) deep learning methods. First, we develop linear discriminative models such as logistic regression and support vector machines to establish the baseline performance of the SER classification task. These methods have been shown to be reasonably effective at audio sample classification tasks. Furthermore, we can use the test results to determine if our task is linearly separable given our extracted features, or if more complex models are required. Second, we develop three neural network architectures: 1) multi-layer perceptron (MLP), 2) 1D convolutional (CNN), and 3) 2D convolutional networks (CNN). In recent works, neural networks have been shown to achieve state-of-the-art performance in audio classification tasks. We use Scikit-Learn implementations of LR and SVM, and Pytorch to build all deep learning models. 



## Results

| ML Model     | accuracy | precision | recall | f1 score | accuracy | precision | recall | f1 score | accuracy | precision | recall | f1 score |
| ---          | ---      | ---       | ---    | ---      | ---      | ---       | ---    | ---      | ---      | ---       | ---    | ---      |
| LR           | 0.710    | 0.707     | 0.705  | 0.707    | 0.571    | 0.583     | 0.557  | 0.571    | 0.422    | 0.476     | 0.419  | 0.422    |


LR & 0.710 & 0.707 & 0.705 & 0.707 & 0.571 & 0.583 & 0.557 & 0.571 & 0.422 & 0.476 & 0.419 & 0.422\\
 SVM$_\text{lin}$ & 0.798 & 0.800 & 0.795 & 0.798 & 0.595 & 0.612 & 0.589 & 0.595 & 0.378 & 0.431 & 0.365 & 0.378\\
 SVM$_\text{rbf}$ & 0.979 & 0.979 & 0.979 & 0.979 & 0.611 & 0.611 & 0.605 & 0.611 & 0.400 & 0.428 & 0.385 & 0.400 \\ 
 
 MLP  & 0.997 & 0.997 & 0.997 & 0.997 & 0.738 & 0.748 & 0.736 & 0.738 & 0.417 & 0.425 & 0.384 & 0.384 \\ 
 
 1D-CNN  & 1 & 1 & 1 & 1 & 0.746 & 0.769 & 0.748 & 0.748 & 0.356 & 0.364  & 0.320 & 0.320 \\ 
 
 2D-CNN & 0.951 & 0.952 & 0.951 & 0.951 & 0.714 & 0.762 & 0.712 & 0.714 & \textbf{0.489} & \textbf{0.479} & \textbf{0.463} & \textbf{0.489}\\ 

<img src="https://github.com/mohnikbakht/Synthetic_ECG_SCG_Generator_Demo/blob/main/Images/RPi.png" alt="Image of The ECG/SCG Patch" width="300"/>


<img src="https://github.com/mohnikbakht/Synthetic_ECG_SCG_Generator_Demo/blob/main/Images/Actuator.png" alt="Image of The ECG/SCG Patch" width="300"/>


The actuator is non-linear in low frequency and needs calibration to make sure it replicates the input signal to the output in form of acceleration.
The code is written in Python and in a way that you have two options:

1) Calibration mode: Calibrate the system (in a new environment):
```console
python3 Synthetic_SCG.py calibrate
```

<img src="https://github.com/mohnikbakht/Synthetic_ECG_SCG_Generator_Demo/blob/main/Images/calibrate_1.png" alt="Image of The ECG/SCG Patch" width="400"/>


2) Generation mode: Generate SCG waveforms:

```console
python3 Synthetic_SCG.py generate signal1.wav
```

<img src="https://github.com/mohnikbakht/Synthetic_ECG_SCG_Generator_Demo/blob/main/Images/generate_1.png" alt="Image of The ECG/SCG Patch" width="400"/>


## Use

Depending on the type of the connection to the RPi, there are 2 options:
1) If a display is connected to the RPi (or have X11 forwarding enabled), the plots will be shown in separate windows. Use this command (this mode is default):
```console
python3 Synthetic_SCG.py generate signal1.wav desktop
```
2) If communication is through a terminal only, the plots will be plotted in the terminal (lower quality). Use this command:
```console
python3 Synthetic_SCG.py generate signal1.wav terminal
```
## Output Recording Samples

A 10s generated ECG and SCG with a heart-rate of 60 bpm:

<img src="https://github.com/mohnikbakht/Synthetic_ECG_SCG_Generator_Demo/blob/main/Images/sample_1.png" alt="Image of The ECG/SCG Patch" width="600"/>

Zoomed in:

<img src="https://github.com/mohnikbakht/Synthetic_ECG_SCG_Generator_Demo/blob/main/Images/sample_2.png" alt="Image of The ECG/SCG Patch" width="600"/>



