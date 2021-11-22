# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:39:38 2021

@author: skhan
"""
import os
from sys import argv
import re
from shutil import copy2
import wave
import matplotlib.pyplot as plt
import numpy as np
import sys
from python_speech_features import mfcc
from scipy.io.wavfile import write
import librosa
import pickle
    
    

Modality = '03'         #   (01 = full-AV, 02 = video-only, 03 = audio-only).
VocalChannel = '01'     #   (01 = speech, 02 = song).
Emotion = '04'          #   (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry,
                        #   (06 = fearful, 07 = disgust, 08 = surprised)
EmotionalInt = '01'     #   (01 = normal, 02 = strong).
Statement = '01'        #   (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
Repetition = '01'       #   (01 = 1st repetition, 02 = 2nd repetition).
Actor = '20'            #   (01 to 24. Odd number =  male, even number =  female).

fileName = Modality +'-'+ VocalChannel +'-'+ Emotion +'-'+ EmotionalInt +'-'+ Statement +'-'+ Repetition +'-'+ Actor + '.wav'
rootDir = 'Audio_Speech_Actors_01-24/'

emotionDict =	{
  "01" : "neutral",
  "02" : "calm",
  "03" : "happy",
  "04" : "sad",
  "05" : "angry",
  "06" : "fearful",
  "07" :  "disgust",
  "08" : "surprised"
}

nFilesPerEmotion =	{
  "neutral": 96,
  "calm"   : 192,
  "happy"  : 192,
  "sad"    : 192,
  "angry"  : 192
}

sampleRate = int(48e3);

nFilesPerActor = 60

actorMaleArray = [f"{v:02}" for v in list(range(1, 25, 2))]     # use if want to loop over male speeches
actorFemaleArray = [f"{v:02}" for v in list(range(2, 25, 2))]   # use if want to loop over female speeches
actorAll = [f"{v:02}" for v in list(range(1, 25, 1))]           # use if want to loop over all actors speeches
actorTrain = [f"{v:02}" for v in list(range(1, 21, 1))]
actorTest = [f"{v:02}" for v in list(range(21, 25, 1))]
emotionAll =  [f"{v:02}" for v in list(range(1, 9, 1))] 
statementAll =  [f"{v:02}" for v in list(range(1, 3, 1))] 
repAll =  [f"{v:02}" for v in list(range(1, 3, 1))] 
emotionalInt =  [f"{v:02}" for v in list(range(1, 3, 1))] 

os.chdir(r"Audio_Speech_Actors_01-24/")
minSignalLen = int(140941)
maxSignalLen = int(419618)

trimSigFlag = 1
debugPoint = 0
stopVAr = 0
lenVec = []

for Actor in actorTrain:
    for Emotion in emotionAll:
        for EmotionalInt in emotionalInt:
            for Statement in statementAll:
                for Repetition in repAll:
        
                    if (Emotion == "01") and (EmotionalInt == "02"):
                        EmotionalInt = "01"
                        
                    
                    
                    fileName = Modality +'-'+ VocalChannel +'-'+ Emotion +'-'+ EmotionalInt +'-'+ Statement +'-'+ Repetition +'-'+ Actor + '.wav'
                    
                    os.chdir(r"Actor_" + fileName[-6:-4] + '_extended')
                    
                    fileName = Modality +'-'+ VocalChannel +'-'+ Emotion +'-'+ EmotionalInt +'-'+ Statement +'-'+ Repetition +'-'+ Actor + '_extended' + '.wav'
                    
                    nFiles = os.walk(os.getcwd())
                    
                    signalWav = wave.open(fileName, 'r')
                    
                    # Extract Raw Audio from Wav File
                    signal = signalWav.readframes(-1)
                    signal = np.frombuffer(signal, "int16")
                    
                    # FEATURE EXTRACTION
                    
                    
                    # mfcc
                    mfccCoeffs = librosa.feature.mfcc(y=signal.astype(float), sr=sampleRate)
                    
                    # chroma
                    chromaCoeffs = librosa.feature.chroma_stft(y=signal.astype(float), sr=sampleRate)
                    
                    # mel spectrogram
                    melspectCoeffs = librosa.feature.melspectrogram(y=signal.astype(float), sr=sampleRate)
                    
                    # contrast
                    S = np.abs(librosa.stft(signal.astype(float)))
                    contrastCoeffs = librosa.feature.spectral_contrast(S=S, sr=sampleRate)
                    
                    # tonnetz
                    harmon = librosa.effects.harmonic(signal.astype(float))
                    tonnetz = librosa.feature.tonnetz(y=harmon, sr=sampleRate)
                    
                    featureDict =	{
                      "ID" :   fileName[0:-13],
                      "emotion" : emotionDict[Emotion],
                      "mfccCoeffs": mfccCoeffs,
                      "chromaCoeffs"   : chromaCoeffs,
                      "melspectCoeffs"  : melspectCoeffs,
                      "contrastCoeffs"    : contrastCoeffs,
                      "tonnetz"  : tonnetz
                    }
                    
                    os.chdir("../")
                    newpath = r"FeatruesAll"
                    if not os.path.exists(newpath):
                            os.makedirs(newpath)
                    
                    os.chdir(newpath)
                    
                    # SAVE DATA
                    with open(fileName[0:-13] + '_features.pkl', 'wb') as f:
                       pickle.dump((featureDict), f)
                    
                    os.chdir("../")
                        


stopVAr = 1
# fig, ax = plt.subplots()

# img = librosa.display.specshow(chromaCoeffs, x_axis='time', ax=ax)

# fig.colorbar(img, ax=ax)

# ax.set(title='MFCC')