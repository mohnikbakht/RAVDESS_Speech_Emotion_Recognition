# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 21:52:13 2021

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
import librosa.display




# ---Filename identifiers---

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
  "05" : "angry"
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
emotionAll =  [f"{v:02}" for v in list(range(1, 9, 1))] 
statementAll =  [f"{v:02}" for v in list(range(1, 3, 1))] 
repAll =  [f"{v:02}" for v in list(range(1, 3, 1))] 
emotionalInt =  [f"{v:02}" for v in list(range(1, 3, 1))] 

os.chdir(r"Audio_Speech_Actors_01-24/")
minSignalLen = int(140941)
maxSignalLen = int(419618)

extendSigFlag = 1
debugPoint = 0
lenVec = []

for Actor in actorAll:
    for Emotion in emotionAll:
        for EmotionalInt in emotionalInt:
            for Statement in statementAll:
                for Repetition in repAll:
        
                    if (Emotion == "01") and (EmotionalInt == "02"):
                        EmotionalInt = "01"
                        
                        
                    fileName = Modality +'-'+ VocalChannel +'-'+ Emotion +'-'+ EmotionalInt +'-'+ Statement +'-'+ Repetition +'-'+ Actor + '.wav'
                    
                    os.chdir(r"Actor_" + fileName[-6:-4])
                    
                    nFiles = os.walk(os.getcwd())
                    
                    signalWav = wave.open(fileName, 'r')

                    
                    # Extract Raw Audio from Wav File
                    signal = signalWav.readframes(-1)
                    signal = np.frombuffer(signal, "int16")
                    
                    lenVec.append(len(signal))
                    
                    
                    os.chdir("../")
                    
                    if (extendSigFlag == 1):
                        
                        newpath = r"Actor_" + fileName[-6:-4] + "_extended"
                        if not os.path.exists(newpath):
                            os.makedirs(newpath)
                        
                        samplesToAppend = int(maxSignalLen - len(signal))      # extra samples
                        
                        if (not(samplesToAppend == 0)):
                            signalExtended = np.append(signal, (np.zeros(samplesToAppend)))
                        else:
                            signalExtended = signal
                        
                        
                        os.chdir(newpath)
                        
                        write(fileName[0:-4] + "_extended" ".wav", sampleRate, signalExtended.astype(np.int16))
                        
                        os.chdir("../")


stopVar = 0
