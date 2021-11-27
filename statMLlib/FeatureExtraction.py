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


def zeroPadFiles(root_dir):


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
    rootDir = root_dir

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

    os.chdir(root_dir)
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

                        # signalWav = wave.open(fileName, 'r')


                        # Extract Raw Audio from Wav File
                        
                        signal, sampleRate = librosa.load(fileName)
                        # signal = signalWav.readframes(-1)
                        # signal = np.frombuffer(signal, "int16")

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

    
def extractFeatures(root_dir, split_by, Feature_2D=False):    

    Modality = '03'         #   (01 = full-AV, 02 = video-only, 03 = audio-only).
    VocalChannel = '01'     #   (01 = speech, 02 = song).
    Emotion = '04'          #   (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry,
                            #   (06 = fearful, 07 = disgust, 08 = surprised)
    EmotionalInt = '01'     #   (01 = normal, 02 = strong).
    Statement = '01'        #   (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
    Repetition = '01'       #   (01 = 1st repetition, 02 = 2nd repetition).
    Actor = '20'            #   (01 to 24. Odd number =  male, even number =  female).

    fileName = Modality +'-'+ VocalChannel +'-'+ Emotion +'-'+ EmotionalInt +'-'+ Statement +'-'+ Repetition +'-'+ Actor + '.wav'
    rootDir = root_dir

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

    # sampleRate = int(48e3);

    nFilesPerActor = 60

    actorMaleArray = [f"{v:02}" for v in list(range(1, 25, 2))]     # use if want to loop over male speeches
    actorFemaleArray = [f"{v:02}" for v in list(range(2, 25, 2))]   # use if want to loop over female speeches
    actorAll = [f"{v:02}" for v in list(range(1, 25, 1))]           # use if want to loop over all actors speeches
    actorTrain = [f"{v:02}" for v in list(range(1, 25, 1))]
    actorTest = [f"{v:02}" for v in list(range(21, 25, 1))]
    emotionAll =  [f"{v:02}" for v in list(range(1, 9, 1))] 
    statementAll =  [f"{v:02}" for v in list(range(1, 3, 1))] 
    repAll =  [f"{v:02}" for v in list(range(1, 3, 1))] 
    emotionalInt =  [f"{v:02}" for v in list(range(1, 3, 1))] 

    os.chdir(rootDir)
    minSignalLen = int(140941)
    maxSignalLen = int(419618)

    trimSigFlag = 1
    debugPoint = 0
    stopVAr = 0
    lenVec = []
    
    # max_t=0
    for Actor in actorTrain:
        print(f"Actor {Actor}")
        for Emotion in emotionAll:
            for EmotionalInt in emotionalInt:
                for Statement in statementAll:
                    for Repetition in repAll:

                        if (Emotion == "01") and (EmotionalInt == "02"):
                            EmotionalInt = "01"


                        
                        # fileName = Modality +'-'+ VocalChannel +'-'+ Emotion +'-'+ EmotionalInt +'-'+ Statement +'-'+ Repetition +'-'+ Actor + '.wav'

#                         os.chdir(r"Actor_" + fileName[-6:-4] + '_extended')

#                         fileName = Modality +'-'+ VocalChannel +'-'+ Emotion +'-'+ EmotionalInt +'-'+ Statement +'-'+ Repetition +'-'+ Actor + '_extended' + '.wav'
                        
    
                        os.chdir(r"Actor_" + Actor )

                        fileName = Modality +'-'+ VocalChannel +'-'+ Emotion +'-'+ EmotionalInt +'-'+ Statement +'-'+ Repetition +'-'+ Actor + '.wav'

                        nFiles = os.walk(os.getcwd())

                        # signalWav = wave.open(fileName, 'r')

                        # Extract Raw Audio from Wav File
                        signal, sampleRate = librosa.load(fileName)
#                         if (max_t<(len(signal)/sampleRate)):
#                             max_t=(len(signal)/sampleRate)
#                             print(f" {len(signal)}   {sampleRate}")
                            
                        
                        
                        #pad zeros
                        max_len=116247
                        signal = np.pad(signal, (0, max_len-len(signal)), 'constant')
                        # print(len(signal)/sampleRate)
                        #normalize signal
                        signal = (signal-np.mean(signal))/np.std(signal)
                        
                        # FEATURE EXTRACTION
                        
                        
                        if Feature_2D:
                        #2D feature
                            win_len=1024#int(len(signal)/80)#2400
                            hop_len=int(win_len/2)

                             # mfcc
                            mfccCoeffs = librosa.feature.mfcc(y=signal.astype(float), sr=sampleRate, n_fft=win_len, hop_length=hop_len, n_mfcc=40)

                            # chroma
                            chromaCoeffs = librosa.feature.chroma_stft(y=signal.astype(float), sr=sampleRate, n_fft=win_len, hop_length=hop_len)

                            # mel spectrogram
                            melspectCoeffs = librosa.feature.melspectrogram(y=signal.astype(float), sr=sampleRate, n_fft=win_len, hop_length=hop_len, n_mels=128)

                            # contrast
                            contrastCoeffs = librosa.feature.spectral_contrast(y=signal.astype(float), sr=sampleRate, n_fft=win_len, hop_length=hop_len, n_bands=6)

                            # tonnetz
                            harmon = librosa.effects.harmonic(signal.astype(float))
                            tonnetz = librosa.feature.tonnetz(y=harmon, sr=sampleRate)

                        else: 
                            #1D feature
                            win_len=int(len(signal)/40)#2400
                            hop_len=int(win_len/2)

                            ## MFCCs
                            mfccCoeffs=np.mean(librosa.feature.mfcc(y=signal.astype(float), sr=sampleRate, n_mfcc=40), axis=1)

                            ## Chroma
                            stft=np.abs(librosa.stft(signal.astype(float)))
                            chromaCoeffs=np.mean(librosa.feature.chroma_stft(S=stft, sr=sampleRate),axis=1)

                            ## Mel Scale
                            melspectCoeffs=np.mean(librosa.feature.melspectrogram(signal.astype(float), sr=sampleRate),axis=1)
                            
#                             # mfcc
#                             mfccCoeffs = np.mean(librosa.feature.mfcc(y=signal.astype(float), sr=sampleRate, n_fft=win_len, hop_length=hop_len, n_mfcc=40,fmax=8000),axis=1)

#                             # chroma
#                             chromaCoeffs = np.mean(librosa.feature.chroma_stft(y=signal.astype(float), sr=sampleRate, n_fft=win_len, hop_length=hop_len, n_chroma=20),axis=1)

#                             # mel spectrogram
#                             melspectCoeffs = np.mean(librosa.feature.melspectrogram(y=signal.astype(float), sr=sampleRate, n_fft=win_len, hop_length=hop_len, n_mels=64,fmax=8000), axis=1)

                            # contrast
                            contrastCoeffs = np.mean(librosa.feature.spectral_contrast(y=signal.astype(float), sr=sampleRate, n_fft=win_len, hop_length=hop_len, n_bands=6), axis=1)

                            # tonnetz
                            harmon = librosa.effects.harmonic(signal.astype(float))
                            tonnetz = librosa.feature.tonnetz(y=harmon, sr=sampleRate)

                        featureDict =	{
                          "ID" :   fileName[0:-13],
                          "emotion" : emotionDict[Emotion],
                          "emotion_number" : int(Emotion)-1,
                          "mfccCoeffs": mfccCoeffs,
                          "chromaCoeffs"   : chromaCoeffs,
                          "melspectCoeffs"  : melspectCoeffs,
                          "contrastCoeffs"    : contrastCoeffs,
                          "tonnetz"  : tonnetz
                        }

                        os.chdir(root_dir)
                        if Feature_2D:
                            newpath = r"FeaturesAll/2d-cnn"
                        else:
                            newpath = r"FeaturesAll/1d-cnn"

                                
                        if not os.path.exists(newpath):
                                os.makedirs(newpath)

                        os.chdir(newpath)
                        
                        if split_by=='actor':
                            # SAVE DATA
                            actor_dir=f"split_by_actor/actor_{Actor}"
                            
                            if not os.path.exists(actor_dir):
                                os.makedirs(actor_dir)
                            with open(actor_dir+'/'+fileName[0:-4] + '_features.pkl', 'wb') as f:
                                pickle.dump((featureDict), f)
                        elif split_by=='statement':
                            # SAVE DATA
                            statement_dir=f'split_by_statement/actor_{Statement}'
                            if not os.path.exists(statement_dir):
                                os.makedirs(statement_dir)
                            with open(statement_dir+'/'+fileName[0:-4] + '_features.pkl', 'wb') as f:
                                pickle.dump((featureDict), f)
                                
                        elif split_by=='arb':
                            # SAVE DATA
                            statement_dir=f'arbitrary/all'
                            if not os.path.exists(statement_dir):
                                os.makedirs(statement_dir)
                            with open(statement_dir+'/'+fileName[0:-4] + '_features.pkl', 'wb') as f:
                                pickle.dump((featureDict), f)
                        else:
                            print("error: not a valid split option")
                            return 1
                        
                        os.chdir(root_dir)


    # print(max_t)
    stopVAr = 1
# fig, ax = plt.subplots()

# img = librosa.display.specshow(chromaCoeffs, x_axis='time', ax=ax)

# fig.colorbar(img, ax=ax)

# ax.set(title='MFCC')