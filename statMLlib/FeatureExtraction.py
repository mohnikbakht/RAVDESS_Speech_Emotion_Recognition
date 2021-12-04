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
    actorTrain = [f"{v:02}" for v in list(range(1, 21, 1))]
    actorTest = [f"{v:02}" for v in list(range(21, 25, 1))]
    emotionAll =  [f"{v:02}" for v in list(range(2, 6, 1))] 
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
    for Actor in (actorTrain+actorTest):
        print(f"Actor {Actor}")
        for Emotion in emotionAll:
            for EmotionalInt in emotionalInt:
                for Statement in statementAll:
                    for Repetition in repAll:
                        
                        
                            
                        if (Emotion == "01") and (EmotionalInt == "02"):
                            EmotionalInt = "01"


                        
    
                        os.chdir(r"Actor_" + Actor )

                        fileName = Modality +'-'+ VocalChannel +'-'+ Emotion +'-'+ EmotionalInt +'-'+ Statement +'-'+ Repetition +'-'+ Actor + '.wav'

                        nFiles = os.walk(os.getcwd())


                        # Extract Raw Audio from Wav File
                        signal, sampleRate = librosa.load(fileName)      
                        
                        
                        #pad zeros
                        max_len=116247
                        signal = np.pad(signal, (0, max_len-len(signal)), 'constant')
                        # print(len(signal)/sampleRate)
                        
                        
                        
                        #normalize signal
                        signal = (signal-np.mean(signal))/np.std(signal)
                        
                        #extract global features
                        tmp_signal=signal.astype(float)
                        global_features = [np.max(tmp_signal), np.min(tmp_signal), np.mean(tmp_signal), np.std(tmp_signal), np.median(tmp_signal), np.max(tmp_signal)-np.min(tmp_signal)]
                        # FEATURE EXTRACTION
                        
                        win_len=1024#int(len(signal)/80)#2400
                        hop_len=int(win_len/2)
                        if Feature_2D:
                        #2D feature
                            

                             # mfcc
                            mfccCoeffs = librosa.feature.mfcc(y=signal.astype(float), sr=sampleRate, n_fft=win_len, hop_length=hop_len, n_mfcc=40)

                            # chroma
                            chromaCoeffs = librosa.feature.chroma_stft(y=signal.astype(float), sr=sampleRate, n_fft=win_len, hop_length=hop_len)

                            # mel spectrogram
                            melspectCoeffs = librosa.feature.melspectrogram(y=signal.astype(float), sr=sampleRate, n_fft=win_len, hop_length=hop_len, n_mels=160)

                            # contrast
                            contrastCoeffs = librosa.feature.spectral_contrast(y=signal.astype(float), sr=sampleRate, n_fft=win_len, hop_length=hop_len, n_bands=6)

                            # tonnetz
                            harmon = librosa.effects.harmonic(signal.astype(float))
                            tonnetz = librosa.feature.tonnetz(y=harmon, sr=sampleRate)

                        else: 
                            #1D feature

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
                          "emotion_number" : int(Emotion)-2,
                          "mfccCoeffs": mfccCoeffs,
                          "chromaCoeffs"   : chromaCoeffs,
                          "melspectCoeffs"  : melspectCoeffs,
                          "contrastCoeffs"    : contrastCoeffs,
                          "tonnetz"  : tonnetz,
                          "global_features": global_features
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
                            if Actor in actorTrain:
                                actor_dir=f"split_by_actor/training_pot/actor_{Actor}"

                                if not os.path.exists(actor_dir):
                                    os.makedirs(actor_dir)
                                with open(actor_dir+'/'+fileName[0:-4] + '_features.pkl', 'wb') as f:
                                    pickle.dump((featureDict), f)
                            if Actor in actorTest:
                                actor_dir=f"split_by_actor/test/actor_{Actor}"

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
                            if Actor in actorTrain:
                                statement_dir=f'arbitrary/training_pot'
                                if not os.path.exists(statement_dir):
                                    os.makedirs(statement_dir)
                                with open(statement_dir+'/'+fileName[0:-4] + '_features.pkl', 'wb') as f:
                                    pickle.dump((featureDict), f)
                            elif Actor in actorTest:
                                statement_dir=f'arbitrary/test'
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