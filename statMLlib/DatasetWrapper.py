import torch
from torch.utils.data import Dataset, DataLoader
import glob
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np


class RAVDESSFeatureDataset(Dataset):
    
    def __init__(self, split='train', split_by='actor', root_dir='', mean= torch.tensor(0), std= torch.tensor(1), feature_2d=False):
        
        self.mean_vec=mean
        self.std_vec=std
        self.feature_2d=feature_2d
        
        if split=='train':
            if split_by=='actor':
                self.path=root_dir+'split_by_actor/training/'

                file_list = glob.glob(self.path + "*")

                self.data = []
                for class_path in file_list:
                    # class_name = class_path.split("/")[-1]
                    for wave_path in glob.glob(class_path+"/*.pkl"):
                        self.data.append(wave_path)
                random.shuffle(self.data)
            elif split_by=='statement':
                self.path=root_dir+'split_by_statement/training/'

                file_list = glob.glob(self.path + "*")

                self.data = []
                for class_path in file_list:
                    # class_name = class_path.split("/")[-1]
                    for wave_path in glob.glob(class_path+"/*.pkl"):
                        self.data.append(wave_path)
                random.shuffle(self.data)
                # print(self.data[0])
            elif split_by=='arb':
                self.path=root_dir+'arbitrary/training/'

                file_list = glob.glob(self.path + "/*.pkl")

                self.data = []
                for class_path in file_list:
                    self.data.append(class_path)
                random.shuffle(self.data)

        elif split=='valid':
            if split_by=='actor':
                self.path=root_dir+'split_by_actor/valid/'

                file_list = glob.glob(self.path + "*")

                self.data = []
                for class_path in file_list:
                    # class_name = class_path.split("/")[-1]
                    for wave_path in glob.glob(class_path+"/*.pkl"):
                        self.data.append(wave_path)
                random.shuffle(self.data)
            elif split_by=='statement':
                self.path=root_dir+'split_by_statement/valid/'

                file_list = glob.glob(self.path + "*")

                self.data = []
                for class_path in file_list:
                    # class_name = class_path.split("/")[-1]
                    for wave_path in glob.glob(class_path+"/*.pkl"):
                        self.data.append(wave_path)
                random.shuffle(self.data)
            elif split_by=='arb':
                self.path=root_dir+'arbitrary/valid/'

                
                file_list = glob.glob(self.path + "/*.pkl")

                self.data = []
                for class_path in file_list:
                    self.data.append(class_path)
                random.shuffle(self.data)
            
        elif split=='test':
            if split_by=='actor':
                self.path=root_dir+'split_by_actor/test/'

                file_list = glob.glob(self.path + "*")

                self.data = []
                for class_path in file_list:
                    # class_name = class_path.split("/")[-1]
                    for wave_path in glob.glob(class_path+"/*.pkl"):
                        self.data.append(wave_path)
                random.shuffle(self.data)
            elif split_by=='statement':
                self.path=root_dir+'split_by_statement/test/'

                file_list = glob.glob(self.path + "*")

                self.data = []
                for class_path in file_list:
                    # class_name = class_path.split("/")[-1]
                    for wave_path in glob.glob(class_path+"/*.pkl"):
                        self.data.append(wave_path)
                random.shuffle(self.data)
            elif split_by=='arb':
                self.path=root_dir+'arbitrary/test/'

               
                file_list = glob.glob(self.path + "/*.pkl")

                self.data = []
                for class_path in file_list:
                    self.data.append(class_path)
                random.shuffle(self.data)
                
        elif split=='train_all':
            if split_by=='actor':
                self.path=root_dir+'split_by_actor/training_pot/'

                file_list = glob.glob(self.path + "*")

                self.data = []
                for class_path in file_list:
                    # class_name = class_path.split("/")[-1]
                    for wave_path in glob.glob(class_path+"/*.pkl"):
                        self.data.append(wave_path)
                random.shuffle(self.data)
            elif split_by=='statement':
                self.path=root_dir+'split_by_statement/training_pot/'

                file_list = glob.glob(self.path + "*")

                self.data = []
                for class_path in file_list:
                    # class_name = class_path.split("/")[-1]
                    for wave_path in glob.glob(class_path+"/*.pkl"):
                        self.data.append(wave_path)
                random.shuffle(self.data)
                # print(self.data[0])
            elif split_by=='arb':
                self.path=root_dir+'arbitrary/training_pot/'

                file_list = glob.glob(self.path + "/*.pkl")

                self.data = []
                for class_path in file_list:
                    self.data.append(class_path)
                random.shuffle(self.data)

            
        else:
            print(f"no data with split={split}")

    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, idx):
        
        wave_path = self.data[idx]
        pickle_file = open(wave_path, "rb")
        loaded=pickle.load(pickle_file)

        
        mfcc=loaded.get("mfccCoeffs")
        chrom=np.log(loaded.get("chromaCoeffs")+1)
        mel=np.log(loaded.get("melspectCoeffs")+1)
        # mel=(mel-np.mean(mel))/np.std(mel)
        contrast=np.log(loaded.get("contrastCoeffs")+1)
        # tonnetz=np.log(loaded.get("tonnetz")+1)
        
        if self.feature_2d:
            #2d training data extractor
            # 1) [mel]
            # 2) [mfcc, mel]
            # 3) [mfcc, chrom, mel]
            # 4) [mfcc, chrom, contrast, mel]
            src=np.r_[mfcc, chrom, contrast, mel]
            # src= mel

        else:
            #1d training data extractor
            global_features=loaded.get("global_features")
            
            # 1) [mel]
            # 2) [mel, global_features]
            # 2) [mfcc, mel, global_features]
            # 3) [mfcc, chrom, mel, global_features]

            src=np.r_[mfcc, mel, chrom, global_features]
            
            src=np.reshape(src, (-1,1))

        # src=mel
       
        # print(self.mean_vec)
        # src=(src-self.mean_vec)/self.std_vec
        
        tgt=loaded.get("emotion_number")


        src_tensor = torch.from_numpy(src.T).float()
        # print(src_tensor.shape)
        # print(self.mean_vec.shape)
        src_tensor=(src_tensor-self.mean_vec)/self.std_vec
        tgt_tensor = torch.tensor(tgt).long()
        

        return src_tensor, tgt_tensor