import torch
from torch.utils.data import Dataset, DataLoader
import glob
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np


class RAVDESSFeatureDataset(Dataset):
    
    def __init__(self, split='train', split_by='actor', root_dir=''):
        
        self.mean_vec=None
        self.var_vec=None

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
            
        else:
            print(f"no data with split={split}")

    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, idx):
        
        wave_path = self.data[idx]
        pickle_file = open(wave_path, "rb")
        loaded=pickle.load(pickle_file)

        
        mfcc=loaded.get("mfccCoeffs")
        chrom=loaded.get("chromaCoeffs")
        mel=loaded.get("melspectCoeffs")
        contrast=loaded.get("contrastCoeffs")
        # tonnetz=loaded.get("tonnetz")
        src=np.r_[mfcc,chrom,mel,contrast]
        
        
        tgt=loaded.get("emotion_number")


        src_tensor = torch.from_numpy(src).transpose(0,1).float()
        tgt_tensor = torch.tensor(tgt).long()


        return src_tensor, tgt_tensor