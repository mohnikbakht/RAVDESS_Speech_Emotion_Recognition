import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.convnet = nn.Sequential(
            
            nn.Conv1d(1, 128, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(128, 128, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(128, 128, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Flatten(),
#             nn.Linear(73728, 512),
            nn.Linear(283648, 512),
            
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 8),
        )

    def forward(self, x):

        logits = self.convnet(x)
#         sm = nn.functional.softmax(logits+10, dim=1) # numerically stable softmax
        return logits


from torch import nn

class NN(nn.Module):
    def __init__(self, input_dim):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        n_features = 512
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, n_features),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
#             nn.Dropout(0.2),
            
            nn.Linear(n_features, n_features),
#             nn.BatchNorm1d(n_features),
            nn.ReLU(),
#             nn.Dropout(0.2),
            
            nn.Linear(n_features, n_features),
#             nn.BatchNorm1d(n_features),
            nn.ReLU(),
            
            nn.Linear(n_features, 8),
            

        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        
#         sm = nn.functional.softmax(logits+10, dim=1) # numerically stable softmax
        return x
    
