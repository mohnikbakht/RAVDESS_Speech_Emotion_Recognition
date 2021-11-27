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
    

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size*2, 1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*4, 128)
        self.fc2 = nn.Linear(128, num_classes)  # 2 for bidirection
        
        
        
    def forward(self, x):
        # Set initial states
        # Forward propagate LSTM
#         out, _ = self.gru(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out[:,:300,:])
        
        # Decode the hidden state of the last time step
        out = nn.ReLU()(self.fc1(out[:,-1,:]))
        out = self.fc2(out)
        return out