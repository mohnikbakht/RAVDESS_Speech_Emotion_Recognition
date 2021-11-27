from sklearn.metrics import confusion_matrix

def confusionMatrix(net, dataloader):
    
   
    y_true = []
    y_pred = []
    for data in dataloader:
        inputs,labels = data
        outputs = net(inputs.float())
        predictions = outputs.argmax(1)
        y_true.append(labels.argmax(1).numpy())
        y_pred.append(predictions.numpy())


    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    return cm

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0, reset_patience=4):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.epoch = 0
        self.last_epoch_trigger = 0
        self.reset_patience = reset_patience
        
        
        
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            self.last_epoch_trigger = self.epoch
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
        if self.epoch - self.last_epoch_trigger > self.reset_patience and self.counter > 0:
            print(f'INFO: Resetting Patience')
            self.counter = 0
            
        
        self.epoch += 1
        
        
from torch.utils.data import Dataset, DataLoader
import numpy as np

class AudioData(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.Y_oh  = np.c_[[np.eye(self.Y.max()+1)[i] for i in self.Y.flatten()]]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):        
        return self.X[idx], self.Y_oh[idx]
    
    
def mfccCoeffScale(X):
    k = np.clip(X, X.min(), 50)
    s = k/np.std(k)
    return s
    
def chromaCoeffScale(X):
    return X


def contrastCoeffScale(X):
    return (X-np.median(X))/X.std()

def tonnetzScale(X):
    return X/X.std()

def melspectCoeffScale(X):
    s = X**(1/15)
    return s/s.std()

standard_scale = {'mfccCoeffs': mfccCoeffScale,
                 'chromaCoeffs': chromaCoeffScale,
                 'contrastCoeffs': contrastCoeffScale,
                 'tonnetz': tonnetzScale,
                 'melspectCoeffs': melspectCoeffScale}

emotion_labels = {'angry': 0, 
                  'calm': 1, 
                  'disgust': 2, 
                  'fearful': 3, 
                  'happy': 4 , 
                  'neutral': 5, 
                  'sad': 6,
                  'surprised': 7}