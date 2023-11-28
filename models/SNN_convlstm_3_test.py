import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_score is None:
            self.best_score = loss
        elif loss > self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = loss
            self.counter = 0

class ModelCheckpoint:
    def __init__(self, filepath, monitor="loss", verbose=0, save_best_only=True):
        self.filepath       = filepath
        self.monitor        = monitor
        self.verbose        = verbose
        self.save_best_only = save_best_only
        self.best_loss      = float('inf')

    def __call__(self, loss, model, epoch):
        if self.save_best_only:
            if loss < self.best_loss:
                self.best_loss = loss
                torch.save(model.state_dict(), self.filepath)
                if self.verbose:
                    print(f"Epoch {epoch}: {self.monitor} improved from {self.best_loss} to {loss}, saving model to {self.filepath}")
        else:
            torch.save(model.state_dict(), self.filepath)
            if self.verbose:
                print(f"Epoch {epoch}: saving model to {self.filepath}")


def calculate_same_padding(input_size, kernel_size, stride):
    output_size = np.ceil(float(input_size) / float(stride))
    pad = ((output_size - 1) * stride + kernel_size - input_size) / 2
    return int(pad)

class ConvNN(nn.Module):
    def __init__(self, X_train_shape, nb_classes = 2):
        print(X_train_shape)
        super(ConvNN, self).__init__()
        # Layers here
        self.normal1 = nn.BatchNorm3d(num_features=X_train_shape[1])
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, 
                               kernel_size=(X_train_shape[2], 5, 5), stride=(1, 2, 2))
        # Example usage in your model's __init__ method
        
        pool1_padding = [
            calculate_same_padding(X_train_shape[2], 1, 1),  # Depth dimension
            calculate_same_padding(X_train_shape[3], 2, 2),  # Height dimension
            calculate_same_padding(X_train_shape[4], 2, 2)   # Width dimension
        ]

        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), padding=pool1_padding)

        self.ts = int(np.round(np.floor((X_train_shape[3]-4)/2) / 2 + 0.1))
        self.fs = int(np.round(np.floor((X_train_shape[4]-4)/2) / 2 + 0.1))

        self.normal2 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.normal3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flatten = nn.Flatten()
        self.drop1   = nn.Dropout(0.5)
        self.dens1   = nn.Linear(320,128)
        self.drop2   = nn.Dropout(0.5)
        self.dens2   = nn.Linear(128, nb_classes)

    def forward(self, x):
        # Forward pass here
        x = self.normal1(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        # x = x.view(x.size(0), 16, self.ts, self.fs)  # Reshape operation
        x = x.squeeze(2)

        x = self.normal2(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.normal3(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.drop1(x)
        x = F.sigmoid(self.dens1(x))  # Using sigmoid activation
        x = self.drop2(x)
        x = self.dens2(x)
        return F.softmax(x / 1.0, dim=1)  # Applying temperature to softmax

    def fit(self,
            batch_size, epochs, target, mode,
            X_train, Y_train, X_val=None, y_val=None):
        
        Y_train       = torch.tensor(Y_train).type(torch.LongTensor)
        X_train       = torch.tensor(X_train).type(torch.FloatTensor)
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader  = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            y_val       = torch.tensor(y_val).type(torch.LongTensor)
            X_val       = torch.tensor(X_val).type(torch.FloatTensor)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader  = DataLoader(dataset=val_dataset, batch_size=batch_size)


        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)

        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        checkpoint     = ModelCheckpoint(filepath=f"weights_{target}_{mode}.pth", 
                                         verbose=1, save_best_only=True)

        pbar = tqdm(total = epochs * len(train_loader))
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for X_batch, Y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.update(1)
            
            avg_loss = total_loss / len(train_loader)
            early_stopping(avg_loss)
            checkpoint(avg_loss, self, epoch)
            
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

            if X_val is not None and y_val is not None:
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, Y_batch in val_loader:
                        outputs = self(X_batch)
                        val_loss += criterion(outputs, Y_batch).item()
                    val_loss /= len(val_loader)

                # early_stopping(val_loss)
                # checkpoint(val_loss, self, epoch)

                # if early_stopping.early_stop:
                #     print("Early stopping triggered")
                #     break

        pbar.close()
        if os.path.exists(f"weights_{target}_{mode}.pth"):
            self.load_state_dict(torch.load(f"weights_{target}_{mode}.pth"))
            if mode == 'cv':
                os.remove(f"weights_{target}_{mode}.pth")
    
    def evaluate(self, X, y):
        self.eval()

        X_tensor = torch.tensor(X).float()
        y_tensor = torch.tensor(y).long()

        with torch.no_grad():
            predictions = self(X_tensor)

        predictions = predictions[:, 1].cpu().numpy()
        auc_test = roc_auc_score(y_tensor, predictions)
        print('Test AUC is:', auc_test)

