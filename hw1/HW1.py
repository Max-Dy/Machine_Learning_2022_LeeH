import math
import numpy as np
import pandas as pd
import os
import tensorflow as tf
# damn stupid today! debug for a long time when tf is not even installed.
import csv

import tensorboard
from sympy.utilities.iterables import runs
from tqdm import tqdm  # for showing of the

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # uncertainty
    np.random.seed(seed)  # same seed on np, torch, cuda
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    v_size = int(valid_ratio * len(data_set))
    t_size = len(data_set) - v_size
    t_set, v_set = random_split(data_set, [t_size, v_size],
                                generator=torch.Generator().manual_seed(seed))  # random_split
    return np.array(t_set), np.array(v_set)


def predict(test_loader, model, device):
    model.eval()
    preds = []
    for x in tqdm(test_loader):  # tqdm is for progress bar showing
        x = x.to(device)
        with torch.no_grad():  # no grad
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()  # connet in dim = 0
    return preds


# dataset
class COVID19Dataset(Dataset):
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


# model
class My_Model(nn.Module):
    def __init__(self, input_dim):  # here input dim refers to how many features in one training data, not batch size
        # question: how to work on a batch of training data
        # dataloder generate a batch of x and feed to the model
        super(My_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),  # x (3,1), W (1, 3), --- Wx (1, 1), fully connection and shrimped
            nn.ReLU(),  # unlinear change
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)  # here for each input, output is a scalar
            # for an input with batch size = B, then output goes (B, 1), so we need squeeze in forward func.
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x


# feature selection # the part to modified
def select_feature(t_data, v_data, test_data, select_all=True):
    y_t, y_v = t_data[:, -1], v_data[:, -1]
    row_x_t, row_x_v, row_x_test = t_data[:, :-1], v_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(row_x_t.shape[1]))  # train data include a batch
    else:
        feat_idx = [0, 1, 2, 3, 4]
    return row_x_t[:, feat_idx], row_x_v[:, feat_idx], row_x_test[:, feat_idx], y_t, y_v


# train

def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')  # 1
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)  # 2
    writer = SummaryWriter()  # to draw loss during trainning

    if not os.path.isdir('./models'):
        os.makedirs('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    for epoch in range(n_epochs):  # 3
        model.train()  # 4
        loss_record = []
        train_pbar = tqdm(train_loader, position=0, leave=True)  # leave:if save the bar after loop
        for x, y in train_pbar:  # 5
            optimizer.zero_grad()  # 5
            x, y = x.to(device), y.to(device)  # 6
            pred = model(x)  # 7
            loss = criterion(pred, y)  # 8
            loss.backward()  # 9
            optimizer.step()  # 10
            step += 1
            loss_record.append(loss.detach().item())

            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # validation in the same epoch
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():  # validation no_grade
                pred = model(x)
                loss = criterion(pred, y)
            loss_record.append(loss.detach().item())
        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train Loss: {mean_train_loss:.4f}, Valid Loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/Valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])
            print('Saving the model with loss {:.3f} ... '.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count >= config['early_stop']:
            print("\Model is not improving, halt the training")
            return


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314, 'select_all': True, "valid_ratio": 0.2, "n_epochs": 3000, "batch_size": 256, "learning_rate": 1e-6,
    'early_stop': 400, "save_path": './models/model.ckpt'
}

# load data
same_seed(config['seed'])
train_data, test_data = pd.read_csv('./covid.train.csv').values, pd.read_csv('./covid.test.csv').values
# .values---transfer DataFrame to nparray while get rid of listname&index
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

print(f"""
train_data_size: {train_data.shape}
valid_data_size: {valid_data.shape}
test_data_size: {test_data.shape}
""")
x_train, x_valid, x_test, y_train, y_valid = select_feature(train_data, valid_data, test_data, config['select_all'])
# to decide if you remove some data then see if it's better
print(f'number of features:{x_train.shape[1]}')
train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), COVID19Dataset(x_valid, y_valid), \
                                             COVID19Dataset(x_test)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

if __name__ == '__main__':
    model = My_Model(input_dim=x_train.shape[1]).to(device)
    trainer(train_loader, valid_loader, model, config, device)
