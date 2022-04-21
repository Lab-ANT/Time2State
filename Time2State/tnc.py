"""
Temporal Neighborhood Coding (TNC) for unsupervised learning representation of non-stationary time series
"""

import torch
from torch.utils import data
import math
import sys
import os
sys.path.append(os.path.dirname(__file__))
import networks

import numpy as np
import random
# from networks import RnnEncoder, WFEncoder
from statsmodels.tsa.stattools import adfuller

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Discriminator(torch.nn.Module):
    def __init__(self, input_size, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.input_size = input_size
        self.model = torch.nn.Sequential(torch.nn.Linear(2*self.input_size, 4*self.input_size),
                                         torch.nn.ReLU(inplace=True),
                                         torch.nn.Dropout(0.5),
                                         torch.nn.Linear(4*self.input_size, 1))

        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        p = self.model(x_all)
        return p.view((-1,))

class TNCDataset(data.Dataset):
    def __init__(self, x, mc_sample_size, window_size, augmentation, epsilon=3, state=None, adf=True):
        super(TNCDataset, self).__init__()
        self.time_series = x
        self.T = x.shape[-1]
        self.window_size = window_size
        self.sliding_gap = int(window_size*25.2)
        self.window_per_sample = (self.T-2*self.window_size)//self.sliding_gap
        self.mc_sample_size = mc_sample_size
        self.state = state
        self.augmentation = augmentation
        self.adf = adf
        if not self.adf:
            self.epsilon = epsilon
            self.delta = 5*window_size*epsilon

    def __len__(self):
        return len(self.time_series)*self.augmentation

    def __getitem__(self, ind):
        ind = ind%len(self.time_series)
        t = np.random.randint(2*self.window_size, self.T-2*self.window_size)
        x_t = self.time_series[ind][:,t-self.window_size//2:t+self.window_size//2]
        # plt.savefig('./plots/%s_seasonal.png'%ind)
        X_close = self._find_neighours(self.time_series[ind], t)
        X_distant = self._find_non_neighours(self.time_series[ind], t)

        if self.state is None:
            y_t = -1
        else:
            y_t = torch.round(torch.mean(self.state[ind][t-self.window_size//2:t+self.window_size//2]))
        return x_t, X_close, X_distant, y_t

    def _find_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if self.adf:
            gap = self.window_size
            corr = []
            for w_t in range(self.window_size,4*self.window_size, gap):
                try:
                    p_val = 0
                    for f in range(x.shape[-2]):
                        p = adfuller(np.array(x[f, max(0,t - w_t):min(x.shape[-1], t + w_t)].reshape(-1, )))[1]
                        # print(f, max(0,t - w_t), min(x.shape[-1], t + w_t))
                        p_val += 0.01 if math.isnan(p) else p
                    corr.append(p_val/x.shape[-2])
                except:
                    corr.append(0.6)
            self.epsilon = len(corr) if len(np.where(np.array(corr) >= 0.01)[0])==0 else (np.where(np.array(corr) >= 0.01)[0][0] + 1)
            self.delta = 5*self.epsilon*self.window_size

        ## Random from a Gaussian
        t_p = [int(t+np.random.randn()*self.epsilon*self.window_size) for _ in range(self.mc_sample_size)]
        t_p = [max(self.window_size//2+1,min(t_pp,T-self.window_size//2)) for t_pp in t_p]
        x_p = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_p])
        return x_p

    def _find_non_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if t>T/2:
            t_n = np.random.randint(self.window_size//2, max((t - self.delta + 1), self.window_size//2+1), self.mc_sample_size)
        else:
            t_n = np.random.randint(min((t + self.delta), (T - self.window_size-1)), (T - self.window_size//2), self.mc_sample_size)
        x_n = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_n])

        if len(x_n)==0:
            rand_t = np.random.randint(0,self.window_size//5)
            if t > T / 2:
                x_n = x[:,rand_t:rand_t+self.window_size].unsqueeze(0)
            else:
                x_n = x[:, T - rand_t - self.window_size:T - rand_t].unsqueeze(0)
        return x_n


def epoch_run(loader, disc_model, encoder, device, w=0, optimizer=None, train=True):
    if train:
        encoder.train()
        disc_model.train()
    else:
        encoder.eval()
        disc_model.eval()
    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    encoder.to(device)
    disc_model.to(device)
    epoch_loss = 0
    epoch_acc = 0
    batch_count = 0
    for x_t, x_p, x_n, _ in loader:
        mc_sample = x_p.shape[1]
        batch_size, f_size, len_size = x_t.shape
        x_p = x_p.reshape((-1, f_size, len_size))
        x_n = x_n.reshape((-1, f_size, len_size))
        x_t = np.repeat(x_t, mc_sample, axis=0)
        neighbors = torch.ones((len(x_p))).to(device)
        non_neighbors = torch.zeros((len(x_n))).to(device)
        x_t, x_p, x_n = x_t.to(device), x_p.to(device), x_n.to(device)

        z_t = encoder(x_t)
        z_p = encoder(x_p)
        z_n = encoder(x_n)

        d_p = disc_model(z_t, z_p)
        d_n = disc_model(z_t, z_n)

        p_loss = loss_fn(d_p, neighbors)
        n_loss = loss_fn(d_n, non_neighbors)
        n_loss_u = loss_fn(d_n, neighbors)
        loss = (p_loss + w*n_loss_u + (1-w)*n_loss)/2

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        p_acc = torch.sum(torch.nn.Sigmoid()(d_p) > 0.5).item() / len(z_p)
        n_acc = torch.sum(torch.nn.Sigmoid()(d_n) < 0.5).item() / len(z_n)
        epoch_acc = epoch_acc + (p_acc+n_acc)/2
        epoch_loss += loss.item()
        batch_count += 1
    return epoch_loss/batch_count, epoch_acc/batch_count

def learn_encoder(x, encoder, window_size, w, lr=0.003, decay=0.005, mc_sample_size=20,
                  n_epochs=10, device='cuda', augmentation=1, cont=False, in_channels=4, out_channels=4):
    batch_size = 10

    # choose encoder.
    encoder = networks.causal_cnn.CausalCNNEncoder(
            in_channels, 30, 10, 80, out_channels, 3)
    encoder.encoding_size=out_channels

    # encoder = networks.rnn.RnnEncoder(hidden_size=100, in_channel=4, encoding_size=2, device=device,cell_type='LSTM')
    # encoder = WFEncoder(encoding_size=64).to(device)
    # encoder = RnnEncoder(hidden_size=100, in_channel=561, encoding_size=2, device=device)

    disc_model = Discriminator(encoder.encoding_size, device)
    params = list(disc_model.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)
    inds = list(range(len(x)))
    random.shuffle(inds)
    x = x[inds]

    for epoch in range(n_epochs):
        # print(epoch)
        trainset = TNCDataset(x=torch.Tensor(x), mc_sample_size=mc_sample_size,
                              window_size=window_size, augmentation=augmentation, adf=True)
        train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
        epoch_loss, epoch_acc = epoch_run(train_loader, disc_model, encoder, optimizer=optimizer,
                                          w=w, train=True, device=device)
    return encoder.to(device)

class CausalConv_TNC():
    def __init__(self, window_size, out_channels, in_channels):
        self.encoder = None
        self.out_channels = out_channels
        self.window_size = window_size
        self.in_channels = in_channels

    def fit_encoder(self, data, epoch, w):
        self.encoder = learn_encoder(data, self.encoder, w=w, lr=0.003, decay=1e-5, window_size=self.window_size, n_epochs=epoch,
            mc_sample_size=40, device=device, augmentation=5,
            in_channels=self.in_channels, out_channels=self.out_channels)
       
    def encode(self, X, win_size=512, step=50, window_batch_size=10):
        num_batch, num_channel, length = np.shape(X)
        num_window = int((length-self.window_size)/step)+1
        # print('num_window', num_window)

        windowed_data = []
        i=0
        for k in range(num_window):
            windowed_data.append(X[:,:,i:i+self.window_size])
            i+=step
        batch_windowed_data = np.concatenate(windowed_data, 0)
        batch_windowed_data = torch.Tensor(batch_windowed_data).to(device)
        # print(batch_windowed_data.shape)
        
        e_list = []
        i = 0
        while i < num_window:
            representations = self.encoder(batch_windowed_data[i:i+10,:,:])
            e_list.append(representations.detach().cpu().numpy())
            # print(representations.shape, i, num_window)
            i+=10
        embeddings = np.vstack(e_list)
        # print(embeddings.shape)
        return embeddings
        
    # def encode_window(self, X, win_size=512, step=10, window_batch_size=10):
    #     w_list = []
    #     i=0
    #     num_batch, num_channel, length = np.shape(X)
    #     num_window = 1000#int((length-self.window_size)/step)+1
    #     for k in range(num_window):
    #         w_list.append(X[:,:,i:i+self.window_size])
    #         i+=step
    #     x_window = np.concatenate(w_list, 0)
    #     print(x_window.shape)
    #     print(torch.Tensor(x_window).shape)
    #     dataset = torch.utils.data.TensorDataset(torch.Tensor(x_window), torch.Tensor(np.zeros((num_window,4,self.window_size))))
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

    #     embeddings = []
    #     for x,_ in dataloader:
    #         embeddings.append(self.encoder(x.to(device)).detach().cpu().numpy())
    #     embeddings = np.concatenate(embeddings, 0)
    #     return embeddings