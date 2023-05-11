import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import sys
sys.path.append(os.path.dirname(__file__))
import networks
from TSpy.dataset import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def epoch_run(data, ds_estimator, auto_regressor, encoder, device, window_size, n_size=5, optimizer=None, train=True):
    if train:
        encoder.train()
        ds_estimator.train()
        auto_regressor.train()
    else:
        encoder.eval()
        ds_estimator.eval()
        auto_regressor.eval()
    encoder.to(device)
    ds_estimator.to(device)
    auto_regressor.to(device)

    epoch_loss = 0
    acc = 0
    for sample in data:
        rnd_t = np.random.randint(5*window_size,sample.shape[-1]-5*window_size)
        # rnd_t = np.random.randint(window_size,sample.shape[-1]-window_size)
        sample = torch.Tensor(sample[:,max(0,(rnd_t-20*window_size)):min(sample.shape[-1], rnd_t+20*window_size)])

        T = sample.shape[-1]
        windowed_sample = np.split(sample[:, :(T // window_size) * window_size], (T // window_size), -1)
        windowed_sample = torch.tensor(np.stack(windowed_sample, 0), device=device)
        encodings = encoder(windowed_sample)
        window_ind = torch.randint(2,len(encodings)-2, size=(1,))
        _, c_t = auto_regressor(encodings[max(0, window_ind[0]-10):window_ind[0]+1].unsqueeze(0))
        density_ratios = torch.bmm(encodings.unsqueeze(1),
                                       ds_estimator(c_t.squeeze(1).squeeze(0)).expand_as(encodings).unsqueeze(-1)).view(-1,)
        r = set(range(0, window_ind[0] - 2))
        r.update(set(range(window_ind[0] + 3, len(encodings))))
        rnd_n = np.random.choice(list(r), n_size)
        X_N = torch.cat([density_ratios[rnd_n], density_ratios[window_ind[0] + 1].unsqueeze(0)], 0)
        if torch.argmax(X_N)==len(X_N)-1:
            acc += 1
        labels = torch.Tensor([len(X_N)-1]).to(device)
        loss = torch.nn.CrossEntropyLoss()(X_N.view(1, -1), labels.long())
        epoch_loss += loss.item()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return epoch_loss / len(data), acc/(len(data))


def learn_encoder(encoder, x, window_size, out_channels, lr=0.003, decay=0, n_size=5, n_epochs=50, data='simulation', device='cpu', n_cross_val=1):
    encoding_size = out_channels
    encoder.encoding_size = out_channels
    ds_estimator = torch.nn.Linear(encoder.encoding_size, encoder.encoding_size)
    auto_regressor = torch.nn.GRU(input_size=encoding_size, hidden_size=encoding_size, batch_first=True)
    params = list(ds_estimator.parameters()) + list(encoder.parameters()) + list(auto_regressor.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)
    inds = list(range(len(x)))
    random.shuffle(inds)
    x = x[inds]
    # n_train = int(0.8*len(x))
    for epoch in range(n_epochs):
        # print(epoch)
        epoch_loss, acc = epoch_run(x, ds_estimator, auto_regressor, encoder, device, window_size, optimizer=optimizer,
                                        n_size=n_size, train=True)

# def main(lr=0.003, cv=1):
#     out_channels=2
#     window_size=128
#     encoder = networks.causal_cnn.CausalCNNEncoder(6, 30, 10, 80, out_channels, 3)

#     data_path = os.path.join(os.path.dirname(__file__), '../data/')
#     data, _ = load_USC_HAD(1, 1, data_path)
#     data = torch.tensor(data.T).unsqueeze(0)
#     T = data.shape[-1]
    
#     # print(data.shape)
#     windowed_data = np.concatenate(np.split(data[:, :, :T // 5 * 5], 5, -1), 0)
#     # print(windowed_data.shape)

#     learn_encoder(encoder, windowed_data, window_size, 2, n_epochs=10, lr=lr, decay=1e-5,  n_size=4,
#         device=device, data=None, n_cross_val=cv)
    
#     data, _ = load_USC_HAD(1, 1, data_path)
#     data = torch.tensor(data.T.astype(np.float32)).unsqueeze(0)
#     T = data.shape[-1]

#     step=10
#     num_batch, num_channel, length = np.shape(data)
#     num_window = int((length-window_size)/step)+1
#     # print('num_window', num_window)

#     windowed_data = []
#     i=0
#     for k in range(num_window):
#         windowed_data.append(data[:,:,i:i+window_size])
#         i+=step
#     batch_windowed_data = np.concatenate(windowed_data, 0)
#     batch_windowed_data = torch.Tensor(batch_windowed_data).to(device)
        
#     e_list = []
#     i = 0
#     while i < num_window:
#         representations = encoder(batch_windowed_data[i:i+10,:,:])
#         e_list.append(representations.detach().cpu().numpy())
#         i+=10
#     embeddings = np.vstack(e_list)

#     plt.scatter(embeddings[:,0], embeddings[:,1])
#     plt.savefig('emb2.png')

class CausalConv_CPC():
    def __init__(self, window_size, out_channels, in_channels):
        self.encoder = networks.causal_cnn.CausalCNNEncoder(in_channels, 30, 10, 80, out_channels, 3)
        self.encoder.encoding_size = out_channels
        self.out_channels = out_channels
        self.window_size = window_size
        self.in_channels = in_channels
        self.lr=0.03

    def fit_encoder(self, data, epoch=10):
        data = torch.tensor(data.T).unsqueeze(0)
        T = data.shape[-1]
        # print(data.shape)
        windowed_data = np.concatenate(np.split(data[:, :, :T // 5 * 5], 5, -1), 0)
        # windowed_data = np.concatenate([data,data])

        learn_encoder(self.encoder, windowed_data, self.window_size, self.out_channels, n_epochs=epoch, lr=self.lr, decay=1e-5,  n_size=4,
            device=device, data=None, n_cross_val=1)

    def encode(self, data, win_size, step):
        data = torch.tensor(data.T.astype(np.float32)).unsqueeze(0)
        num_batch, num_channel, length = np.shape(data)
        num_window = int((length-self.window_size)/step)+1
        # print('num_window', num_window)

        windowed_data = []
        i=0
        for k in range(num_window):
            windowed_data.append(data[:,:,i:i+self.window_size])
            i+=step
        batch_windowed_data = np.concatenate(windowed_data, 0)
        batch_windowed_data = torch.Tensor(batch_windowed_data).to(device)
        
        e_list = []
        i = 0
        while i < num_window:
            representations = self.encoder(batch_windowed_data[i:i+10,:,:])
            e_list.append(representations.detach().cpu().numpy())
            i+=10
        embeddings = np.vstack(e_list)
        return embeddings