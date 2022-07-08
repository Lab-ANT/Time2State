import torch
import sys
sys.path.append('Baselines/ts2vec')
from models import TSEncoder
encoder = TSEncoder(4, 2, hidden_dims=64, depth=10, mask_mode='binomial')
data = torch.ones(10,512,4)
out = encoder(data)
print(out.size())