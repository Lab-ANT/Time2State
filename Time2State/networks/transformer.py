import torch
import math
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
        
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, out_dim) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, d_model)
        self.encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1,
            # device=device
        )
        self.positional_encoding = PositionalEncoding(input_dim, dropout=0.1)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=4)
        self.fc2 = nn.Linear(d_model, out_dim)
        self.reduce = torch.nn.AdaptiveMaxPool1d(1)

    # def forward(self, x):
    #     x = x.swapaxes(1, 2)
    #     x = self.positional_encoding(x)
    #     x = self.encoder(self.fc(x))
    #     x = x.swapaxes(1, 2)
    #     x = self.reduce(x)
    #     x = x.squeeze(2)
    #     x = self.fc2(x)
    #     # return x[:, :, -1]
    #     # print(x.shape)
    #     return x

    def forward(self, x):
        x = x.swapaxes(1, 2)
        x = self.positional_encoding(x)
        x = self.encoder(self.fc(x))
        x = self.fc2(x)
        x = x.swapaxes(1, 2)
        return x[:, :, -1]

# data = torch.ones(8, 16, 100)
# model = Transformer(16, 320, 2)
# out = model(data)
# print(out.shape)