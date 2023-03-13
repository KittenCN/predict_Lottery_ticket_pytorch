# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
from torch.utils.data import  Dataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_prob=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout_prob):
        super(Transformer_Model, self).__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        positional_encoded = self.positional_encoding(embedded)
        transformer_encoded = self.transformer_encoder(positional_encoded)
        linear_out = self.linear(transformer_encoded.mean(dim=1))
        return linear_out.squeeze(1)

def train_model(model, data, labels, num_epochs, batch_size, learning_rate, device):
    dataset = Data.TensorDataset(data, labels)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_data.shape[0]
        epoch_loss /= len(dataset)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))


# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, data, windows, cut_num):
        tmp = []
        for i in range(len(data) - windows):
            if cut_num > 0:
                sub_data = data[i:(i+windows+1), :cut_num]
            else:
                sub_data = data[i:(i+windows+1), cut_num:]
            tmp.append(sub_data)
        self.data = np.array(tmp)
    
    def __len__(self):
        return len(self.data) - 1
    
    def __getitem__(self, idx):
        # 将每组数据分为输入序列和目标序列
        x = self.data[idx][1:]
        y = self.data[idx][0]
        return x, y

# 定义 Transformer 模型类
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, num_layers=6, num_heads=10, dropout=0.001):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=input_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            dropout=dropout
        )
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = x.permute(1, 0, 2) # 将输入序列转置为 (seq_len, batch_size, input_size)
        x = self.transformer(x, x) # 使用 Transformer 进行编码和解码
        x = x.permute(1, 0, 2) # 将输出序列转置为 (batch_size, seq_len, input_size)
        x = self.linear(x) # 对输出进行线性变换
        x = x[:, -1, :]
        return x
