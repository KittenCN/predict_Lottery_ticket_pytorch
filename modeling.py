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
import torch.nn.functional as F

import  os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def binary_encode_array(input_array, num_classes=80):
    """
    Convert an input array of shape (windows_size, seq_len) to a binary encoded tensor of shape (windows_size, num_classes).
    
    Parameters:
    - input_array: An array of shape (windows_size, seq_len), where each row represents a time step and each element in the row represents a selected number.
    - num_classes: The total number of possible classes (e.g., 1 to 80).
    
    Returns:
    - A binary encoded tensor of shape (windows_size, num_classes).
    """
    windows_size, seq_len = input_array.shape
    # Initialize a tensor of zeros with the desired output shape
    binary_encoded_array = torch.zeros((windows_size, num_classes), dtype=torch.float32)
    
    # Encode each number in the input_array
    for i in range(windows_size):
        for j in range(seq_len):
            number = input_array[i, j]
            if 1 <= number <= num_classes:
                binary_encoded_array[i, number - 1] = 1.0  # Adjust index for 0-based indexing
    
    return binary_encoded_array

def binary_decode_array(binary_encoded_data):
    """
    Decode binary encoded data back to its original numerical representation.
    
    Parameters:
    - binary_encoded_data: A 2D tensor or array of binary encoded data with shape (windows_size, num_classes).
    
    Returns:
    - A list of lists, where each inner list contains the numbers decoded from each row of the binary encoded data.
    """
    windows_size, num_classes = binary_encoded_data.shape
    decoded_data = []
    
    for i in range(windows_size):
        decoded_row = []
        for j in range(num_classes):
            if binary_encoded_data[i, j] == 1:
                decoded_row.append(j + 1)  # Adjust index for 1-based numbering
        decoded_data.append(decoded_row)
    
    return decoded_data

def one_hot_encode_array(input_array, num_classes=80):
    """
    Convert an input array of shape (windows_size, seq_len) to a one-hot encoded tensor of shape (windows_size, seq_len, num_classes).
    
    Parameters:
    - input_array: An array of shape (windows_size, seq_len), where each row represents a time step and each element in the row represents a selected number.
    - num_classes: The total number of possible classes (e.g., 1 to 80).
    
    Returns:
    - A one-hot encoded tensor of shape (windows_size, seq_len, num_classes).
    """
    windows_size, seq_len = input_array.shape
    # Initialize a tensor of zeros with the desired output shape
    one_hot_encoded_array = torch.zeros((windows_size, seq_len, num_classes), dtype=torch.float32)
    
    # Encode each number in the input_array
    for i in range(windows_size):
        for j in range(seq_len):
            number = input_array[i, j]
            if 1 <= number <= num_classes:
                one_hot_encoded_array[i, j, number - 1] = 1.0  # Adjust index for 0-based indexing
    
    return one_hot_encoded_array

def decode_one_hot(one_hot_encoded_data):
    """
    Decode one-hot encoded data back to its original numerical representation.
    
    Parameters:
    - one_hot_encoded_data: A 1D tensor or array of one-hot encoded data with length a multiple of 80.
    
    Returns:
    - A list of decoded numbers, where each number corresponds to the position of 1 in each 80-length segment.
    """
    # Ensure the input is a torch tensor
    if not isinstance(one_hot_encoded_data, torch.Tensor):
        one_hot_encoded_data = torch.tensor(one_hot_encoded_data)
    
    # Check if the data length is a multiple of 80
    assert one_hot_encoded_data.numel() % 80 == 0, "The total number of data points must be a multiple of 80."
    
    # Reshape the data to have shape (-1, 80), where each row is one 80-length segment
    reshaped_data = one_hot_encoded_data.view(-1, 80)
    
    # Decode each segment
    decoded_numbers = []
    for segment in reshaped_data:
        # Find the index of the maximum value in each segment, adjust by 1 for 1-based indexing
        decoded_number = torch.argmax(segment).item() + 1
        decoded_numbers.append(decoded_number)
    
    return decoded_numbers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, input_size, output_size=20, hidden_size=1024, num_layers=8, num_heads=16, dropout=0.1, d_model=128, windows_size=5):
        super(Transformer_Model, self).__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers)
        self.dropout = nn.Dropout(dropout)  # 添加 dropout 层
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.int() # (batch_size, windows_size, seq_len)
        x = x.view(x.size(0), -1) # (batch_size, windows_size * seq_len)
        embedded = self.embedding(x) #(batch_size, seq_len, hidden_size)
        embedded = embedded.permute(1, 0, 2) # (seq_len, batch_size, hidden_size)
        positional_encoded = self.positional_encoding(embedded) 
        transformer_encoded = self.transformer_encoder(positional_encoded)  # (seq_len, batch_size, hidden_size)
        # transformer_encoded = self.dropout(transformer_encoded)
        linear_out = self.linear(transformer_encoded.mean(dim=0))
        return linear_out

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

class CustomSchedule(object):
    def __init__(self, d_model, warmup_steps=4000, optimizer=None):
        super(CustomSchedule, self).__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.steps = 1
    
    def step(self):
        arg1 = self.steps ** -0.5
        arg2 = self.steps * (self.warmup_steps ** -1.5)
        self.steps += 1
        lr = (self.d_model ** -0.5) * min(arg1, arg2)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, data, windows, cut_num):
        tmp = []
        for i in range(len(data) - windows):
            if cut_num > 0:
                sub_data = data[i:(i+windows+1), :cut_num]
            else:
                sub_data = data[i:(i+windows+1), cut_num:]
            tmp.append(sub_data.reshape(windows+1, cut_num))
        self.data = np.array(tmp)
    
    def __len__(self):
        return len(self.data) - 1
    
    def __getitem__(self, idx):
        # 将每组数据分为输入序列和目标序列
        x = torch.from_numpy(self.data[idx][1:][::-1].copy())
        y = torch.from_numpy(self.data[idx][0].copy()).unsqueeze(0)
        x_hot = binary_encode_array(x) 
        y_hot = binary_encode_array(y)
        return x_hot, y_hot

# 定义 Transformer 模型类
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size=20, hidden_size=1024, num_layers=8, num_heads=16, dropout=0.1, d_model=128, windows_size=5):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=input_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)  # 添加 dropout 层
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = x.permute(1, 0, 2) # 将输入序列转置为 (seq_len, batch_size, input_size)
        x = self.transformer(x, x) # 使用 Transformer 进行编码和解码
        x = self.dropout(x)  # 在 Transformer 后添加 dropout
        x = x.permute(1, 0, 2) # 将输出序列转置为 (batch_size, seq_len, input_size)
        x = self.linear(x) # 对输出进行线性变换(batch_size, seq_len, output_size)
        x = x[:, -1, :] # 取最后一个时间步的输出作为模型的输出(batch_size, output_size)
        return x
