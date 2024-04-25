# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import torch.nn.functional as F
from torch.utils.data import  Dataset
from torch.optim.lr_scheduler import _LRScheduler

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

def binary_decode_array(binary_encoded_data, threshold=0.25, top_k=20):
    """
    Decode binary encoded data back to its original numerical representation,
    selecting the top_k classes with probabilities exceeding a given threshold.
    
    Parameters:
    - binary_encoded_data: A 2D tensor or array of binary encoded data with shape (windows_size, num_classes).
    - threshold: A float representing the cutoff threshold for determining whether a class is selected.
    - top_k: The number of highest probability classes to select after applying the threshold.
    
    Returns:
    - A list of lists, where each inner list contains the numbers of the top_k selected classes based on the threshold.
    """
    # sigmoid = torch.sigmoid(binary_encoded_data)  # Convert raw scores to probabilities
    sigmoid = binary_encoded_data
    windows_size, num_classes = sigmoid.shape
    decoded_data = []
    
    for i in range(windows_size):
        # Apply threshold and get indices of classes with probabilities above the threshold
        above_threshold_indices = (sigmoid[i] > threshold).nonzero(as_tuple=True)[0]
        if len(above_threshold_indices) > 0:
            # Get probabilities of classes above the threshold
            probs = sigmoid[i][above_threshold_indices]
            # Sort these probabilities and select the top_k
            top_k_indices = probs.topk(min(top_k, len(probs)), largest=True).indices
            selected_indices = above_threshold_indices[top_k_indices]
            # Adjust indices for 1-based numbering and append to the result
            decoded_row = (selected_indices + 1).tolist()
            decoded_data.append(decoded_row)
        else:
            # If no class probability exceeds the threshold, append an empty list
            decoded_data.append([])
    
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

def decode_one_hot(one_hot_encoded_data, sort_by_max_value=False, num_classes=80):
    """
    Decode one-hot encoded data back to its original numerical representation.
    
    Parameters:
    - one_hot_encoded_data: A 1D tensor or array of one-hot encoded data with length a multiple of 80.
    - sort_by_max_value: A boolean indicating whether to sort the output by the maximum value in each segment.
    
    Returns:
    - A list of decoded numbers, where each number corresponds to the position of 1 in each 80-length segment.
    """
    # Ensure the input is a torch tensor
    if not isinstance(one_hot_encoded_data, torch.Tensor):
        one_hot_encoded_data = torch.tensor(one_hot_encoded_data)
    
    # Check if the data length is a multiple of 80
    assert one_hot_encoded_data.numel() % num_classes == 0, "The total number of data points must be a multiple of 80."
    
    # Reshape the data to have shape (-1, 80), where each row is one 80-length segment
    reshaped_data = one_hot_encoded_data.view(-1, num_classes)
    
    # Decode each segment
    decoded_numbers = []
    max_values = []
    for segment in reshaped_data:
        # Find the index of the maximum value in each segment, adjust by 1 for 1-based indexing
        max_value, max_index = torch.max(segment, dim=0)
        decoded_number = max_index.item() + 1
        decoded_numbers.append(decoded_number)
        max_values.append(max_value.item())
    
    # Sort the decoded numbers by the maximum value in each segment if required
    if sort_by_max_value:
        decoded_numbers = [x for _, x in sorted(zip(max_values, decoded_numbers), reverse=True)]
    
    return decoded_numbers

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_prob=0.1, max_len=500):
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
    def __init__(self, input_size, output_size=20, hidden_size=512, num_layers=8, num_heads=16, dropout=0.1):
        super(Transformer_Model, self).__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, max_len=int(input_size*1.2))
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
        # embedded = self.dropout(embedded)
        positional_encoded = self.positional_encoding(embedded) 
        transformer_encoded = self.transformer_encoder(positional_encoded)  # (seq_len, batch_size, hidden_size)
        # transformer_encoded = self.dropout(transformer_encoded)
        linear_out = self.linear(transformer_encoded.mean(dim=0))
        # linear_out = torch.sigmoid(linear_out)
        return linear_out
    
class LSTM_Model(nn.Module): 
    def __init__(self, input_size, output_size=20, hidden_size=512, num_layers=1, num_heads=16, dropout=0.1, num_embeddings=20, embedding_dim=50):
        super(LSTM_Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings + 1, embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=embedding_dim*input_size, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(embedding_dim*input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden_size, 1)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM 层
        # x = x.view(x.size(0), x.size(1), -1)
        # x: [batch_size, seq_length, num_indices]
        x = self.embedding(x)  # [batch_size, seq_length, num_indices, embedding_dim]
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, num_indices, seq_length, embedding_dim]
        x = x.view(x.size(0), x.size(1), -1)  # [batch_size, num_indices, seq_length*embedding_dim]
        x = self.conv1d(x)  # [batch_size, num_channels, seq_length]
        x = x.permute(0, 2, 1)  # [batch_size, seq_length, num_channels]
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, input_size)
        lstm_out = self.dropout(lstm_out)
        # Applying attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        linear_out = self.linear(context_vector)
        # 取最后一个时间步的输出
        # lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        # linear_out = self.linear(lstm_out)  # (batch_size, output_size)
        # linear_out = torch.sigmoid(linear_out)
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

class CustomSchedule(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(CustomSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        arg1 = (self._step_count) ** -0.5
        arg2 = self._step_count * (self.warmup_steps ** -1.5)
        lr = (self.d_model ** -0.5) * min(arg1, arg2)
        return [lr for group in self.optimizer.param_groups]

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, data, windows, cut_num, model='Transformer', num_classes=80, test_flag=0, test_list=[]):
        tmp = []
        for i in range(len(data) - windows):
            if cut_num > 0:
                if len(test_list) <= 0 or (test_flag == 0 and data[i][1] not in test_list) or (test_flag == 1 and data[i][1] in test_list):
                    sub_data = data[i:(i+windows+1), 2:cut_num+2]
                    tmp.append(sub_data.reshape(windows+1, cut_num))
            else:
                if len(test_list) <= 0 or (test_flag == 0 and data[i][1] not in test_list) or (test_flag == 1 and data[i][1] in test_list):
                    sub_data = data[i:(i+windows+1), cut_num*(-1):]
                    tmp.append(sub_data.reshape(windows+1, data.shape[1]-(cut_num*(-1))))
        self.data = np.array(tmp)
        self.model = model
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.data) - 1
    
    def __getitem__(self, idx):
        # 将每组数据分为输入序列和目标序列
        x = torch.from_numpy(self.data[idx][1:][::-1].copy())
        y = torch.from_numpy(self.data[idx][0].copy()).unsqueeze(0)
        if self.model == 'Transformer':
            x_hot = binary_encode_array(x, self.num_classes) 
            y_hot = binary_encode_array(y, self.num_classes)
        elif self.model == 'LSTM':
            # x_hot = one_hot_encode_array(x, self.num_classes)
            y_hot = one_hot_encode_array(y, self.num_classes)
            x_hot = x - 1
            # y_hot = y - 1
        return x_hot, y_hot

# 定义 Transformer 模型类 (废除不用)
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size=20, hidden_size=1024, num_layers=8, num_heads=16, dropout=0.1):
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


class IoULoss(nn.Module):
    def __init__(self, smooth=1):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    def forward(self, preds, targets):
        # Convert logits to probabilities
        preds_probs = torch.sigmoid(preds)

        # Generate binary predictions by thresholding
        # preds_binary = (preds_probs > threshold).float()

        # Reduce across the position dimension by checking max, simulating any()
        # We directly use the output of torch.max operation to keep grad
        preds_reduced = torch.max(preds_probs, dim=1)[0]  # Correctly using max output
        targets_reduced = torch.max(targets, dim=1)[0]

        # Calculate intersection and union
        intersection = (preds_reduced * targets_reduced).sum(dim=1)
        union = (preds_reduced + targets_reduced - preds_reduced * targets_reduced).sum(dim=1)

        # Calculate IoU and IoU loss
        iou = intersection / (union + 1e-6)  # Adding a small epsilon to avoid division by zero
        iou_loss = 1 - iou.mean()  # Average IoU loss over the batch

        return iou_loss
