import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
from torch.utils.data import  Dataset, DataLoader

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
    def __init__(self, batch_size, input_size, hidden_size, num_layers, num_heads, dropout_prob):
        super(Transformer_Model, self).__init__()

        self.embedding = nn.Embedding(batch_size * input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        embedded = self.embedding(x)
        positional_encoded = self.positional_encoding(embedded)
        transformer_encoded = self.transformer_encoder(positional_encoded)
        x = x.permute(1, 0, 2)
        linear_out = self.linear(transformer_encoded.mean(dim=1))
        return linear_out.squeeze(1)

def train_model(model, dataloader, num_epochs, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            batch_data = batch_data.unsqueeze(1)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_data.shape[0]
        epoch_loss /= len(dataset)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))


# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data) - 1
    
    def __getitem__(self, idx):
        # 将每组数据分为输入序列和目标序列
        x = self.data[idx]
        y = self.data[idx+1]
        return x, y

# 定义 Transformer 模型类
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=4, num_heads=4, dropout=0.1):
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
        return x

# 准备数据
data = torch.randint(1, 81, (1000, 20)) # 生成随机数据
dataset = MyDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义模型和优化器
model = TransformerModel(input_size=20, output_size=20)
# model = Transformer_Model(batch_size=32, input_size=20, hidden_size=256, num_layers=4, num_heads=8, dropout_prob=0.1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(10):
    running_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        x, y = batch
        x = x.unsqueeze(1) # 将输入序列的最后一维扩展为 1
        y = y.unsqueeze(1)
        y_pred = model(x.float())
        loss = criterion(y_pred, y.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    print(f"Epoch {epoch+1}: Loss = {running_loss / len(dataset):.4f}")

# train_model(model, dataloader, 10, 0.001, 'cpu')
