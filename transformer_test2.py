import math
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, src):
        src = self.pos_encoder(src)
        src = self.encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def batch_data(data, batch_size=10):
    num_batches = len(data) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        yield data[start_idx:end_idx]

# 超参数
input_size = 20
hidden_size = 256
output_size = 20
num_layers = 2
num_heads = 4
dropout = 0.2
learning_rate = 0.001
num_epochs = 100

# 数据
data = [[torch.randint(1, 81, size=(20,), dtype=torch.float)] for i in range(100)]
batched_data = list(batch_data(data))

# 模型
model = TransformerModel(input_size, hidden_size, output_size, num_layers, num_heads, dropout)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
for epoch in range(num_epochs):
    total_loss = 0
    for batch in batched_data:
        # 准备数据
        src = torch.cat(batch[:-1], dim=0).unsqueeze(1)
        tgt = batch[-1].unsqueeze(1)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output, tgt)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # 打印损失
    print("Epoch {} Loss {:.4f}".format(epoch+1, total_loss/len(batched_data)))
