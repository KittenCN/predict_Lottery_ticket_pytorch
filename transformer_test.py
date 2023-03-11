import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(10):
    running_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        x, y = batch
        x = x.unsqueeze(1) # 将输入序列的最后一维扩展为 1
        y_pred = model(x.float())
        loss = criterion(y_pred, y.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    print(f"Epoch {epoch+1}: Loss = {running_loss / len(dataset):.4f}")
