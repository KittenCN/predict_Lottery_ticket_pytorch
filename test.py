import torch
import torch.nn as nn
from torch.autograd import Variable

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        iou = intersection / (union + 1e-6)  # 添加一个小的常数，避免除零错误
        loss = 1 - iou
        return loss

pred = torch.randn(5, 3, 256, 256).float() # 假设有5个样本，每个样本有3个类别，图片大小为256x256
pred = Variable(pred, requires_grad=True) # 将pred转换为Variable类型
target = torch.randn(5, 3, 256, 256).float()  # 假设每个样本的真实标签也是3个类别，大小为256x256
loss_fn = IoULoss() # 创建IoU损失函数对象
loss = loss_fn(pred, target) # 计算IoU损失值
loss.backward() # 反向传播
print(loss) # 输出损失值