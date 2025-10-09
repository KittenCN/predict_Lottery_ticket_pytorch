# API 文档

## 模块说明

### src.config
配置文件模块，包含所有彩票类型的配置参数。

### src.modeling  
模型定义模块，包含Transformer和LSTM模型的实现。

### src.common
通用功能模块，包含数据处理、模型训练、预测等功能。

## 使用方法

### 数据获取
```bash
python scripts/get_data.py --name kl8 --cq 0
```

### 模型训练
```bash
python scripts/train_model.py --name kl8 --seq_len 5 --red_epochs 100
```

- 训练脚本会在内部将标签转换为**多热向量**，并使用 `BCEWithLogitsLoss`（或 FocalLoss）配合 Sigmoid 概率进行多标签优化，无需手动调整损失函数；
- `tests` 阶段默认复用与训练一致的多热语义，并通过 Top-K 命中率评估模型效果。

### 模型预测
```bash
python scripts/predict.py --name kl8 --seq_len 5 --model Transformer
```

- 预测脚本会先根据已保存的模型 ckpt、`modeling.extra_classes` 等元数据自动推导输入/输出维度；
- 输出结果基于 Sigmoid 概率，默认使用阈值 0.25 生成候选号码列表，并统计 Top-K 命中情况。

## 配置参数说明

| 参数 | 描述 | 默认值 |
|------|------|--------|
| name | 彩票类型 (kl8/ssq/dlt/pls/qxc) | kl8 |
| seq_len | 训练窗口大小 | 5 |  
| red_epochs | 红球训练轮数 | 100 |
| blue_epochs | 蓝球训练轮数 | 1 |
| batch_size | 批次大小 | 32 |
| hidden_size | 隐藏层大小 | 2560 |
| num_layers | 网络层数 | 6 |
| num_heads | 注意力头数 | 8 |
| model | 模型类型 (Transformer/LSTM) | Transformer |
