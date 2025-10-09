# 🎯 基于 PyTorch 的彩票预测系统

## ⚠️ 重要提醒
**彩票理论上属于完全随机事件，任何一种单一算法，都不可能精确的预测彩票结果！！**  
**请合理投资博彩行业，切勿成迷！！**

## 📖 项目简介
基于 Transformer 模型的彩票预测系统，源自 [基于tensorflow lstm模型的彩票预测](https://github.com/KittenCN/predict_Lottery_ticket) 项目，采用 PyTorch 重构并优化了所有核心代码。

## 📁 项目结构
```
predict_Lottery_ticket_pytorch/
├── src/                     # 核心源代码
│   ├── __init__.py         # 包初始化文件
│   ├── config.py           # 配置文件
│   ├── common.py           # 通用功能模块
│   └── modeling.py         # 模型定义模块
├── scripts/                # 可执行脚本
│   ├── get_data.py         # 数据获取脚本
│   ├── train_model.py      # 模型训练脚本
│   └── predict.py          # 预测脚本
├── tests/                  # 测试文件
│   ├── test_config.py      # 配置测试
│   └── test_modeling.py    # 模型测试
├── examples/               # 示例代码
│   └── DQN_test.ipynb      # Jupyter notebook示例
├── docs/                   # 文档
│   ├── README.md           # 详细文档
│   └── API.md              # API文档
├── data/                   # 数据目录
├── model/                  # 模型保存目录
├── results/                # 结果输出目录
├── .github/workflows/      # GitHub Actions CI/CD
├── requirements.txt        # 依赖包列表
├── README.md               # 项目说明
└── AGENTS.md              # 代理配置文件
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 获取数据
```bash
python scripts/get_data.py --name kl8 --cq 0
```

网络抓取说明：爬虫默认只允许配置中的受信任域名，可在 `src/config.py` 中调整 `HTTP_ALLOWLIST`，并可通过 `HTTP_CACHE_DIR` 指定本地缓存目录以在网络不可用时回退。

### 3. 训练模型
```bash
python scripts/train_model.py --name kl8 --seq_len 5 --red_epochs 100
```

### 4. 执行预测
```bash
python scripts/predict.py --name kl8 --seq_len 5 --model Transformer
```

## � Smoke test 与 Pipeline 快速示例

建议在改动训练/数据逻辑时先运行 smoke test 验证前向/反向与 checkpoint 保存加载链路：

```bash
python -m pytest tests/test_smoke_train.py -q
```

新增的集成 smoke 测试（更完整的前向/反向 + checkpoint roundtrip）：

```bash
pytest tests/test_smoke_integration.py -q
```

新增了一个轻量 Pipeline 帮助类 `src.pipeline.LotteryPipeline`，用于短期内减少对全局可变状态的依赖（非破坏性）：

示例：

```python
from src.pipeline import LotteryPipeline
pipe = LotteryPipeline()
pipe.set_args(args)  # 可选
ds = pipe.create_dataset('kl8', windows=5, dataset=1, ball_type='red')
# 训练、保存
pipe.save_checkpoint('path.ckpt', model, optimizer=opt, epoch=0)
# 加载时会自动恢复 extra_classes
ck = pipe.load_checkpoint('path.ckpt')
```


## �📊 项目进度

✅ **已完成功能：**
- 按论文复现并优化了 Transformer/LSTM 模型
- 完成基本的训练代码，已测试快乐8
- 完成基本的预测代码
- 增加 one-hot 编码方式和混合 one-hot 模式
- 重写了 loss function
- 模型参数保存至 ckpt 文件，支持断点续训
- 增加双向 LSTM 模型，集成嵌入层、卷积层、多头注意力机制
- 增加数据特征计算和归一化处理
- 项目结构重构，符合标准 Python 包结构

🔧 **待解决问题：**
- LSTM 模型复杂度有待提升
- 模型有时会输出历史数据而非预测数据
- 需要更多有效的特征工程方法

📈 **当前效果（20240505测试）：**
- 在特定参数下训练，loss 控制在 10% 以下
- 10 天测试结果：平均命中率 30%，最高 45%，最低 15%

## 🛠️ 技术栈

- **深度学习框架**: PyTorch
- **模型类型**: Transformer, LSTM
- **数据处理**: Pandas, NumPy
- **可视化**: TensorBoard
- **日志记录**: Loguru
- **并发处理**: prefetch_generator

## 📝 使用说明

详细的 API 文档和使用示例请参考 [docs/API.md](docs/API.md)。

## 🤝 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)  
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## ⚖️ 免责声明

本项目仅用于技术研究和学习目的。彩票预测存在极大的不确定性，请理性对待，切勿用于实际投资决策。
