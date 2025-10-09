# `彩票理论上属于完全随机事件，任何一种单一算法，都不可能精确预测彩票结果！`  
# `请合理投资博彩行业，切勿沉迷！`  
---

## 项目简介
本项目基于 Transformer / LSTM 模型构建彩票号码的序列预测系统，源于开源仓库 [predict_Lottery_ticket](https://github.com/KittenCN/predict_Lottery_ticket) 并以 PyTorch 重新实现。核心代码按照标准 Python 包结构拆分至 `src/`、`scripts/`、`tests/`、`docs/` 等目录，便于二次开发与自动化集成。

### 当前功能
1. 复现并优化 Transformer / LSTM 结构，包含嵌入层、卷积层、多头注意力等模块组合；
2. 提供训练 (`scripts/train_model.py`)、预测 (`scripts/predict.py`) 与数据抓取 (`scripts/get_data.py`) 脚本，全部支持命令行参数配置；
3. 自动化保存 / 恢复模型 ckpt，包含优化器、学习率调度器与 AMP 状态，可断点续训；
4. 引入丰富的特征工程（连续值、间隔、奇偶比等），并按需要扩展至多彩种数据；
5. 测试覆盖基础配置与核心建模函数，方便快速回归。

### 最新更新（2025-10）
- 训练流程统一将开奖号码转换为多热向量，默认使用 `BCEWithLogitsLoss`（可切换 Focal Loss）配合 Sigmoid 概率训练，解决多标签语义与 CrossEntropy 不匹配的问题；
- 验证与预测阶段沿用同一概率语义，基于阈值与 Top-K 指标统计命中率；
- `run_predict` / `predict_ball_model` 会从数据集与 checkpoint 中读取 `extra_classes`、模型超参等信息自动推导输入输出维度，CLI 无需再手工参与计算。

## 已知问题
1. 现有 LSTM 结构仍偏简单，对 GPU 利用率不足；
2. 某些场景下模型可能输出历史号码，需进一步排查数据与损失设计；
3. 有待扩展更多有效、易实现的特征工程，或考虑引入序列增强策略。

## 当前效果（2024-05-05）
在固定参数下多次训练可将 loss 压至 10% 以下；连续 10 天测试时，平均命中率约 6/20（30%），最好 9/20（45%），最低 3/20（15%）。因参数限制暂无法整体回测，仅支持逐轮验证。

## 参与开发
- 参考 `MIGRATION.md` 获取目录与命令迁移说明；
- 运行 `python scripts/get_data.py --name kl8 --cq 0` 获取样例数据；
- 使用 `python scripts/train_model.py --name kl8 --seq_len 5 --red_epochs 100` 训练模型；
- 使用 `python scripts/predict.py --name kl8 --seq_len 5 --model Transformer` 进行预测；
- 建议在提交前执行 `python -m compileall scripts/` 或 `make test` 进行快速校验。

> 本项目仅供技术研究，任何预测结果请勿用于实际投注或投资决策。
