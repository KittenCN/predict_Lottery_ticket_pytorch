# `彩票理论上属于完全随机事件，任何一种单一算法，都不可能精确的预测彩票结果！！`  
# `请合理投资博彩行业，切勿成迷！！`  
--
基于transformer模型的彩票预测
源自我另一个项目[基于tensorflow lstm模型的彩票预测](https://github.com/KittenCN/predict_Lottery_ticket),按照transformer的特性重构了所有关键代码
目前进度：
1. 按论文复现并优化了transformer/LSTM模型
2. 完成基本的训练代码，已经测试快乐8 ~~和双色球，理论上config里面配置好的项目，均可以支持~~ 。
3. 完成基本的预测代码 ~~，理论上config里面配置好的项目，均可以支持~~ 。
4. 增加one hot 编码的方式
5. 增加混合one hot模式
6. 尝试重写loss func
7. 大部分参数均已保存至ckpt文件，读取并继续训练时将自动更新状态
8. 增加双向lstm模型, 并在模型中，增加嵌入层，卷积层，多头注意力机制，以期待更好的效果
9. 部分参数优化
10. 增加小部分数据特征计算，并归一化，以期待更好的效果
11. 尝试增加一些新的特征计算方式和新的算法：
    - [x] Domain-Aware Features
        - [x] Hotness
        - [x] Cold-Decay
        - [x] Trend Index
    - [ ] Set Transformer / DeepSets
    - [ ] Lightweight Models + Regularization
        - [ ] Logistic Regression
        - [ ] Random Forest
        - [ ] XGBoost / LightGBM
    - [ ] Simulation & Monte Carlo
    - [ ] Reinforcement Learning


目前碰到的问题：
1. 现有的lstm模型还是觉得有点简单，主要体现在GPU资源吃不满。
2. ~~有时候模型会偷懒，给出之前的历史数据，而不是预测的数据。这个问题很奇怪，时有时无。~~
3. 期望有更多有效，且容易实现的特征计算。

目前效果：  
20240505：  
1. 在特定参数下反复训练，将loss压制在10%以下，经过10天的测试，平均命中率为6/20（30%），最大命中9/20（45%），最低命中3/20（15%）。
2. 特定参数原因，无法进行回测, 只能逐轮测试。

## 目前项目还处在初步阶段，愿意尝试的朋友，请自行研究和修改，且不保证以后会不会进行重构
