基于transformer模型的彩票预测
源自我另一个项目[基于tensorflow lstm模型的彩票预测](https://github.com/KittenCN/predict_Lottery_ticket),按照transformer的特性重构了所有关键代码
目前进度：
1. 按论文复现并优化了transformer/LSTM模型
2. 完成基本的训练代码，已经测试快乐8和双色球，理论上config里面配置好的项目，均可以支持。
3. 完成基本的预测代码，理论上config里面配置好的项目，均可以支持。
4. 增加one hot 编码的方式
5. 增肌混合one hot模式
6. 尝试重写loss func
7. 大部分参数均已保存至ckpt文件，读取并继续训练时将自动更新状态

## 目前项目还处在初步阶段，愿意尝试的朋友，请自行研究和修改，且不保证以后会不会进行重构
