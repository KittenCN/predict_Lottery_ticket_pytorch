# 4. 模型结构、激励与损失函数创新方向（2025-10-10）

为进一步提升模型对信号的捕捉能力，建议从以下几个方向探索：

## 4.1 损失函数创新
- **Focal Loss**：对 BCEWithLogitsLoss 的改进，提升对难分类样本的关注度，适合类别极度不均衡场景。可直接在 `src/common.py` 实现 FocalLoss 类，并在训练脚本中切换。
- **标签平滑（Label Smoothing）**：在 BCE/CE 损失基础上引入标签平滑，缓解过拟合与过置信，提升泛化。
- **Top-K Loss/Ranking Loss**：直接优化 Top-K 命中率或排序指标（如 NDCG、MAP），可用 differentiable ranking loss 或 soft top-k。
- **自定义奖励函数**：结合历史命中率、号码分布等业务先验，设计 reward shaping（如命中历史未出号奖励、连号惩罚等），通过 RL 或加权 loss 实现。

## 4.2 激励机制与正则化
- **稀疏激活（Sparsity Regularization）**：在输出层加 L1/L0 正则，鼓励模型输出更稀疏的概率分布，提升 Top-K 预测准确性。
- **多任务学习**：引入辅助任务（如预测和值、奇偶比、连号数等），通过多头输出联合训练，提升主任务泛化。
- **对抗训练**：引入 adversarial perturbation（如 FGSM），提升模型鲁棒性。
- **DropBlock/Spatial Dropout**：替换常规 Dropout，提升特征选择能力。

## 4.3 模型结构探索
- **注意力可解释性增强**：输出 attention map，分析模型关注的历史期号与特征，辅助调参与特征工程。
- **图神经网络（GNN）**：将号码共现关系建模为图结构，捕捉高阶组合信号。
- **混合专家（Mixture of Experts）**：对不同特征子空间或历史窗口分配专家网络，提升复杂模式建模能力。
- **概率建模/贝叶斯方法**：引入不确定性估计，输出置信区间，辅助决策。

## 4.4 实验与评估建议
- 每次引入新损失/激励/结构，建议先在小样本上做 ablation study，记录 loss 曲线与 Top-K 指标变化。
- 结合 TensorBoard/CSV 日志，系统性记录实验参数与结果，便于回溯。

> 所有上述思路均可分阶段落地，优先推荐 Focal Loss、标签平滑与多任务辅助头，后续可逐步引入 GNN/专家模型等结构创新。
# system_analysis — 变更与风险修复记录

本文件记录项目中高风险项的分析与修复状态，供 review 与后续迭代使用。

已修复项：
- checkpoint 中增加并恢复 extra_classes：训练保存与加载时会写入 `extra_classes`，加载时会恢复到 `src.modeling.extra_classes`，避免训练/预测时特征维度不一致导致 shape 错误。
- 请求超时与重试：对爬虫与获取最新期号的 network 请求增加了 `timeout` 和一次重试保障，减少临时网络问题导致脚本中断。
- 增加单元测试：新增 `tests/test_checkpoint_extra.py`，用于验证 checkpoint 中 `extra_classes` 的保存/恢复行为，同时修正测试导入路径以便在本仓库直接运行测试。

部分已缓解但需后续跟进的项：
- 损失函数与标签编码一致性（高风险）：训练代码已默认使用 `BCEWithLogitsLoss` 并在训练流程中将标签转换为 multi-hot（见 `scripts/train_model.py` 中 `to_multi_hot`），这已在训练路径中被使用，但建议补充更全面的单测（例如小批量前向/反向）来覆盖边界情况与 label 缺失场景。
- 全局状态耦合：项目中仍有全局变量（`ori_data`、`mini_args`、`modeling.extra_classes` 等）。已在 checkpoint/load 流程中恢复 `extra_classes`，并在 `src.common.init()` 中提供重置函数。长期建议将这些状态封装进 Pipeline/Context 类以减少隐藏依赖。
- 全局状态耦合（已进行中）：已逐步将模块级 `ori_data` 与 `mini_args` 的责任迁移到 `src.pipeline.LotteryPipeline`：
  - `ori_data` 的缓存已移入 `LotteryPipeline.get_ori_data`（支持 per-key 缓存、TTL 与并发锁），默认 TTL=300s（可在 `src.config` 中配置 `ORI_DATA_TTL`）。
  - `mini_args` 与 `setMiniargs` 已移除。所有调用方应直接使用 `src.pipeline.DEFAULT_PIPELINE.set_args(args)` 或显式将 args 传入相关函数。未设置 pipeline args 的调用将抛出错误以确保显式注入。
  - 迁移策略：按小批次替换调用方（scripts 与模块）以使用 pipeline 单例或显式传参，运行测试以确保兼容性。已完成对 `src.common` 的主干修改，脚本 `scripts/train_model.py` 与 `scripts/predict.py` 已开始使用 pipeline。

未修复的建议（需要额外工时）：
- 将 scripts 转为可安装的 console_scripts entry points（当前通过 sys.path hack 支持）。建议发布包或使用 editable install 来简化运行环境。

如何验证本次修复：
1. 运行新增的单元测试：
   - python tests/test_checkpoint_extra.py
2. 运行新增的 smoke 集成测试：
  - pytest tests/test_smoke_integration.py -q
2. 运行已有测试：
   - python tests/test_config.py
   - python tests/test_modeling.py
3. 运行数据采集确认网络逻辑：
   - python scripts/get_data.py --name kl8 --cq 0

附加验证：
- 检查配置项 `src.config.HTTP_ALLOWLIST` 与 `src.config.HTTP_CACHE_DIR` 已存在，并根据需要调整允许域名或缓存目录。
- 当网络不可达时，爬虫将尝试使用本地缓存（若存在），或按照指数退避后失败并抛出异常以便上层处理。

下一步建议：
- 为训练流程添加一个小规模 smoke test（创建随机数据，做一次前向和后向传播，检查 loss 可计算且 checkpoint 无错误）。
- 逐步重构：将全局状态注入到函数签名或 Context 对象中，移除模块级可变状态。
- 在爬虫中引入更完善的重试策略（如 tenacity），并对关键外部域名使用配置化白名单。

近期变更（2025-10-09）：
- 新增 `src.pipeline.LotteryPipeline` 的 `run_predict` 与 `predict_ball_model` wrapper，脚本 (`scripts/predict.py`, `scripts/train_model.py`, `scripts/get_data.py`) 已开始迁移以通过 pipeline.set_args(args) 注入参数，减少对 `mini_args` 的直接依赖。
 - 新增 `src.pipeline.LotteryPipeline` 的 `run_predict` 与 `predict_ball_model` wrapper，脚本 (`scripts/predict.py`, `scripts/train_model.py`, `scripts/get_data.py`) 已开始迁移以通过 pipeline.set_args(args) 注入参数，减少对 `mini_args` 的直接依赖。`get_ori_data` 现在实现了并发锁和 TTL 缓存，支持 `refresh=True` 强制刷新。
- 已添加 `tests/test_pipeline.py` 验证 pipeline 的 save/load extra_classes 行为与 create_dataset 返回基本对象。

更新时间: 2025-10-09
# 项目技术分析报告

## 1. 技术原理概述

### 1.1 数据采集与预处理
- `scripts/get_data.py` 通过调用 `src.common.get_data_run` 触发爬虫流程，针对不同彩种选择 `spider` / `spider_cq`（`src/common.py` 第169-280行），从公开数据站点抓取开奖历史并落地到 `data/<彩种>/` 目录的 CSV 文件。
- `get_current_number` 与 `get_url` 负责解析开奖期号和历史数据接口，便于训练阶段判断增量数据。
- 数据文件加载由 `create_train_data` 完成（`src/common.py` 第45-143行），函数支持训练/测试切分、窗口滑动与按期号抽样，并在首次读取后缓存全局 `ori_data` 以降低磁盘 IO。

### 1.2 数据集构建与特征工程
- `modeling.MyDataset`（`src/modeling.py` 第322-520行）在滑动窗口内构造样本：输入包含历史开奖数字及其派生特征（号码连续性、相邻间隔、奇偶/大小比、质合比、统计量等），并将特征拼接在原始号码之后，维度增量通过全局变量 `extra_classes` 暴露给训练与预测脚本。
- `create_train_data` 在 `dataset=1` 时返回该 PyTorch Dataset，并按红/蓝球区分样本，预测阶段会复用同一构造逻辑确保特征对齐。

### 1.3 模型结构
- **Transformer_Model**（`src/modeling.py` 第194-231行）：先使用全连接层将输入映射到隐层维度，再叠加位置编码，经过 `nn.TransformerEncoder` 若干层处理后，做时序平均池化并输出到号码维度，适合建模长序列依赖。
- **LSTM_Model**（`src/modeling.py` 第233-286行）：将号码索引嵌入后与手工特征拼接，送入双向 LSTM，使用 `nn.MultiheadAttention` 做上下文聚合，最终通过全连接层输出，旨在同时吸收统计特征与序列模式。
- `binary_encode_array` / `decode_one_hot` 等辅助函数支持将号码序列与概率向量互转，配合多标签评估。

### 1.4 训练与评估流程
- `scripts/train_model.py` 的 `train_ball_model`（第193-520行）负责训练红球/蓝球两个子模型，采用 `DataLoaderX` + `prefetch_generator` 异步加载数据，启用 AMP（`torch.amp.GradScaler`, `autocast`）与自定义学习率调度器 `CustomSchedule`（`src/modeling.py` 第308-318行），并按固定周期保存 `ckpt`。
- 评估逻辑在每个保存周期对验证集计算 loss、Top-K 命中率（`scripts/train_model.py` 第362-408行），同时记录最优模型用于后续推理。

### 1.5 预测与结果输出
- `scripts/predict.py` 读取最新模型，调用 `src.common.run_predict` 创建预测所需数据集，随后在内部根据 `modeling.extra_classes` 自动推导输入维度，使用 `binary_decode_array` 解析候选号码并将命中率写入 `results/<日期>.txt`。
- `write_strings_to_file`（`src/common.py` 第525-542行）负责输出预测摘要；`prefetch_generator` 与 `loguru` 提供性能与日志支撑。

## 2. 当前问题与风险分析

### 2.1 损失函数与标签编码不匹配（高）
- **表现**：`train_ball_model` 当前使用 `nn.CrossEntropyLoss`（`scripts/train_model.py` 第238行），但目标张量是多标签集合（每期多个开奖号码），并且在训练循环里被转换为浮点张量（第337-346行），与 Cross Entropy 所需的单标签长整型格式不符，容易导致梯度失真甚至运行时错误。
- **方案 A（推荐）**：改用 `nn.BCEWithLogitsLoss` 或现有的 `FocalLoss`（`src/common.py` 第26-43行），并将目标转换为多热编码向量。
  - 优点：契合多标签任务，兼容当前 Top-K 评估方式；实现改动小，只需在 Dataset 中输出 multi-hot 标签。
  - 缺点：需要同步调整预测阶段的阈值与指标统计以匹配 logits。
- **方案 B**：保持 Cross Entropy，但将每个位置视为单独的分类任务（输出 reshape 为 `[batch, 序列长度, num_classes]`，目标改为整数索引）。
  - 优点：能利用位置条件分布，便于对不同号码位定义约束。
  - 缺点：改动面大，需重构模型输出维度、数据集标签结构以及评估逻辑。
- **结论**：优先执行方案 A，确保损失与标签语义一致，再视效果决定是否进一步细化为位置建模。
- **进展**：`scripts/train_model.py` 已改用多热标签配合 `nn.BCEWithLogitsLoss`，并在训练/评估过程中统一通过 Sigmoid 概率统计 Top-K 准确率；预测阶段亦按相同标签语义输出结果。

### 2.2 训练与预测阶段特征维度不一致（高）
- **表现**：`MyDataset` 在构造样本时动态扩展特征维度并通过全局 `modeling.extra_classes` 暴露，`scripts/predict.py` 在调用 `run_predict` 前即根据该全局变量计算模型输入大小（第45-63行），此时 `extra_classes` 尚未被赋值，导致预测阶段实例化的模型与训练时的真实维度不符；`predict_ball_model` 在加载旧权重时又尝试用 `seq_len * hidden_size` 作为输入维度（`src/common.py` 第368-384行），进一步放大错配风险。
- **方案 A（推荐）**：将 `extra_classes`、输入/输出维度等信息写入 checkpoint，并在 `predict.py` 读取权重后复用存档配置，同时在推理前调用一次 `create_train_data(..., dataset=1, ...)` 以刷新 `extra_classes`，避免手工推断。
  - 优点：训练、预测配置强一致，可兼容历史模型；错误可以在加载阶段提前暴露。
  - 缺点：需要调整保存/加载逻辑，补充兼容分支处理旧模型缺少元信息的情况。
- **方案 B**：去除动态特征，仅保留原始号码序列，令输入维度固定。
  - 优点：实现简单，避免全局状态。
  - 缺点：损失已有特征工程成果，预测效果可能下降。
- **结论**：实施方案 A，补充 checkpoint 元数据与加载流程，优先保证模型成功复现训练态。
- **进展**：`run_predict` 现于内部创建数据集后再依据 `modeling.extra_classes` 推导输入维度，并由 `predict_ball_model` 在加载 checkpoint 时复用该配置，消除 CLI 计算维度与真实模型不一致的问题。

### 2.3 全局状态耦合导致流程不透明（中）
- **表现**：`src/common.py` 中的 `ori_data`、`pred_key_d`、`mini_args`、`extra_classes` 等均为模块级变量，训练与预测脚本通过隐式副作用共享状态，增加复现难度，也使得并行运行多彩种或多窗口时容易出现数据串扰。
- **方案 A**：封装 `LotteryPipeline` 类管理数据、特征与配置，将原有全局变量改为实例属性，并在脚本入口创建上下文对象。
  - 优点：隔离状态、提升可测试性，可在单元测试中注入假数据。
  - 缺点：重构工作量较大，需要梳理多处函数签名。
- **方案 B**：保留全局结构，但引入显式的 `reset()` 方法，在每次训练/预测前重置全局变量，并通过参数显式传递关键配置。
  - 优点：改动较小，短期即可缓解串扰。
  - 缺点：仍然存在多个调用者共享状态的风险。
- **结论**：短期采用方案 B 保底，长期逐步迁移到面向对象的封装模式。

### 2.4 数据采集健壮性不足（中）
- **表现**：`spider`/`spider_cq` 请求第三方站点时未设置超时、重试与失败降级，并直接将 HTML 解析结果写入 CSV；若对方站点结构变动或网络抖动，训练流程会直接失败。
- **方案 A（推荐）**：为 `requests.get` 增加超时与重试机制，捕获异常后记录友好日志，同时提供本地缓存或示例数据作为回退。
  - 优点：提升稳定性，便于本地开发与 CI 使用。
  - 缺点：需要引入简单的重试控制（可复用 `urllib3.Retry` 或自定义装饰器）。
- **方案 B**：将数据采集拆分为独立脚本或服务，训练脚本改为仅消费规范化的数据文件。
  - 优点：边界清晰，训练代码不再依赖网络。
  - 缺点：需额外维护数据服务或同步机制。
- **结论**：先落实方案 A，确保当前流程可用，再评估是否演进到方案 B。

### 2.5 训练监控与实验记录缺失（中）
- **表现**：当前仅在终端日志中输出 loss 和命中率，缺乏系统化的实验追踪、超参记录与可视化；`tensorboard` 开关默认关闭且路径固定，容易遗漏。
- **方案 A**：引入标准化的实验记录（如 MLflow、Weights & Biases 或自建 CSV 日志），同步记录数据版本、超参、模型指标。
  - 优点：便于比较实验结果、自动生成报告。
  - 缺点：需要维护额外的依赖或服务。
- **方案 B**：至少在现有日志中输出关键超参、样本量、最优模型路径，并默认开启 TensorBoard。
  - 优点：改动小，快速提升可观测性。
  - 缺点：仍然依赖人工整理。
- **结论**：先执行方案 B，形成最低可用的监控手段，再按团队资源决定是否接入专业实验平台。

## 3. 后续开发方向
- **配置与环境统一**：整合 `config/` 与 `.env`，在 `README`、`docs/ops.md` 中补充 `conda python311` 激活步骤，结合 Makefile 打通 `make setup/test/run`。
- **模型与特征迭代**：在修复损失函数与维度问题后，引入注意力解释、特征重要性分析，评估是否需要扩展到图模型或概率模型以捕捉号码共现关系。
- **数据质量治理**：建立数据校验脚本（检查列名、缺失、异常值），并定期更新采集源以防接口变动；必要时构建本地镜像或缓存层。
- **测试体系完善**：针对特征构造、损失函数选择、预测输出编写单元/集成测试，确保覆盖率≥80%，为后续重构提供安全网。
- **自动化报告**：结合 `agent_report.md` 与 CI，将训练完成后的指标、覆盖率与风险提示自动写入，便于团队协作与审计。
