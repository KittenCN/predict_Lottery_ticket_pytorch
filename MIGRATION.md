# 📋 项目重构迁移指南

## 🔄 重构概述

项目已按照标准 Python 包结构重新整理，所有文件都移动到了相应的目录中，并修正了所有的引用关系。

## 📁 目录结构变更

### 旧结构 → 新结构

```
旧文件位置              →  新文件位置
config.py              →  src/config.py
common.py              →  src/common.py  
modeling.py            →  src/modeling.py
get_data.py            →  scripts/get_data.py
run_train_model.py     →  scripts/train_model.py
run_predict.py         →  scripts/predict.py
DQN_test.ipynb         →  examples/DQN_test.ipynb
README.md              →  docs/README.md (复制)
```

## 🚀 新的使用方式

### 1. 数据获取
**旧命令:**
```bash
python get_data.py --name kl8 --cq 0
```

**新命令:**
```bash
python scripts/get_data.py --name kl8 --cq 0
```

### 2. 模型训练
**旧命令:**
```bash
python run_train_model.py --name kl8 --seq_len 5 --red_epochs 100
```

**新命令:**
```bash
python scripts/train_model.py --name kl8 --seq_len 5 --red_epochs 100
```

### 3. 模型预测
**旧命令:**
```bash
python run_predict.py --name kl8 --seq_len 5 --model Transformer
```

**新命令:**
```bash
python scripts/predict.py --name kl8 --seq_len 5 --model Transformer
```

## 📦 Python 包导入方式

### 旧方式:
```python
import config
import common  
import modeling
```

### 新方式:
```python
from src import config
from src import common
from src import modeling

# 或者
from src.config import model_args, name_path
from src.common import create_train_data, run_predict
from src.modeling import Transformer_Model, LSTM_Model
```

## 🔧 开发者功能

### 运行测试
```bash
python tests/test_config.py
python tests/test_modeling.py
```

### 安装为包
```bash
pip install -e .
```

### 构建包
```bash
python setup.py build
python setup.py sdist bdist_wheel
```

## 📋 新增功能

1. **标准化项目结构**: 符合 Python 包开发最佳实践
2. **完整的测试套件**: 包含配置和模型的单元测试
3. **详细的文档**: API 文档和使用指南
4. **CI/CD 支持**: GitHub Actions 自动化测试
5. **包管理**: 支持 pip 安装和分发
6. **更好的组织**: 源码、脚本、测试、文档分离

## ⚠️ 注意事项

1. **所有的 import 路径都已更新**: 如果你有自定义的脚本，需要更新导入路径
2. **脚本位置变更**: 原来根目录的脚本现在在 `scripts/` 目录
3. **配置文件**: 配置文件现在在 `src/config.py`
4. **模型保存路径**: 保持不变，仍在 `model/` 目录
5. **数据路径**: 保持不变，仍在 `data/` 目录

## ✅ 验证重构结果

运行验证脚本确保一切正常:
```bash
python verify_structure.py
```

## 🆘 遇到问题？

如果在使用过程中遇到任何问题，请检查：

1. Python 路径是否正确
2. 是否使用了正确的新命令格式
3. 所有依赖是否已安装
4. 文件路径是否正确

重构完成后，项目结构更加清晰，便于维护和扩展！