# -*- coding: utf-8 -*-
"""
验证项目重构后的功能
"""
import sys
import os

# 测试能否正确导入模块
try:
    from src.config import model_args, name_path
    print("✅ 配置模块导入成功")
except Exception as e:
    print(f"❌ 配置模块导入失败: {e}")
    sys.exit(1)

try:
    from src import modeling
    print("✅ 模型模块导入成功")
except Exception as e:
    print(f"❌ 模型模块导入失败: {e}")
    sys.exit(1)

try:
    from src import common
    print("✅ 通用模块导入成功")
except Exception as e:
    print(f"❌ 通用模块导入失败: {e}")
    sys.exit(1)

# 测试基本功能
print("\n📋 配置信息:")
print(f"支持的彩票类型: {list(name_path.keys())}")
print(f"快乐8红球类别数: {model_args['kl8']['model_args']['red_n_class']}")

print("\n📁 项目结构验证:")
required_dirs = ['src', 'scripts', 'tests', 'examples', 'docs', '.github/workflows']
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"✅ {dir_name} 目录存在")
    else:
        print(f"❌ {dir_name} 目录缺失")

required_files = [
    'src/__init__.py',
    'src/config.py', 
    'src/common.py',
    'src/modeling.py',
    'scripts/get_data.py',
    'scripts/train_model.py', 
    'scripts/predict.py',
    'tests/test_config.py',
    'tests/test_modeling.py',
    'setup.py',
    'README.md'
]

for file_name in required_files:
    if os.path.exists(file_name):
        print(f"✅ {file_name} 文件存在")
    else:
        print(f"❌ {file_name} 文件缺失")

print("\n🎉 项目重构验证完成！")