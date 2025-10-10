# smoke_train_predict.py
"""
极简端到端训练-推理-评估冒烟脚本
- 训练 2 轮，batch=2，随机数据
- 训练完成后自动调用 predict 脚本
- 检查模型 ckpt 是否生成，输出关键日志
"""
import os
import sys
import subprocess
import glob

# 训练
print("[smoke] 开始训练...")
ret = subprocess.run([
    sys.executable, 'scripts/train_model.py', '--name', 'kl8', '--seq_len', '5', '--red_epochs', '2', '--batch_size', '2'
], capture_output=True, text=True)
print(ret.stdout)
assert ret.returncode == 0, "训练失败"

# 检查模型文件
ckpt_dir = os.path.join('model', 'kl8', '5', 'red_ball_model')
ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
print(f"[smoke] ckpt 文件数: {len(ckpt_files)}")
assert len(ckpt_files) > 0, "未生成 ckpt 文件"

# 推理
print("[smoke] 开始推理...")
ret2 = subprocess.run([
    sys.executable, 'scripts/predict.py', '--name', 'kl8', '--seq_len', '5', '--model', 'Transformer'
], capture_output=True, text=True)
print(ret2.stdout)
assert ret2.returncode == 0, "推理失败"

print("[smoke] 冒烟测试通过！")
