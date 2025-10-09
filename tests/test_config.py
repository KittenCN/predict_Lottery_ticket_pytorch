# -*- coding: utf-8 -*-
"""
测试配置模块
"""
import unittest
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import model_args, name_path

class TestConfig(unittest.TestCase):
    
    def test_name_path_exists(self):
        """测试彩票类型配置是否存在"""
        self.assertIn('kl8', name_path)
        self.assertIn('ssq', name_path)
        self.assertIn('dlt', name_path)
        self.assertIn('pls', name_path)
        self.assertIn('qxc', name_path)
    
    def test_model_args_structure(self):
        """测试模型参数配置结构"""
        for key in model_args:
            self.assertIn('model_args', model_args[key])
            self.assertIn('train_args', model_args[key])
            self.assertIn('path', model_args[key])
            
    def test_kl8_config(self):
        """测试快乐8配置"""
        kl8_config = model_args['kl8']
        self.assertEqual(kl8_config['model_args']['red_n_class'], 80)
        self.assertEqual(kl8_config['model_args']['red_sequence_len'], 20)

if __name__ == '__main__':
    unittest.main()