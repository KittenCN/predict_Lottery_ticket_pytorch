# -*- coding: utf-8 -*-
"""
测试模型模块
"""
import unittest
import torch
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.modeling import Transformer_Model, LSTM_Model, binary_encode_array, decode_one_hot

class TestModeling(unittest.TestCase):
    
    def setUp(self):
        self.device = torch.device("cpu")
        
    def test_transformer_model_creation(self):
        """测试Transformer模型创建"""
        model = Transformer_Model(
            input_size=100,
            output_size=20, 
            hidden_size=128,
            num_layers=2,
            num_heads=8
        )
        self.assertIsNotNone(model)
        
    def test_lstm_model_creation(self):
        """测试LSTM模型创建"""
        model = LSTM_Model(
            input_size=20,
            output_size=1600,
            hidden_size=128,
            num_layers=2,
            num_embeddings=80,
            embedding_dim=50,
            seq_len=5
        )
        self.assertIsNotNone(model)
        
    def test_binary_encode_array(self):
        """测试二进制编码"""
        input_array = torch.tensor([[1, 2, 3, 4, 5]])
        encoded = binary_encode_array(input_array, num_classes=10)
        self.assertEqual(encoded.shape[1], 10)
        
    def test_decode_one_hot(self):
        """测试one-hot解码"""
        # 创建一个简单的one-hot测试数据
        test_data = torch.zeros(80)
        test_data[0] = 1.0  # 第一个位置为1
        test_data[10] = 1.0  # 第十一个位置为1
        
        decoded = decode_one_hot(test_data, num_classes=80)
        self.assertIsInstance(decoded, list)

if __name__ == '__main__':
    unittest.main()