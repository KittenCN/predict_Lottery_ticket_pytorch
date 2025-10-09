# -*- coding: utf-8 -*-
"""
Lottery Ticket Prediction PyTorch Package
基于transformer模型的彩票预测
"""

__version__ = "1.0.0"
__author__ = "KittenCN"

from .config import *
from . import modeling
from . import common

__all__ = ['config', 'modeling', 'common']