# -*- coding:utf-8 -*-
"""
Author: BigCat
Modifier: KittenCN
"""
import argparse
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.common import get_data_run
from src.pipeline import DEFAULT_PIPELINE
from scripts.check_pipeline_init import require_pipeline_args

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="kl8", type=str, help="选择爬取数据")
parser.add_argument('--cq', default=0, type=int, help="是否使用出球顺序，0：不使用（即按从小到大排序），1：使用")
args = parser.parse_args()

def main():
    """主函数"""
    if not args.name:
        raise Exception("玩法名称不能为空！")
    else:
        if DEFAULT_PIPELINE.args is None:
            DEFAULT_PIPELINE.set_args(args)
        require_pipeline_args()
        get_data_run(name=args.name, cq=args.cq)

if __name__ == "__main__":
    main()