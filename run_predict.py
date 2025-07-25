# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""
import argparse
import modeling
import torch
from config import *
from loguru import logger
from common import setMiniargs, get_current_number, run_predict, init

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="kl8", type=str, help="选择训练数据")
parser.add_argument('--seq_len', default='5', type=str, help="训练窗口大小,如有多个，用'，'隔开")
parser.add_argument('--cq', default=0, type=int, help="是否使用出球顺序，0：不使用（即按从小到大排序），1：使用")
parser.add_argument('--batch_size', default=32, type=int, help="集合数量")
parser.add_argument('--hidden_size', default=2560, type=int, help="hidden_size")
parser.add_argument('--num_layers', default=6, type=int, help="num_layers")
parser.add_argument('--num_heads', default=8, type=int, help="num_heads")
parser.add_argument('--f_data', default=0, type=int, help="指定预测期数")
parser.add_argument('--model', default='Transformer', type=str, help="model name")
parser.add_argument('--test_mode', default=0, type=int, help="test_mode")
parser.add_argument('--cpu', default=0, type=int, help="using cpu, 1: cpu, 0: checking gpu")
args = parser.parse_args()

if args.cpu == 0:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

if __name__ == '__main__':
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not args.name:
        raise Exception("玩法名称不能为空！")
    elif not args.seq_len:
        raise Exception("窗口大小不能为空！")
    else:
        init()
        setMiniargs(args)
        list_seq_len = args.seq_len.split(",")
        if list_seq_len[0] == "-1":
            list_seq_len = []
            path = model_path + model_args[args.name]["pathname"]['name']
            dbtype_list = os.listdir(path)
            for dbtype in dbtype_list:
                try:
                    list_seq_len.append(int(dbtype))
                except:
                    pass
            if len(list_seq_len) == 0:
                raise Exception("没有找到训练模型！")
            list_seq_len.sort(reverse=True)   
            logger.info(path)
            logger.info("seq_len: {}".format(list_seq_len))
        for size in list_seq_len:
            current_number = get_current_number(args.name)
            # run_predict(int(size), model_args[args.name]["model_args"]['red_sequence_len'], hidden_size=args.hidden_size, num_layers=args.num_layers, num_heads=args.num_heads, input_size=model_args[args.name]["model_args"]["red_n_class"]*int(size), output_size=model_args[args.name]["model_args"]["red_n_class"], f_data=args.f_data, model=args.model)
            # window_size, sequence_len, hidden_size=128, num_layers=8, num_heads=16, input_size=20, output_size=20, f_data=0, model="Transformer", args=None, test_mode=0
            if args.model == "Transformer":
                _input_size=(model_args[args.name]["model_args"]["{}_sequence_len".format("red")]+modeling.extra_classes)*model_args[args.name]["model_args"]["seq_len"]
                _output_size=model_args[args.name]["model_args"]["{}_sequence_len".format("red")]
            elif args.model == "LSTM":
                _input_size=model_args[args.name]["model_args"]["{}_sequence_len".format("red")]
                _output_size=(model_args[args.name]["model_args"]["{}_sequence_len".format("red")]+modeling.extra_classes)*model_args[args.name]["model_args"]["{}_n_class".format("red")]
            run_predict(window_size=int(size), \
                        sequence_len=model_args[args.name]["model_args"]['red_sequence_len']+modeling.extra_classes, \
                        hidden_size=args.hidden_size, \
                        num_layers=args.num_layers, \
                        num_heads=args.num_heads, \
                        input_size=_input_size, \
                        output_size=_output_size, \
                        f_data=args.f_data, \
                        model=args.model, \
                        args=args, \
                        test_mode=args.test_mode)
        
