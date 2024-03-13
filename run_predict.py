# -*- coding:utf-8 -*-
"""
Author: BigCat
Modifier: KittenCN
"""
import argparse
from config import *
from loguru import logger
from common import setMiniargs, get_current_number, run_predict, init

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="kl8", type=str, help="选择训练数据")
parser.add_argument('--windows_size', default='5', type=str, help="训练窗口大小,如有多个，用'，'隔开")
parser.add_argument('--cq', default=0, type=int, help="是否使用出球顺序，0：不使用（即按从小到大排序），1：使用")
parser.add_argument('--hidden_size', default=32, type=int, help="hidden_size")
parser.add_argument('--num_layers', default=8, type=int, help="num_layers")
parser.add_argument('--num_heads', default=16, type=int, help="num_heads")
parser.add_argument('--f_data', default=0, type=int, help="f_data")
args = parser.parse_args()

if __name__ == '__main__':
    if not args.name:
        raise Exception("玩法名称不能为空！")
    elif not args.windows_size:
        raise Exception("窗口大小不能为空！")
    else:
        init()
        setMiniargs(args)
        list_windows_size = args.windows_size.split(",")
        if list_windows_size[0] == "-1":
            list_windows_size = []
            path = model_path + model_args[args.name]["pathname"]['name']
            dbtype_list = os.listdir(path)
            for dbtype in dbtype_list:
                try:
                    list_windows_size.append(int(dbtype))
                except:
                    pass
            if len(list_windows_size) == 0:
                raise Exception("没有找到训练模型！")
            list_windows_size.sort(reverse=True)   
            logger.info(path)
            logger.info("windows_size: {}".format(list_windows_size))
        for size in list_windows_size:
            current_number = get_current_number(args.name)
            run_predict(int(size), model_args[args.name]["model_args"]['red_sequence_len'], hidden_size=args.hidden_size, num_layers=args.num_layers, num_heads=args.num_heads, input_size=base_size, output_size=base_size, f_data=args.f_data)
            # _data, _title = predict_run(args.name)
        # filename = datetime.datetime.now().strftime('%Y%m%d')
        # filepath = "{}{}/".format(predict_path, args.name)
        # fileadd = "{}{}{}".format(filepath, filename, ".csv")
        # if not os.path.exists(filepath):
        #     os.makedirs(filepath)
        # df = pd.DataFrame(_data, columns=_title)
        # df.to_csv(fileadd, encoding="utf-8",index=False)
        
