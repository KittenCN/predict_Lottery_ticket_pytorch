# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""
import datetime
import os
import random
import time
import json
import argparse
import numpy as np
import pandas as pd
import warnings
from config import *
from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import modeling
import torch.optim as optim
from torch.utils.data import DataLoader
from common import get_data_run


warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="kl8", type=str, help="选择训练数据")
parser.add_argument('--windows_size', default='1', type=str, help="训练窗口大小,如有多个，用'，'隔开")
parser.add_argument('--red_epochs', default=100, type=int, help="红球训练轮数")
parser.add_argument('--blue_epochs', default=1, type=int, help="蓝球训练轮数")
parser.add_argument('--batch_size', default=32, type=int, help="集合数量")
parser.add_argument('--predict_pro', default=0, type=int, help="更新batch_size")
parser.add_argument('--epochs', default=1, type=int, help="训练轮数(红蓝球交叉训练)")
parser.add_argument('--cq', default=0, type=int, help="是否使用出球顺序，0：不使用（即按从小到大排序），1：使用")
parser.add_argument('--download_data', default=1, type=int, help="是否下载数据")
args = parser.parse_args()

pred_key = {}
ori_data = None
save_epoch = 10
save_interval = 600
last_save_time = time.time()

def create_train_data(name, windows, dataset=0):
    """ 创建训练数据
    :param name: 玩法，双色球/大乐透
    :param windows: 训练窗口
    :return:
    """
    global ori_data
    if ori_data is None:
        if args.cq == 1 and name == "kl8":
            ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_cq_file_name))
        else:
            ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
    data = ori_data.copy()
    if not len(data):
        raise logger.error(" 请执行 get_data.py 进行数据下载！")
    else:
        # 创建模型文件夹
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        logger.info("训练数据已加载! ")

    data = data.iloc[:, 2:].values
    logger.info("训练集数据维度: {}".format(data.shape))
    x_data, y_data = [], []
    if dataset == 0:
        for i in range(len(data) - windows - 1):
            sub_data = data[i:(i+windows+1), :]
            x_data.append(sub_data[1:])
            y_data.append(sub_data[0])

        cut_num = model_args[name]["model_args"]["red_sequence_len"]
        return {
            "red": {
                "x_data": np.array(x_data)[:, :, :cut_num], "y_data": np.array(y_data)[:, :cut_num]
            },
            "blue": {
                "x_data": np.array(x_data)[:, :, cut_num:], "y_data": np.array(y_data)[:, cut_num:]
            }
        }
    else:
        dataset = modeling.MyDataset(data)
        return dataset


def train_red_ball_model(name, dataset):
    """ 红球模型训练
    :param name: 玩法
    :param x_data: 训练样本
    :param y_data: 训练标签
    :return:
    """
    global last_save_time
    m_args = model_args[name]
    syspath = model_path + model_args[args.name]["pathname"]['name'] + str(m_args["model_args"]["windows_size"]) + model_args[args.name]["subpath"]['red']
    if not os.path.exists(syspath):
        os.makedirs(syspath)
    # data_len = dataset.data.shape[0]
    # if len(x_data) % m_args["model_args"]["batch_size"] != 0:
    #         diff = m_args["model_args"]["batch_size"] - (len(x_data) % m_args["model_args"]["batch_size"])
    #         while diff > 0:
    #             random_index = np.random.randint(0, data_len)
    #             x_data = np.append(x_data, [x_data[random_index]], axis=0)
    #             y_data = np.append(y_data, [y_data[random_index]], axis=0)
    #             diff -= 1
    logger.info("标签数据维度: {}".format(dataset.data.shape))

    dataloader = DataLoader(dataset, batch_size=model_args[args.name]["model_args"]["batch_size"], shuffle=True)

    # 定义模型和优化器
    model = modeling.TransformerModel(input_size=20, output_size=20).to(device)
    if os.path.exists("{}red_ball_model_pytorch.ckpt".format(syspath)):
        # saver = tf.compat.v1.train.Saver()
        # saver.restore(sess, "{}red_ball_model.ckpt".format(syspath))
        model.load_state_dict(torch.load("{}red_ball_model_pytorch.ckpt".format(syspath)))
        logger.info("已加载红球模型！")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(model_args[args.name]["model_args"]["red_epochs"]):
        running_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            x = x.unsqueeze(1) # 将输入序列的最后一维扩展为 1
            y_pred = model(x.float())
            loss = criterion(y_pred, y.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        print(f"Epoch {epoch+1}: Loss = {running_loss / len(dataset):.4f}")
        if (epoch + 1) % save_epoch == 0:
            if time.time() - last_save_time > save_interval:
                last_save_time = time.time()
                torch.save(model.state_dict(), "{}{}_pytorch.{}".format(syspath, red_ball_model_name, extension))
    torch.save(model.state_dict(), "{}{}_pytorch.{}".format(syspath, red_ball_model_name, extension))
    logger.info("【{}】红球模型训练完成!".format(name_path[name]["name"]))

def train_blue_ball_model(name, x_data, y_data):
    """ 蓝球模型训练
    :param name: 玩法
    :param x_data: 训练样本
    :param y_data: 训练标签
    :return:
    """
    global last_save_time
    

def action(name):
    logger.info("正在创建【{}】数据集...".format(name_path[name]["name"]))
    train_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"], 1)
    for i in range(args.epochs):
        if model_args[name]["model_args"]["red_epochs"] > 0:
            # tf.compat.v1.reset_default_graph()  # 重置网络图
            logger.info("开始训练【{}】红球模型...".format(name_path[name]["name"]))
            start_time = time.time()
            # train_red_ball_model(name, x_data=train_data["red"]["x_data"], y_data=train_data["red"]["y_data"])
            train_red_ball_model(name, dataset=train_data)
            logger.info("训练耗时: {:.4f}".format(time.time() - start_time))

        if name not in ["pls", "kl8"] and model_args[name]["model_args"]["blue_epochs"] > 0:
            # tf.compat.v1.reset_default_graph()  # 重置网络图

            logger.info("开始训练【{}】蓝球模型...".format(name_path[name]["name"]))
            start_time = time.time()
            train_blue_ball_model(name, x_data=train_data["blue"]["x_data"], y_data=train_data["blue"]["y_data"])
            logger.info("训练耗时: {:.4f}".format(time.time() - start_time))

        # 保存预测关键结点名
        # with open("{}/{}".format(model_path + model_args[args.name]["pathname"]['name'] + str(model_args[args.name]["model_args"]["windows_size"]), pred_key_name), "w") as f:
        #     json.dump(pred_key, f)

def run(name, windows_size):
    """ 执行训练
    :param name: 玩法
    :return:
    """
    total_start_time = time.time()
    if int(windows_size[0]) == 0:
        action(name)
    else:
        for size in windows_size:
            model_args[name]["model_args"]["windows_size"] = int(size)
            action(name)
            filename = datetime.datetime.now().strftime('%Y%m%d')
            filepath = "{}{}/".format(predict_path, args.name)
            fileadd = "{}{}{}".format(filepath, filename, ".csv")
            if args.predict_pro == 0 and int(time.strftime("%H", time.localtime())) >=18 and os.path.exists(fileadd) == False:
                logger.info("开始预测【{}】...".format(name_path[name]["name"]))
                _tmpRedEpochs =  model_args[args.name]["model_args"]["red_epochs"]
                _tmpBlueEpochs = model_args[args.name]["model_args"]["blue_epochs"]
                _tmpBatchSize = model_args[args.name]["model_args"]["batch_size"]
                if model_args[args.name]["model_args"]["red_epochs"] >= 1:
                    model_args[args.name]["model_args"]["red_epochs"] = 1
                    args.red_eopchs = 1
                if model_args[args.name]["model_args"]["blue_epochs"] >= 1:
                    model_args[args.name]["model_args"]["blue_epochs"] = 1
                    args.blue_epochs = 1
                model_args[args.name]["model_args"]["batch_size"] = 1
                args.batch_size = 1
                # init()
                # setMiniargs(args)
                for w_size in windows_size:
                    model_args[name]["model_args"]["windows_size"] = int(w_size)
                    train_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"])
                    if model_args[name]["model_args"]["red_epochs"] > 0:
                        # tf.compat.v1.reset_default_graph()  # 重置网络图
                        logger.info("开始训练【{}】红球模型...".format(name_path[name]["name"]))
                        start_time = time.time()
                        train_red_ball_model(name, x_data=train_data["red"]["x_data"], y_data=train_data["red"]["y_data"])
                        logger.info("训练耗时: {:.4f}".format(time.time() - start_time))

                    if name not in ["pls", "kl8"] and model_args[name]["model_args"]["blue_epochs"] > 0:
                        # tf.compat.v1.reset_default_graph()  # 重置网络图

                        logger.info("开始训练【{}】蓝球模型...".format(name_path[name]["name"]))
                        start_time = time.time()
                        train_blue_ball_model(name, x_data=train_data["blue"]["x_data"], y_data=train_data["blue"]["y_data"])
                        logger.info("训练耗时: {:.4f}".format(time.time() - start_time))

                    # 保存预测关键结点名
                    with open("{}/{}".format(model_path + model_args[args.name]["pathname"]['name'] + str(model_args[args.name]["model_args"]["windows_size"]), pred_key_name), "w") as f:
                        json.dump(pred_key, f)

                    # predict_tf.compat.v1.reset_default_graph()
                    # red_graph = predict_tf.compat.v1.Graph()
                    # blue_graph = predict_tf.compat.v1.Graph()
                    # pred_key_d = {}
                    # red_sess = predict_tf.compat.v1.Session(graph=red_graph)
                    # blue_sess = predict_tf.compat.v1.Session(graph=blue_graph)
                    # current_number = get_current_number(args.name)
                    # run_predict(int(w_size))
                    # _data, _title = predict_run(args.name)
                # df = pd.DataFrame(_data, columns=_title)
                # if not os.path.exists(filepath):
                #     os.makedirs(filepath)
                # df.to_csv(fileadd, encoding="utf-8",index=False)

                model_args[args.name]["model_args"]["red_epochs"] = _tmpRedEpochs
                args.red_epochs = _tmpRedEpochs
                model_args[args.name]["model_args"]["blue_epochs"] = _tmpBlueEpochs
                args.blue_epochs = _tmpBlueEpochs
                model_args[args.name]["model_args"]["batch_size"] = _tmpBatchSize
                args.batch_size = _tmpBatchSize
            if args.download_data == 1 and args.predict_pro == 0 and int(time.strftime("%H", time.localtime())) >=23 and os.path.exists(fileadd):
                print("正在创建【{}】数据集...".format(name_path[args.name]["name"]))
                get_data_run(name=args.name, cq=args.cq)

    epochs = model_args[args.name]["model_args"]["red_epochs"]
    if epochs == 0:
        epochs = model_args[args.name]["model_args"]["blue_epochs"]
    logger.info("总耗时: {:.4f}, 平均效率：{:.4f}".format(time.time() - total_start_time, epochs / ((time.time() - total_start_time) / 3600)))

if __name__ == '__main__':
    list_windows_size = args.windows_size.split(",")
    if not args.name:
        raise Exception("玩法名称不能为空！")
    elif not args.windows_size:
        raise Exception("窗口大小不能为空！")
    else:
        if args.download_data == 1 and args.predict_pro == 0 and int(time.strftime("%H", time.localtime())) < 20:
            print("正在创建【{}】数据集...".format(name_path[args.name]["name"]))
            # get_data_run(name=args.name, cq=args.cq)
        model_args[args.name]["model_args"]["red_epochs"] = int(args.red_epochs)
        model_args[args.name]["model_args"]["blue_epochs"] = int(args.blue_epochs)
        model_args[args.name]["model_args"]["batch_size"] = int(args.batch_size)
        if args.predict_pro == 1:
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
            model_args[args.name]["model_args"]["red_epochs"] = 1
            model_args[args.name]["model_args"]["blue_epochs"] = 1
            model_args[args.name]["model_args"]["batch_size"] = 1
        else:
            if args.epochs > 1:
                model_args[args.name]["model_args"]["red_epochs"] = 1
                model_args[args.name]["model_args"]["blue_epochs"] = 1
            elif args.epochs <= 0:
                raise Exception("训练轮数不能小于1！")
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
        run(args.name, list_windows_size)
