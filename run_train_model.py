# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""
import os
import time
import argparse
import warnings
from config import *
from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
import modeling
import torch.optim as optim
from torch.utils.data import DataLoader
from common import create_train_data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter   # to print to tensorboard

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="kl8", type=str, help="选择训练数据")
parser.add_argument('--windows_size', default='5', type=str, help="训练窗口大小,如有多个，用'，'隔开")
parser.add_argument('--red_epochs', default=100, type=int, help="红球训练轮数")
parser.add_argument('--blue_epochs', default=1, type=int, help="蓝球训练轮数")
parser.add_argument('--batch_size', default=32, type=int, help="集合数量")
parser.add_argument('--predict_pro', default=0, type=int, help="更新batch_size")
parser.add_argument('--epochs', default=1, type=int, help="训练轮数(红蓝球交叉训练)")
parser.add_argument('--cq', default=0, type=int, help="是否使用出球顺序，0：不使用（即按从小到大排序），1：使用")
parser.add_argument('--download_data', default=1, type=int, help="是否下载数据")
parser.add_argument('--hidden_size', default=32, type=int, help="hidden_size")
parser.add_argument('--num_layers', default=8, type=int, help="num_layers")
parser.add_argument('--num_heads', default=16, type=int, help="num_heads")
parser.add_argument('--tensorboard', default=0, type=int, help="tensorboard switch")
args = parser.parse_args()

pred_key = {}
save_epoch = 10
save_interval = 60
last_save_time = time.time()

if args.tensorboard == 1:
    writer = SummaryWriter('../tf-logs')

def train_ball_model(name, dataset, test_dataset, sub_name="红球"):
    """ 模型训练
    :param name: 玩法
    :param x_data: 训练样本
    :param y_data: 训练标签
    :return:
    """
    global last_save_time
    sub_name_eng = "red" if sub_name == "红球" else "blue"
    ball_model_name = red_ball_model_name if sub_name == "红球" else blue_ball_model_name
    m_args = model_args[name]
    syspath = model_path + model_args[args.name]["pathname"]['name'] + str(m_args["model_args"]["windows_size"]) + model_args[args.name]["subpath"][sub_name_eng]
    if not os.path.exists(syspath):
        os.makedirs(syspath)
    logger.info("标签数据维度: {}".format(dataset.data.shape))

    dataloader = DataLoader(dataset, batch_size=model_args[args.name]["model_args"]["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=model_args[args.name]["model_args"]["batch_size"], shuffle=False)
    # 定义模型和优化器
    model = modeling.Transformer_Model(input_size=20, output_size=20*80, windows_size=m_args["model_args"]["windows_size"], hidden_size=args.hidden_size, num_layers=args.num_layers, num_heads=args.num_heads, dropout=0.1, d_model=128).to(modeling.device)
    if os.path.exists("{}{}_ball_model_pytorch.ckpt".format(syspath, sub_name_eng)):
        model.load_state_dict(torch.load("{}{}_ball_model_pytorch.ckpt".format(syspath, sub_name_eng)))
        logger.info("已加载{}模型！".format(sub_name))
    criterion = nn.MSELoss()
    # criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    lr_scheduler=modeling.CustomSchedule(d_model=20*m_args["model_args"]["windows_size"], optimizer=optimizer)
    pbar = tqdm(range(model_args[args.name]["model_args"]["{}_epochs".format(sub_name_eng)]))
    running_loss = 0.0
    running_times = 0
    test_loss = 0.0
    test_times = 0
    for epoch in range(model_args[args.name]["model_args"]["{}_epochs".format(sub_name_eng)]):
        running_loss = 0.0
        running_times = 0
        for batch in dataloader:
            running_times += 1
            x, y = batch
            x = x.float().to(modeling.device)
            y = y.float().to(modeling.device)
            y_pred = model(x)
            t_loss = criterion(y_pred, y.view(y.size(0), -1))
            optimizer.zero_grad()
            t_loss.backward()
            lr_scheduler.step()
            optimizer.step()
            # running_loss += t_loss.item() * x.size(0)
            running_loss += t_loss.item()
        # print(f"Epoch {epoch+1}: Loss = {running_loss / len(dataset):.4f}")
        if (epoch + 1) % save_epoch == 0:
            if time.time() - last_save_time > save_interval:
                last_save_time = time.time()
                torch.save(model.state_dict(), "{}{}_pytorch.{}".format(syspath, ball_model_name, extension))
                # logger.info("【{}】{}模型已保存！".format(name_path[name]["name"], sub_name))
            # run test
            with torch.no_grad():
                test_loss = 0.0
                test_times = 0
                for batch in test_dataloader:
                    test_times += 1
                    x, y = batch
                    x = x.float().to(modeling.device)
                    y = y.float().to(modeling.device)
                    y_pred = model(x)
                    tt_loss = criterion(y_pred, y.view(y.size(0), -1))
                    # test_loss += tt_loss.item() * x.size(0)
                    test_loss += tt_loss.item()
                # logger.info("Epoch {}/{} Test Loss: {:.4f}".format(epoch+1, model_args[args.name]["model_args"]["{}_epochs".format(sub_name_eng)], test_loss / len(test_dataset)))
        if args.tensorboard == 1:
            writer.add_scalar('Loss/Running', running_loss / (running_times if running_times > 0 else 1), epoch)
            if (epoch + 1) % save_epoch == 0:
                writer.add_scalar('Loss/Test', test_loss / (test_times if test_times > 0 else 1), epoch)
        pbar.set_description("Avg_Loss: {:.4f} Test_Loss: {:.4f}".format(running_loss / (running_times if running_times > 0 else 1), test_loss / (test_times if test_times > 0 else 1)))
        pbar.update(1)
    if args.tensorboard == 1:
        writer.close()
    pbar.close()
    torch.save(model.state_dict(), "{}{}_pytorch.{}".format(syspath, ball_model_name, extension))
    logger.info("【{}】{}模型训练完成!".format(name_path[name]["name"], sub_name))
    logger.info("Tran Loss: {:.4f} Test Loss: {:.4f}".format(running_loss / (running_times if running_times > 0 else 1), test_loss / (test_times if test_times > 0 else 1)))

def action(name):
    logger.info("正在创建【{}】数据集...".format(name_path[name]["name"]))
    red_train_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"], 1, "red", args.cq, 0, 2021351)
    red_test_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"], 1, "red", args.cq, 1, 2021351)
    if name not in ["kl8"]:
        blue_train_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"], 1, "blue", args.cq, 0, 2021351)
        blue_test_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"], 1, "blue", args.cq, 1, 2021351)
    for i in range(args.epochs):
        if model_args[name]["model_args"]["red_epochs"] > 0:
            logger.info("开始训练【{}】红球模型...".format(name_path[name]["name"]))
            start_time = time.time()
            train_ball_model(name, dataset=red_train_data, test_dataset=red_test_data, sub_name="红球")
            logger.info("训练耗时: {:.4f}".format(time.time() - start_time))

        if name not in ["pls", "kl8"] and model_args[name]["model_args"]["blue_epochs"] > 0:
            logger.info("开始训练【{}】蓝球模型...".format(name_path[name]["name"]))
            start_time = time.time()
            # train_blue_ball_model(name, x_data=train_data["blue"]["x_data"], y_data=train_data["blue"]["y_data"])
            train_ball_model(name, dataset=blue_train_data, test_dataset=blue_test_data, sub_name="蓝球")
            logger.info("训练耗时: {:.4f}".format(time.time() - start_time))


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
    logger.info("训练总耗时: {:.4f}".format(time.time() - total_start_time))

if __name__ == '__main__':
    list_windows_size = args.windows_size.split(",")
    if not args.name:
        raise Exception("玩法名称不能为空！")
    elif not args.windows_size:
        raise Exception("窗口大小不能为空！")
    else:
        if args.download_data == 1 and args.predict_pro == 0 and int(time.strftime("%H", time.localtime())) < 20:
            logger.info("正在创建【{}】数据集...".format(name_path[args.name]["name"]))
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
