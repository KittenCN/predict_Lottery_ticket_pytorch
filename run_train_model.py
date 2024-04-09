# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""
import os
import time
import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import modeling
import sys
from torch.utils.data import DataLoader
from common import create_train_data, get_data_run
from tqdm import tqdm
from config import *
from loguru import logger
from datetime import datetime as dt
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
parser.add_argument('--hidden_size', default=512, type=int, help="hidden_size")
parser.add_argument('--num_layers', default=6, type=int, help="num_layers")
parser.add_argument('--num_heads', default=8, type=int, help="num_heads")
parser.add_argument('--tensorboard', default=0, type=int, help="tensorboard switch")
parser.add_argument('--num_workers', default=2, type=int, help="num_workers switch")
parser.add_argument('--top_k', default=10, type=int, help="top_k switch")
parser.add_argument('--model', default='Transformer', type=str, help="model name")
parser.add_argument('--lr', default=0.01, type=float, help="learning rate")
parser.add_argument('--plus_mode', default=0, type=int, help="plus mode")
parser.add_argument('--ext_times', default=1000, type=int, help="ext_times")
parser.add_argument('--init', default=0, type=int, help="init")
args = parser.parse_args()

pred_key = {}
save_epoch = 10
save_interval = 60
last_save_time = time.time()
best_score = 999999999
start_dt = dt.now().strftime("%Y%m%d%H%M%S")

if args.tensorboard == 1:
    writer = SummaryWriter('../tf-logs')

if args.model == "Transformer":
    _model = modeling.Transformer_Model
elif args.model == "LSTM":
    _model = modeling.LSTM_Model

def save_model(model, optimizer, lr_scheduler, epoch, syspath, ball_model_name, other=""):
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    scheduler_state_dict = lr_scheduler.state_dict() 
    save_dict = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict,
        'epoch': epoch,
        'start_dt': start_dt,
        'windows_size': args.windows_size,
        'batch_size': args.batch_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'best_score': best_score,
    }
    torch.save(save_dict, "{}{}_pytorch_{}{}.{}".format(syspath, ball_model_name, args.model, other, extension))

def load_model(syspath, sub_name_eng, model, optimizer, lr_scheduler, sub_name="红球", other=""):
    global best_score, start_dt
    current_epoch = 0
    address = "{}{}_ball_model_pytorch_{}{}.{}".format(syspath, sub_name_eng, args.model, other, extension)
    if os.path.exists(address):
        # model.load_state_dict(torch.load("{}{}_ball_model_pytorch.ckpt".format(syspath, sub_name_eng)))
        checkpoint = torch.load(address)
        if 'windows_size' in checkpoint and 'batch_size' in checkpoint and 'hidden_size' in checkpoint and 'num_layers' in checkpoint and 'num_heads' in checkpoint:
            if checkpoint['windows_size'] != args.windows_size or checkpoint['batch_size'] != args.batch_size or checkpoint['hidden_size'] != args.hidden_size or checkpoint['num_layers'] != args.num_layers or checkpoint['num_heads'] != args.num_heads:
                logger.info("模型参数不一致，重新训练！")
                sys.exit()
        else:
            logger.info("模型不是最新版本，建议重新训练！")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'epoch' in checkpoint:
            current_epoch = checkpoint['epoch']
            if current_epoch >= model_args[args.name]["model_args"]["{}_epochs".format(sub_name_eng)] - 1:
                current_epoch = 0
        if 'start_dt' in checkpoint:
            start_dt = checkpoint['start_dt']
        if 'best_score' in checkpoint:
            best_score = checkpoint['best_score']
        if args.init == 1:
            current_epoch = 0
        logger.info("已加载{}模型！".format(sub_name))
        logger.info("当前epoch是 {}, 初次启动时间是 {}, 最佳分数是 {:.2e}".format(current_epoch, start_dt, best_score))
    return current_epoch

def train_ball_model(name, dataset, test_dataset, sub_name="红球"):
    """ 模型训练
    :param name: 玩法
    :param x_data: 训练样本
    :param y_data: 训练标签
    :return:
    """
    global last_save_time, best_score, start_dt
    sub_name_eng = "red" if sub_name == "红球" else "blue"
    ball_model_name = red_ball_model_name if sub_name == "红球" else blue_ball_model_name
    m_args = model_args[name]
    syspath = model_path + model_args[args.name]["pathname"]['name'] + str(m_args["model_args"]["windows_size"]) + model_args[args.name]["subpath"][sub_name_eng]
    if not os.path.exists(syspath):
        os.makedirs(syspath)
    logger.info("标签数据维度: {}".format(dataset.data.shape))

    dataloader = DataLoader(dataset, batch_size=model_args[args.name]["model_args"]["batch_size"], shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=model_args[args.name]["model_args"]["batch_size"], shuffle=False, num_workers=args.num_workers, pin_memory=False)
    # 定义模型和优化器
    if args.model == "Transformer":
        model = _model(input_size=m_args["model_args"]["red_n_class"]*m_args["model_args"]["windows_size"], output_size=m_args["model_args"]["red_n_class"], hidden_size=args.hidden_size, num_layers=args.num_layers, num_heads=args.num_heads, dropout=0.1).to(modeling.device)
    elif args.model == "LSTM":
        model = _model(input_size=m_args["model_args"]["red_sequence_len"]*m_args["model_args"]["red_n_class"], output_size=m_args["model_args"]["red_sequence_len"]*m_args["model_args"]["red_n_class"], hidden_size=args.hidden_size, num_layers=args.num_layers, num_heads=args.num_heads, dropout=0.1).to(modeling.device)
    # criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss() # 二分类交叉熵
    criterion = nn.BCELoss() # 二分类交叉熵
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # lr_scheduler=modeling.CustomSchedule(d_model=args.hidden_size, optimizer=optimizer)
    lr_scheduler = modeling.CustomSchedule(optimizer=optimizer, d_model=args.hidden_size, warmup_steps=model_args[args.name]["model_args"]["{}_epochs".format(sub_name_eng)]*0.2)
    current_epoch = 0
    current_epoch = load_model(syspath, sub_name_eng, model, optimizer, lr_scheduler, sub_name)
    pbar = tqdm(range(model_args[args.name]["model_args"]["{}_epochs".format(sub_name_eng)]), ncols=150)
    running_loss = 0.0
    running_times = 0
    test_loss = 0.0
    test_times = 0
    topk_loss = 0.0
    topk_times = 0
    top20_loss = 0.0
    top20_times = 0
    no_update_times = 0
    for epoch in range(current_epoch, model_args[args.name]["model_args"]["{}_epochs".format(sub_name_eng)]):
        no_update_times += 1
        if no_update_times > args.ext_times and args.plus_mode == 1:
            print()
            no_update_times = 0
            _ = load_model(syspath, sub_name_eng, model, optimizer, lr_scheduler, sub_name, other="_{}_{}".format(start_dt, "best"))
        if epoch == current_epoch:
            pbar.update(current_epoch)
        running_loss = 0.0
        running_times = 0
        for batch in dataloader:
            model.train()
            running_times += 1
            x, y = batch
            x = x.float().to(modeling.device)
            y = y.float().to(modeling.device)
            y_pred = model(x)
            t_loss = criterion(y_pred, y.view(y.size(0), -1))
            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()
            # running_loss += t_loss.item() * x.size(0)
            running_loss += t_loss.item()
        # print(f"Epoch {epoch+1}: Loss = {running_loss / len(dataset):.4f}")
        lr_scheduler.step()
        if (epoch + 1) % save_epoch == 0:
            if time.time() - last_save_time > save_interval:
                last_save_time = time.time()
                save_model(model, optimizer, lr_scheduler, epoch, syspath, ball_model_name)
            # run test
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                test_times = 0
                topk_loss = 0.0
                topk_times = 0
                total_correct = 0.0
                top20_loss = 0.0
                top20_times = 0
                tatal20_correct = 0.0
                for batch in test_dataloader:
                    test_times += 1
                    x, y = batch
                    x = x.float().to(modeling.device)
                    y = y.float().to(modeling.device)
                    y_pred = model(x)
                    tt_loss = criterion(y_pred, y.view(y.size(0), -1))
                    # test_loss += tt_loss.item() * x.size(0)
                    test_loss += tt_loss.item()
                    # calculate topk loss
                    if args.model == "Transformer":
                        probs, indices = torch.topk(y_pred, args.top_k, dim=1)
                        for i in range(x.size(0)):
                            topk_times += args.top_k
                            target_indices = y[i].nonzero(as_tuple=False).squeeze()
                            total_correct += sum([1 for j in indices[i] if j in target_indices])
                        probs, indices = torch.topk(y_pred, 20, dim=1)
                        for i in range(x.size(0)):
                            top20_times += 20
                            target_indices = y[i].nonzero(as_tuple=False).squeeze()
                            tatal20_correct += sum([1 for j in indices[i] if j in target_indices])
                    elif args.model == "LSTM":
                        for i in range(x.size(0)):
                            _ele = modeling.decode_one_hot(y_pred[i], sort_by_max_value=True)
                            topk_times += args.top_k
                            top20_times += 20
                            # target_indices = y[i].nonzero(as_tuple=False).squeeze()
                            target_indices = modeling.decode_one_hot(y[i])
                            total_correct += sum([1 for j in _ele[0:args.top_k] if j in target_indices])
                            tatal20_correct += sum([1 for j in _ele if j in target_indices])
                # logger.info("Epoch {}/{} Test Loss: {:.2e}".format(epoch+1, model_args[args.name]["model_args"]["{}_epochs".format(sub_name_eng)], test_loss / len(test_dataset)))
            topk_loss = 1 - total_correct / (topk_times if topk_times > 0 else 1)
            top20_loss = 1 - tatal20_correct / (top20_times if top20_times > 0 else 1)
            if top20_loss < best_score:
                no_update_times = 0
                best_score = top20_loss
                save_model(model, optimizer, lr_scheduler, epoch, syspath, ball_model_name, other="_{}_{}".format(start_dt, "best"))
            if topk_loss < best_score:
                no_update_times = 0
                best_score = topk_loss
                save_model(model, optimizer, lr_scheduler, epoch, syspath, ball_model_name, other="_{}_{}".format(start_dt, "best"))
        if args.tensorboard == 1:
            writer.add_scalar('Loss/Running', running_loss / (running_times if running_times > 0 else 1), epoch)
            if (epoch + 1) % save_epoch == 0:
                writer.add_scalar('Loss/Test', test_loss / (test_times if test_times > 0 else 1), epoch)
                writer.add_scalar('Loss/TopK{}'.format(args.top_k), topk_loss, epoch)
                writer.add_scalar('Loss/TopK20', top20_loss, epoch)
        pbar.set_description("AL:{:.2e} TL:{:.2e} KL{}:{:.2e} KL20:{:.2e} lr:{:.2e} hs:{:.2e}".format(running_loss / (running_times if running_times > 0 else 1), test_loss / (test_times if test_times > 0 else 1), args.top_k, topk_loss, top20_loss, optimizer.param_groups[0]['lr'], best_score))
        pbar.update(1)
    if args.tensorboard == 1:
        writer.close()
    save_model(model, optimizer, lr_scheduler, epoch, syspath, ball_model_name, other="_{}".format(start_dt))
    logger.info("【{}】{}模型训练完成!".format(name_path[name]["name"], sub_name))
    pbar.set_description("AL:{:.2e} TL:{:.2e} KL{}:{:.2e} KL20:{:.2e} lr:{:.2e} hs:{:.2e}".format(running_loss / (running_times if running_times > 0 else 1), test_loss / (test_times if test_times > 0 else 1), args.top_k, topk_loss, top20_loss, optimizer.param_groups[0]['lr'], best_score))
    pbar.close()

def action(name):
    logger.info("正在创建【{}】数据集...".format(name_path[name]["name"]))
    red_train_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"], 1, "red", args.cq, 0, 2021351, model=args.model)
    red_test_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"], 1, "red", args.cq, 1, 2021351, model=args.model)
    if name not in ["kl8"]:
        blue_train_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"], 1, "blue", args.cq, 0, 2021351, model=args.model)
        blue_test_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"], 1, "blue", args.cq, 1, 2021351, model=args.model)
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
            get_data_run(name=args.name, cq=args.cq)
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
