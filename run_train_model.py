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
import glob
import pandas as pd
from torch.utils.data import DataLoader
from common import create_train_data, get_data_run
from tqdm import tqdm
from config import *
from loguru import logger
from datetime import datetime as dt
from prefetch_generator import BackgroundGenerator
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter   # to print to tensorboard

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
parser.add_argument('--num_workers', default=0, type=int, help="num_workers switch")
parser.add_argument('--top_k', default=10, type=int, help="top_k switch")
parser.add_argument('--model', default='Transformer', type=str, help="model name")
parser.add_argument('--lr', default=0.01, type=float, help="learning rate")
parser.add_argument('--plus_mode', default=0, type=int, help="plus mode")
parser.add_argument('--ext_times', default=1000, type=int, help="ext_times")
parser.add_argument('--init', default=0, type=int, help="init")
parser.add_argument('--train_mode', default=0, type=int, help="0: mormal, 1: new trainning, 2: best test model, 3: best loss model")
parser.add_argument('--split_time', default=2021351, type=int, help="tranning data split time, greater than 0, will saving best test model")
parser.add_argument('--save_best_loss', default=0, type=int, help="save best loss model")
args = parser.parse_args()

warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pred_key = {}
save_epoch = 50
save_interval = 60
last_save_time = time.time()
best_score = 999999999
best_loss = 999999999
start_dt = dt.now().strftime("%Y%m%d%H%M%S")
test_list = []
red_train_data = None
red_test_data = None
blue_train_data = None
blue_test_data = None

if args.tensorboard == 1:
    if not os.path.exists('../tf-logs'):
        os.makedirs('../tf-logs')
    writer = SummaryWriter('../tf-logs')

if args.model == "Transformer":
    _model = modeling.Transformer_Model
elif args.model == "LSTM":
    _model = modeling.LSTM_Model

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def save_model(model, optimizer, lr_scheduler, scaler, epoch, syspath, ball_model_name, other="", no_update_times=0):
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    scheduler_state_dict = lr_scheduler.state_dict() 
    scaler_state_dict = scaler.state_dict()
    save_dict = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict,
        'scaler_state_dict': scaler_state_dict,
        'epoch': epoch,
        'start_dt': start_dt,
        'windows_size': args.windows_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'best_score': best_score,
        'no_update_times': no_update_times,
        'split_time': args.split_time,
        'test_list': test_list,
        'best_lost': best_loss,
    }
    torch.save(save_dict, "{}{}_pytorch_{}{}.{}".format(syspath, ball_model_name, args.model, other, extension))

def load_model(m_args, syspath, sub_name_eng, model, optimizer, lr_scheduler, scaler, sub_name="红球", other=""):
    global best_score, start_dt, best_loss
    _test_list = []
    current_epoch = 0
    no_update_times = 0
    split_time = args.split_time
    address = "{}{}_ball_model_pytorch_{}{}.{}".format(syspath, sub_name_eng, args.model, other, extension)
    if os.path.exists(address):
        # model.load_state_dict(torch.load("{}{}_ball_model_pytorch.ckpt".format(syspath, sub_name_eng)))
        checkpoint = torch.load(address, map_location=device)
        if 'windows_size' in checkpoint  and 'hidden_size' in checkpoint and 'num_layers' in checkpoint and 'num_heads' in checkpoint:
            if checkpoint['windows_size'] != args.windows_size or  checkpoint['hidden_size'] != args.hidden_size or checkpoint['num_layers'] != args.num_layers or checkpoint['num_heads'] != args.num_heads:
                logger.info("模型参数不一致！")
                logger.info("保存的参数为: windows_size: {}, hidden_size: {}, num_layers: {}, num_heads: {}".format(checkpoint['windows_size'], checkpoint['hidden_size'], checkpoint['num_layers'], checkpoint['num_heads']))
                if args.train_mode in [0, 2]:
                    logger.info("当前为继续训练模式，将自动调整训练参数！")
                    args.windows_size = checkpoint['windows_size']
                    args.hidden_size = checkpoint['hidden_size']
                    args.num_layers = checkpoint['num_layers']
                    args.num_heads = checkpoint['num_heads']
                    if args.model == "Transformer":
                        model = _model(input_size=m_args["model_args"]["{}_n_class".format(sub_name_eng)]*m_args["model_args"]["windows_size"], 
                                       output_size=m_args["model_args"]["{}_n_class".format(sub_name_eng)], 
                                       hidden_size=args.hidden_size, 
                                       num_layers=args.num_layers, 
                                       num_heads=args.num_heads, 
                                       dropout=0.5).to(device)
                    elif args.model == "LSTM":
                        model = _model(input_size=m_args["model_args"]["{}_sequence_len".format(sub_name_eng)], 
                                       output_size=m_args["model_args"]["{}_sequence_len".format(sub_name_eng)]*m_args["model_args"]["{}_n_class".format(sub_name_eng)], 
                                       hidden_size=args.hidden_size, 
                                       num_layers=args.num_layers, 
                                       num_heads=args.num_heads, 
                                       dropout=0.5, 
                                       num_embeddings=m_args["model_args"]["{}_n_class".format(sub_name_eng)], 
                                       embedding_dim=50, 
                                       windows_size=int(args.windows_size)).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=args.lr)
                    lr_scheduler = modeling.CustomSchedule(optimizer=optimizer, 
                                                           d_model=args.hidden_size, 
                                                           warmup_steps=model_args[args.name]["model_args"]["{}_epochs".format(sub_name_eng)]*0.2)
                else:
                    logger.info("请修改参数或重新训练！")
                    sys.exit()
        else:
            logger.info("模型不是最新版本，建议重新训练！")
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.init != 1:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'epoch' in checkpoint:
                current_epoch = checkpoint['epoch']
                if current_epoch >= model_args[args.name]["model_args"]["{}_epochs".format(sub_name_eng)] - 1:
                    current_epoch = 0
            if 'no_update_times' in checkpoint:
                no_update_times = checkpoint['no_update_times']
            if 'split_time' in checkpoint:
                split_time = checkpoint['split_time']
            if 'test_list' in checkpoint:
                _test_list = checkpoint['test_list']
            if 'best_lost' in checkpoint:
                best_loss = checkpoint['best_lost']
            if split_time < 0 and len(_test_list) <= 0:
                logger.warning("测试数据集丢失，请重新训练！")
                sys.exit()
        if 'start_dt' in checkpoint:
            start_dt = checkpoint['start_dt']
        if 'best_score' in checkpoint:
            best_score = checkpoint['best_score']
        logger.info("已加载{}模型！".format(sub_name))
    else:
        logger.info("没有找到{}模型，将重新训练！".format(sub_name))
    return current_epoch, no_update_times, split_time, _test_list

def train_ball_model(name, dataset, test_dataset, sub_name="红球"):
    """ 模型训练
    :param name: 玩法
    :param x_data: 训练样本
    :param y_data: 训练标签
    :return:
    """
    global last_save_time, best_score, start_dt, test_list, red_train_data, red_test_data, blue_train_data, blue_test_data, best_loss
    _test_list = []
    sub_name_eng = "red" if sub_name == "红球" else "blue"
    ball_model_name = red_ball_model_name if sub_name == "红球" else blue_ball_model_name
    m_args = model_args[name]
    syspath = model_path + model_args[args.name]["pathname"]['name'] + str(m_args["model_args"]["windows_size"]) + model_args[args.name]["subpath"][sub_name_eng]
    if not os.path.exists(syspath):
        os.makedirs(syspath)
    logger.info("标签数据维度: {}".format(dataset.data.shape))
    # 定义模型和优化器
    if args.model == "Transformer":
        model = _model(input_size=m_args["model_args"]["{}_n_class".format(sub_name_eng)]*m_args["model_args"]["windows_size"], 
                       output_size=m_args["model_args"]["{}_n_class".format(sub_name_eng)], 
                       hidden_size=args.hidden_size, 
                       num_layers=args.num_layers, 
                       num_heads=args.num_heads, 
                       dropout=0.5).to(device)
    elif args.model == "LSTM":
        model = _model(input_size=m_args["model_args"]["{}_sequence_len".format(sub_name_eng)], 
                       output_size=m_args["model_args"]["{}_sequence_len".format(sub_name_eng)]*m_args["model_args"]["{}_n_class".format(sub_name_eng)], 
                       hidden_size=args.hidden_size, 
                       num_layers=args.num_layers, 
                       num_heads=args.num_heads, 
                       dropout=0.5, 
                       num_embeddings=m_args["model_args"]["{}_n_class".format(sub_name_eng)], 
                       embedding_dim=50, 
                       windows_size=int(args.windows_size)).to(device)
    # criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss() # 二分类交叉熵
    # criterion = nn.BCELoss() # 二分类交叉熵
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # lr_scheduler=modeling.CustomSchedule(d_model=args.hidden_size, optimizer=optimizer)
    lr_scheduler = modeling.CustomSchedule(optimizer=optimizer, 
                                           d_model=args.hidden_size, 
                                           warmup_steps=model_args[args.name]["model_args"]["{}_epochs".format(sub_name_eng)]*0.2)
    current_epoch = 0
    no_update_times = 0
    split_time = args.split_time
    _other = ""
    scaler = GradScaler()
    if args.train_mode != 1:
        if args.train_mode in [2, 3]:
            if args.train_mode == 2:
                _files = glob.glob(os.path.join(syspath, '*best_test*'))
            else:
                _files = glob.glob(os.path.join(syspath, '*best_loss*'))
            if len(_files) <= 0:
                logger.info("模型没有最优版本，将读取最后版本继续训练！")
            else:
                newest_file = os.path.basename(max(_files, key=os.path.getmtime)).split('_')
                if len(newest_file) == 8:
                    _other = '_' + newest_file[-3] +'_' + newest_file[-2] + '_' + newest_file[-1].split('.')[0]
                    logger.info("模型最优版本是：{}， 系统将尝试读取...".format(os.path.basename(max(_files, key=os.path.getmtime))),)
                else:
                    logger.info("模型没有最优版本，将读取最后版本继续训练！")
        elif args.train_mode == 0:
            logger.info("系统将尝试读取最后版本继续训练！")
        current_epoch, no_update_times, split_time, _test_list = load_model(m_args, syspath, sub_name_eng, model, optimizer, 
                                                                            lr_scheduler, scaler, sub_name, other=_other)
    else:
        logger.info("系统将重新训练！")
    if split_time != args.split_time or (split_time < 0 and set(_test_list) != set(test_list) and len(_test_list) > 0):
        logger.info("读取已保存的测试数据集，将重新载入数据！")
        args.split_time = split_time
        test_list = _test_list
        red_train_data = create_train_data(name=args.name, windows=model_args[name]["model_args"]["windows_size"], 
                                           dataset=1, ball_type="red", cq=args.cq, test_flag=0, test_begin=args.split_time, 
                                           f_data=0, model=args.model, num_classes=model_args[name]["model_args"]["red_n_class"], 
                                           test_list=test_list)
        if args.split_time != 0:
            red_test_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"], 1, "red", args.cq, 1, 
                                              args.split_time, model=args.model, num_classes=model_args[name]["model_args"]["red_n_class"], 
                                              test_list=test_list)
        if name not in ["kl8"]:
            blue_train_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"], 1, "blue", args.cq, 0, 
                                                args.split_time, model=args.model, num_classes=model_args[name]["model_args"]["blue_n_class"], 
                                                test_list=test_list)
            if args.split_time != 0:
                blue_test_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"], 1, "blue", args.cq, 1, 
                                                   args.split_time, model=args.model, num_classes=model_args[name]["model_args"]["blue_n_class"], 
                                                   test_list=test_list)
        if sub_name_eng == "red":
            dataset = red_train_data
            test_dataset = red_test_data
        elif sub_name_eng == "blue":
            dataset = blue_train_data
            test_dataset = blue_test_data
    dataloader = DataLoaderX(dataset, batch_size=model_args[args.name]["model_args"]["batch_size"], shuffle=False, 
                             num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoaderX(test_dataset, batch_size=model_args[args.name]["model_args"]["batch_size"], shuffle=False, 
                                  num_workers=args.num_workers, pin_memory=True)
    if args.init == 1:
        current_epoch = 0
        no_update_times = 0
        scaler = GradScaler()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = modeling.CustomSchedule(optimizer=optimizer, d_model=args.hidden_size, 
                                               warmup_steps=model_args[args.name]["model_args"]["{}_epochs".format(sub_name_eng)]*0.2)   
    logger.info("当前epoch是 {}, 初次启动时间是 {}, 最佳分数是 {:.2e}, 最佳损失是 {:.2e}".format(current_epoch, start_dt, best_score, best_loss))
    pbar = tqdm(range(model_args[args.name]["model_args"]["{}_epochs".format(sub_name_eng)]))
    running_loss = 0.0
    running_times = 0
    test_loss = 0.0
    test_times = 0
    topk_loss = 0.0
    topk_times = 0
    top_loss = 0.0
    top_times = 0
    for epoch in range(current_epoch, model_args[args.name]["model_args"]["{}_epochs".format(sub_name_eng)]):
        no_update_times += 1
        if no_update_times > args.ext_times and args.plus_mode == 1:
            print()
            no_update_times = 0
            if args.save_best_loss == 0:
                _, _, _, _ = load_model(m_args, syspath, sub_name_eng, model, optimizer, lr_scheduler, scaler, sub_name, 
                                        other="_{}_{}".format(start_dt, "best_test"))
            else:
                _, _, _, _ = load_model(m_args, syspath, sub_name_eng, model, optimizer, lr_scheduler, scaler, sub_name, 
                                        other="_{}_{}".format(start_dt, "best_loss"))
        if epoch == current_epoch:
            pbar.update(current_epoch)
        running_loss = 0.0
        running_times = 0
        for batch in dataloader:
            model.train()
            running_times += 1
            x, y = batch
            x = x.float().to(device)
            y = y.float().to(device)
            optimizer.zero_grad()
            with autocast():
                y_pred = model(x).view(-1, m_args["model_args"]["{}_sequence_len".format(sub_name_eng)], 
                                       m_args["model_args"]["{}_n_class".format(sub_name_eng)])
                t_loss = criterion(y_pred.transpose(1,2), y.long().view(y.size(0), -1))
            scaler.scale(t_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.zero_grad()
            # t_loss.backward()
            # optimizer.step()
            # running_loss += t_loss.item() * x.size(0)
            running_loss += t_loss.item()
        # print(f"Epoch {epoch+1}: Loss = {running_loss / len(dataset):.4f}")
        lr_scheduler.step()
        if (epoch + 1) % save_epoch == 0:
            if time.time() - last_save_time > save_interval:
                last_save_time = time.time()
                save_model(model, optimizer, lr_scheduler, scaler, epoch, syspath, ball_model_name, no_update_times=no_update_times)
            if args.split_time != 0 and test_dataset is not None and  test_dataset.__len__() > 0:
                # run test
                model.eval()
                with torch.no_grad():
                    test_loss = 0.0
                    test_times = 0
                    topk_loss = 0.0
                    topk_times = 0
                    totalK_correct = 0.0
                    top_loss = 0.0
                    top_times = 0
                    total_correct = 0.0
                    for batch in test_dataloader:
                        test_times += 1
                        x, y = batch
                        x = x.float().to(device)
                        y = y.float().to(device)
                        with autocast():
                            y_pred = model(x).view(-1, m_args["model_args"]["{}_sequence_len".format(sub_name_eng)], 
                                                   m_args["model_args"]["{}_n_class".format(sub_name_eng)])
                            # _, targets = torch.squeeze(y, 1).max(dim=1)
                            tt_loss = criterion(y_pred.transpose(1,2), y.long().view(y.size(0), -1)) 
                            # tt_loss = criterion(y_pred, torch.squeeze(y, 1))
                        # test_loss += tt_loss.item() * x.size(0)
                        test_loss += tt_loss.item()
                        # calculate topk loss
                        if args.name not in ["kl8"]:
                            args.top_k = m_args["model_args"]["{}_sequence_len".format(sub_name_eng)]
                        if args.model == "Transformer":
                            probs, indices = torch.topk(y_pred, args.top_k, dim=1)
                            for i in range(x.size(0)):
                                topk_times += args.top_k
                                target_indices = y[i].nonzero(as_tuple=False).squeeze()
                                totalK_correct += sum([1 for j in indices[i] if j in target_indices])
                            probs, indices = torch.topk(y_pred, m_args["model_args"]["{}_sequence_len".format(sub_name_eng)], dim=1)
                            for i in range(x.size(0)):
                                top_times += m_args["model_args"]["{}_sequence_len".format(sub_name_eng)]
                                target_indices = y[i].nonzero(as_tuple=False).squeeze()
                                total_correct += sum([1 for j in indices[i] if j in target_indices])
                        elif args.model == "LSTM":
                            for i in range(x.size(0)):
                                softmax = nn.Softmax(dim=1)
                                _ele = modeling.decode_one_hot(softmax(y_pred[i]), sort_by_max_value=True, 
                                                               num_classes=m_args["model_args"]["{}_n_class".format(sub_name_eng)])
                                topk_times += args.top_k
                                top_times += m_args["model_args"]["{}_sequence_len".format(sub_name_eng)]
                                # target_indices = y[i].nonzero(as_tuple=False).squeeze()
                                # target_indices = modeling.decode_one_hot(y[i], num_classes=m_args["model_args"]["{}_n_class".format(sub_name_eng)])
                                target_indices = (y+1).view(y.size(0), -1).tolist()[i]
                                target_indices_set = set(target_indices)
                                _eleK_set = set(_ele[0:args.top_k])
                                _ele_set = set(_ele)
                                totalK_correct += len(target_indices_set & _eleK_set)
                                total_correct += len(target_indices_set & _ele_set)
                                # totalK_correct += sum([1 for j in _ele[0:args.top_k] if j in target_indices])
                                # total_correct += sum([1 for j in _ele if j in target_indices])
                    # logger.info("Epoch {}/{} Test Loss: {:.2e}".format(epoch+1, model_args[args.name]["model_args"]["{}_epochs".format(sub_name_eng)], test_loss / len(test_dataset)))
                topk_loss = 1 - totalK_correct / (topk_times if topk_times > 0 else 1)
                top_loss = 1 - total_correct / (top_times if top_times > 0 else 1)
                if top_loss < best_score:
                    no_update_times = 0
                    best_score = top_loss
                    save_model(model, optimizer, lr_scheduler, scaler, epoch, syspath, ball_model_name, 
                               other="_{}_{}".format(start_dt, "best_test"), no_update_times=no_update_times)
                if topk_loss < best_score:
                    no_update_times = 0
                    best_score = topk_loss
                    save_model(model, optimizer, lr_scheduler, scaler, epoch, syspath, ball_model_name, 
                               other="_{}_{}".format(start_dt, "best_test"), no_update_times=no_update_times)
            if args.save_best_loss > 0 and best_loss > running_loss / (running_times if running_times > 0 else 1):
                best_loss = running_loss / (running_times if running_times > 0 else 1)
                save_model(model, optimizer, lr_scheduler, scaler, epoch, syspath, ball_model_name, 
                           other="_{}_{}".format(start_dt, "best_loss"), no_update_times=no_update_times)
        if args.tensorboard == 1:
            writer.add_scalar('Loss/Running', running_loss / (running_times if running_times > 0 else 1), epoch)
            if (epoch + 1) % save_epoch == 0:
                writer.add_scalar('Loss/Test', test_loss / (test_times if test_times > 0 else 1), epoch)
                writer.add_scalar('Loss/TopK{}'.format(args.top_k), topk_loss, epoch)
                writer.add_scalar('Loss/Top', top_loss, epoch)
        pbar.set_description("AL:{:.2e} TL:{:.2e} BL:{:.2e} KL{}:{:.2e} KL:{:.2e} LR:{:.2e} HS:{:.2e}".format(
                            running_loss / (running_times if running_times > 0 else 1), 
                            test_loss / (test_times if test_times > 0 else 1), 
                            best_loss, 
                            args.top_k, 
                            topk_loss, 
                            top_loss, 
                            optimizer.param_groups[0]['lr'], 
                            best_score
                            ))
        pbar.update(1)
    if args.tensorboard == 1:
        writer.close()
    save_model(model, optimizer, lr_scheduler, scaler, epoch, syspath, ball_model_name, other="_{}".format(start_dt), no_update_times=no_update_times)
    pbar.set_description("AL:{:.2e} TL:{:.2e} BL:{:.2e} KL{}:{:.2e} KL:{:.2e} LR:{:.2e} HS:{:.2e}".format(
                        running_loss / (running_times if running_times > 0 else 1), 
                        test_loss / (test_times if test_times > 0 else 1), 
                        best_loss, 
                        args.top_k, 
                        topk_loss, 
                        top_loss, 
                        optimizer.param_groups[0]['lr'], 
                        best_score
                        ))
    pbar.close()
    print()
    logger.info("【{}】{}模型训练完成!".format(name_path[name]["name"], sub_name))

def action(name):
    global best_score, test_list, red_train_data, red_test_data, blue_train_data, blue_test_data
    logger.info("正在创建【{}】数据集...".format(name_path[name]["name"]))
    if args.split_time < 0 and len(test_list) <= 0:
        logger.info("抽取测试数据...")
        ori_data = None
        if args.cq == 1 and name == "kl8":
            ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_cq_file_name))
        else:
            ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
        n = -1 * args.split_time
        if n <= 100:
            n_samples = int(len(ori_data['期数'].unique()) * n / 100)
        if n > 100:
            n_samples = 1
        test_list = sorted(ori_data['期数'].drop_duplicates().sample(n_samples).tolist())
        if n > 100:
            _n_samples = int(len(ori_data['期数'].unique()) * (n - 100) / 100)
            while int(ori_data[ori_data['期数'] == test_list[0]]['Unnamed: 0']) < _n_samples:
                test_list = sorted(ori_data['期数'].drop_duplicates().sample(_n_samples).tolist())
            for item in range(int(ori_data[ori_data['期数'] == test_list[0]]['Unnamed: 0']) - 1, int(ori_data[ori_data['期数'] == test_list[0]]['Unnamed: 0']) - _n_samples, -1):
                test_list.append(int(ori_data[ori_data['Unnamed: 0'] == item]['期数']))
    # name, windows, dataset=0, ball_type="red", cq=0, test_flag=0, test_begin=2021351, f_data=0, model="Transformer"
    red_train_data = create_train_data(name=args.name, windows=model_args[name]["model_args"]["windows_size"], 
                                       dataset=1, ball_type="red", cq=args.cq, test_flag=0, test_begin=args.split_time, 
                                       f_data=0, model=args.model, num_classes=model_args[name]["model_args"]["red_n_class"], test_list=test_list)
    if args.split_time != 0:
        red_test_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"], 1, "red", 
                                          args.cq, 1, args.split_time, model=args.model, num_classes=model_args[name]["model_args"]["red_n_class"], 
                                          test_list=test_list)
    if name not in ["kl8"]:
        blue_train_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"], 1, "blue", 
                                            args.cq, 0, args.split_time, model=args.model, num_classes=model_args[name]["model_args"]["blue_n_class"], 
                                            test_list=test_list)
        if args.split_time != 0:
            blue_test_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"], 1, "blue", 
                                               args.cq, 1, args.split_time, model=args.model, num_classes=model_args[name]["model_args"]["blue_n_class"], 
                                               test_list=test_list)
    for i in range(args.epochs):
        if model_args[name]["model_args"]["red_epochs"] > 0:
            best_score = 999999999
            logger.info("开始训练【{}】红球模型...".format(name_path[name]["name"]))
            start_time = time.time()
            train_ball_model(name, dataset=red_train_data, test_dataset=red_test_data, sub_name="红球")
            logger.info("训练耗时: {:.4f}".format(time.time() - start_time))

        if name not in ["pls", "kl8"] and model_args[name]["model_args"]["blue_epochs"] > 0:
            best_score = 999999999
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
