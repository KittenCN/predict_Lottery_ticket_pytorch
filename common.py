# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""
from math import log
import requests
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
import torch
from torch import nn
import torch.nn.functional as F
from config import *
import datetime
import numpy as np
import modeling
from torch.utils.data import DataLoader
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ori_data = None
filedata = []
filetitle = []
pred_key_d = {}
mini_args = {}

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # 输入：inputs (模型预测，shape: [batch_size, num_classes]), 
        # targets (真实标签，shape: [batch_size, num_classes], 独热编码)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.float32)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)  # alpha系数调整
        pt = torch.exp(-BCE_loss)  # 转换为概率
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

def create_train_data(name, windows, dataset=0, ball_type="red", cq=0, test_flag=0, test_begin=2021351, f_data=0, model="Transformer"):
    """ 创建训练数据
    :param name: 玩法，双色球/大乐透
    :param windows: 训练窗口
    :return:
    """
    global ori_data
    strflag = "训练" if test_flag == 0 else "测试"
    strball = "红球" if ball_type == "red" else "蓝球"
    if ori_data is None:
        if cq == 1 and name == "kl8":
            ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_cq_file_name))
        else:
            ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
    data = ori_data.copy()
    if f_data == 0:
        if test_flag == 0:
            data = data[data['期数'] > test_begin]
        else:
            data = data[data['期数'] <= test_begin]
    else:
        data = data[data['期数'] <= f_data]
        data = data.head(windows + 1)
    if not len(data):
        raise logger.error(" 请执行 get_data.py 进行数据下载！")
    else:
        # 创建模型文件夹
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        # logger.info(strball + strflag + "数据已加载! ")

    data = data.iloc[:, 2:].values
    tmp = []
    for _data in data:
        _tmp = []
        for item in _data:
           _tmp.append([item])
        tmp.append(_tmp)
    data = np.array(tmp)       
    cut_num = model_args[name]["model_args"]["red_sequence_len"]
    if dataset == 0:
        x_data, y_data = [], []
        for i in range(len(data) - windows - 1):
            sub_data = data[i:(i+windows+1), :]
            x_data.append(sub_data[1:])
            y_data.append(sub_data[0])

        return {
            "red": {
                "x_data": np.array(x_data)[:, :, :cut_num], "y_data": np.array(y_data)[:, :cut_num]
            },
            "blue": {
                "x_data": np.array(x_data)[:, :, cut_num:], "y_data": np.array(y_data)[:, cut_num:]
            }
        }
    else:
        if ball_type == "red":
            dataset = modeling.MyDataset(data, windows, cut_num, model)
        else:
            dataset = modeling.MyDataset(data, windows, cut_num * -1, model)
        logger.info(strball + strflag + "集数据维度: {}".format(dataset.data.shape))
        return dataset


def get_data_run(name, cq=0):
    """
    :param name: 玩法名称
    :return:
    """
    current_number = get_current_number(name)
    logger.info("【{}】最新一期期号：{}".format(name_path[name]["name"], current_number))
    logger.info("正在获取【{}】数据。。。".format(name_path[name]["name"]))
    if not os.path.exists(name_path[name]["path"]):
        os.makedirs(name_path[name]["path"])
    if cq == 1 and name == "kl8":
        data = spider_cq(name, 1, current_number, "train")
    else:
        data = spider(name, 1, current_number, "train")
    if "data" in os.listdir(os.getcwd()):
        logger.info("【{}】数据准备就绪，共{}期, 下一步可训练模型...".format(name_path[name]["name"], len(data)))
    else:
        logger.error("数据文件不存在！")

def get_url(name):
    """
    :param name: 玩法名称
    :return:
    """
    url = "https://datachart.500.com/{}/history/".format(name)
    path = "newinc/history.php?start={}&end={}&limit={}"
    if name == "qxc" or name == "pls":
        path = "inc/history.php?start={}&end={}&limit={}"
    elif name == "kl8":
        url = "https://datachart.500.com/{}/zoushi/".format(name)
        path = "newinc/jbzs_redblue.php?from=&to=&shujcount=0&sort=1&expect=-1"
    return url, path

def get_current_number(name):
    """ 获取最新一期数字
    :return: int
    """
    url, _ = get_url(name)
    if name in ["qxc", "pls"]:
        r = requests.get("{}{}".format(url, "inc/history.php"), verify=False)
    elif name in ["ssq", "dlt"]:
        r = requests.get("{}{}".format(url, "history.shtml"), verify=False)
    elif name in ["kl8"]:
        r = requests.get("{}{}".format(url, "newinc/jbzs_redblue.php"), verify=False)
    r.encoding = "gb2312"
    soup = BeautifulSoup(r.text, "lxml")
    if name in ["kl8"]:
        current_num = soup.find("div", class_="wrap_datachart").find("input", id="to")["value"]
    else:
        current_num = soup.find("div", class_="wrap_datachart").find("input", id="end")["value"]
    return current_num

def spider_cq(name="kl8", start=1, end=999999, mode="train", windows_size=0):
    syspath = name_path[name]["path"]
    if not os.path.exists(syspath):
        os.makedirs(syspath)
    if name == "kl8" and mode == "train":
        url = "https://data.917500.cn/kl81000_cq_asc.txt"
        r = requests.get(url, headers = {'User-agent': 'chrome'})
        data = []
        lines = sorted(r.text.split('\n'), reverse=True)
        for line in lines:
            if len(line) < 10:
                continue
            item = dict()
            line = line.split(',')
            line = line[0].split(' ')
            # item[u"id"] = line[0]
            strdate = line[1].split('-')
            item[u"日期"] = strdate[0] + strdate[1] + strdate[2]
            item[u"期数"] = line[0]  
            for i in range(1, 21):
                item[u"红球_{}".format(i)] = line[i + 1]
            data.append(item)
        df = pd.DataFrame(data)
        df.to_csv("{}{}".format(syspath, data_cq_file_name), encoding="utf-8",index=False)
        return pd.DataFrame(data)
    elif name == "kl8" and mode == "predict":
        ori_data = pd.read_csv("{}{}".format(syspath, data_cq_file_name))  
        data = []
        if windows_size > 0:
            ori_data = ori_data[0:windows_size]
        for i in range(len(ori_data)):
            item = dict()
            item[u"期数"] = ori_data.iloc[i, 1]
            for j in range(20):
                item[u"红球_{}".format(j+1)] = ori_data.iloc[i, j+2]
            data.append(item)
        return pd.DataFrame(data)
    else:
        spider(name, start, end, mode)

def spider(name="ssq", start=1, end=999999, mode="train", windows_size=0):
    """ 爬取历史数据
    :param name 玩法
    :param start 开始一期
    :param end 最近一期
    :param mode 模式，train：训练模式，predict：预测模式（训练模式会保持文件）
    :return:
    """
    syspath = name_path[name]["path"]
    if not os.path.exists(syspath):
        os.makedirs(syspath)
    if mode == "train":
        url, path = get_url(name)
        limit = int(end) - int(start) + 1
        url = "{}{}".format(url, path.format(int(start), int(end), limit))
        r = requests.get(url=url, verify=False)
        r.encoding = "gb2312"
        soup = BeautifulSoup(r.text, "lxml")
        if name in ["ssq", "dlt", "kl8"]:
            trs = soup.find("tbody", attrs={"id": "tdata"}).find_all("tr")
        elif name in ["qxc", "pls"]:
            trs = soup.find("div", class_="wrap_datachart").find("table", id="tablelist").find_all("tr")
        data = []
        for tr in trs:
            item = dict()
            if name == "ssq":
                item[u"期数"] = tr.find_all("td")[0].get_text().strip()
                for i in range(6):
                    item[u"红球_{}".format(i+1)] = tr.find_all("td")[i+1].get_text().strip()
                item[u"蓝球"] = tr.find_all("td")[7].get_text().strip()
                data.append(item)
            elif name == "dlt":
                item[u"期数"] = tr.find_all("td")[0].get_text().strip()
                for i in range(5):
                    item[u"红球_{}".format(i+1)] = tr.find_all("td")[i+1].get_text().strip()
                for j in range(2):
                    item[u"蓝球_{}".format(j+1)] = tr.find_all("td")[6+j].get_text().strip()
                data.append(item)
            elif name == "pls":
                if tr.find_all("td")[0].get_text().strip() == "注数" or tr.find_all("td")[1].get_text().strip() == "中奖号码":
                    continue
                item[u"期数"] = tr.find_all("td")[0].get_text().strip()
                numlist = tr.find_all("td")[1].get_text().strip().split(" ")
                for i in range(3):
                    item[u"红球_{}".format(i+1)] = numlist[i]
                data.append(item)
            elif name == "kl8":
                tds = tr.find_all("td")
                index = 1
                for td in tds:
                    if td.has_attr('align') and td['align'] == 'center':
                        item[u"期数"] = td.get_text().strip()
                    elif td.has_attr('class') and td['class'][0] == 'chartBall01':
                        item[u"红球_{}".format(index)] = td.get_text().strip()
                        index += 1
                if item:
                    data.append(item)
            else:
                logger.warning("抱歉，没有找到数据源！")

        df = pd.DataFrame(data)
        df.to_csv("{}{}".format(syspath, data_file_name), encoding="utf-8")
        return pd.DataFrame(data)

    elif mode == "predict":
        ori_data = pd.read_csv("{}{}".format(syspath, data_file_name))  
        data = []
        if windows_size > 0:
            ori_data = ori_data[0:windows_size]
        for i in range(len(ori_data)):
            item = dict()
            if (ori_data.iloc[i, 1] < int(start) or ori_data.iloc[i, 1] > int(end)) and windows_size == 0:
                continue
            if name == "ssq":
                item[u"期数"] = ori_data.iloc[i, 1]
                for j in range(6):
                    item[u"红球_{}".format(j+1)] = ori_data.iloc[i, j+2]
                item[u"蓝球"] = ori_data.iloc[i, 8]
                data.append(item)
            elif name == "dlt":
                item[u"期数"] = ori_data.iloc[i, 1]
                for j in range(5):
                    item[u"红球_{}".format(j+1)] = ori_data.iloc[i, j+2]
                for k in range(2):
                    item[u"蓝球_{}".format(k+1)] = ori_data.iloc[i, 7+k]
                data.append(item)
            elif name == "pls":
                item[u"期数"] = ori_data.iloc[i, 1]
                for j in range(3):
                    item[u"红球_{}".format(j+1)] = ori_data.iloc[i, j+2]
                data.append(item)
            elif name == "kl8":
                item[u"期数"] = ori_data.iloc[i, 1]
                for j in range(20):
                    item[u"红球_{}".format(j+1)] = ori_data.iloc[i, j+2]
                data.append(item)
            else:
                logger.warning("抱歉，没有找到数据源！")
        return pd.DataFrame(data)

# current_number = get_current_number(mini_args.name)

def setMiniargs(args):
    global mini_args
    mini_args = args

def init():
    global mini_args,pred_key_d, filedata, filetitle
    filedata = []
    filetitle = []
    pred_key_d = {}
    mini_args = {}

def predict_ball_model(name, dataset, sequence_len, sub_name="红球", window_size=1, hidden_size=128, num_layers=8, num_heads=16, input_size=20, output_size=20, model_name="Transformer"):
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
    ball_index = 0 if sub_name == "红球" else 1
    name_list = [(ball_name[ball_index], i + 1) for i in range(sequence_len)]
    syspath = model_path + model_args[mini_args.name]["pathname"]['name'] + str(window_size) + model_args[mini_args.name]["subpath"][sub_name_eng]
    if not os.path.exists(syspath):
        os.makedirs(syspath)
    logger.info("标签数据维度: {}".format(dataset.data.shape))

    dataset = [dataset[0]]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 定义模型和优化器
    if model_name == "Transformer":
        _model = modeling.Transformer_Model
    elif model_name == "LSTM":
        _model = modeling.LSTM_Model
    model = _model(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, dropout=0.1).to(modeling.device)
    if os.path.exists("{}{}_ball_model_pytorch_{}.ckpt".format(syspath, sub_name_eng, model_name)):
        # model.load_state_dict(torch.load("{}{}_ball_model_pytorch.ckpt".format(syspath, sub_name_eng)))
        checkpoint = torch.load("{}{}_ball_model_pytorch_{}.ckpt".format(syspath, sub_name_eng, model_name))
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("已加载{}模型！".format(sub_name))
    for batch in dataloader:
        x, y = batch
        x = x.to(modeling.device)
        y = y.to(modeling.device)
        y_pred = model(x.float())
    return y_pred, name_list

def run_predict(window_size, sequence_len, hidden_size=128, num_layers=8, num_heads=16, input_size=20, output_size=20, f_data=0, model="Transformer"):
    global pred_key_d
    balls = ['red', 'blue'] if mini_args.name not in ["pls", "kl8"] else ['red']
    for sub_name_eng in balls:
        sub_name = "红球" if sub_name_eng == "red" else "蓝球"
        if window_size != 0:
            model_args[mini_args.name]["model_args"]["windows_size"] = window_size
        syspath = model_path + model_args[mini_args.name]["pathname"]['name'] + str(mini_args.windows_size) + model_args[mini_args.name]["subpath"][sub_name_eng]
        # redpath = model_path + model_args[mini_args.name]["pathname"]['name'] + str(model_args[mini_args.name]["model_args"]["windows_size"]) + model_args[mini_args.name]["subpath"]['red']
        # bluepath = model_path + model_args[mini_args.name]["pathname"]['name'] + str(model_args[mini_args.name]["model_args"]["windows_size"]) + model_args[mini_args.name]["subpath"]['blue']
        # model = modeling.TransformerModel(input_size=20, output_size=20).to(modeling.device)
        if os.path.exists("{}{}_ball_model_pytorch_{}.ckpt".format(syspath, sub_name_eng, model)):
            # model.load_state_dict(torch.load("{}{}_ball_model_pytorch.ckpt".format(syspath, sub_name_eng)))
            # logger.info("已加载{}模型！窗口大小:{}".format(sub_name, model_args[mini_args.name]["model_args"]["windows_size"]))
            current_number = get_current_number(mini_args.name)
            logger.info("【{}】最近一期:{}".format(name_path[mini_args.name]["name"], current_number))
            logger.info("正在创建【{}】数据集...".format(name_path[mini_args.name]["name"]))
            data = create_train_data(mini_args.name, model_args[mini_args.name]["model_args"]["windows_size"], 1, sub_name_eng, mini_args.cq,f_data=f_data, model=model)
            y_pred, name_list = predict_ball_model(mini_args.name, data, sequence_len, sub_name, window_size,hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, input_size=input_size, output_size=output_size, model_name=model)
            logger.info("预测{}结果为: \n".format(sub_name))
            if mini_args.name in ["kl8"]:
                if model == "Transformer":
                    y_pred_list = modeling.binary_decode_array(y_pred.cpu(), threshold=0.25, top_k=80)
                    for row in y_pred_list:
                        row_limit = row[0:20]
                        logger.info("超过阈值的数据: {}".format(row))
                        logger.info("前20位超过阈值的数据: {}".format(row_limit))
                        logger.info("排序后前20位超过阈值的数据: {}".format(sorted(row_limit)))
                elif model == "LSTM":
                    y_pred_list = modeling.decode_one_hot(y_pred.cpu(), sort_by_max_value=True)
                    logger.info("前20位超过阈值的数据: {}".format(y_pred_list))
                    logger.info("排序后前20位超过阈值的数据: {}".format(sorted(y_pred_list)))
                
            else:
                y_pred_list = y_pred.cpu().tolist()
                for row in y_pred_list:
                    strrow = ""
                    for col in row:
                        strrow += str("{:.4f}".format(col))
                    logger.info(strrow)
        else:
            logger.warning("抱歉，没有找到{}模型！".format(sub_name))
            exit(0)

def get_year():
    """ 截取年份
    eg：2020-->20, 2021-->21
    :return:
    """
    return int(str(datetime.datetime.now().year)[-2:])


def try_error(name, predict_features, windows_size):
    """ 处理异常
    """
    if len(predict_features) != windows_size:
        logger.warning("期号出现跳期，期号不连续！开始查找最近上一期期号！本期预测时间较久！")
        last_current_year = (get_year() - 1) * 1000
        max_times = 160
        while len(predict_features) != windows_size:
            # predict_features = spider(name, last_current_year + max_times, get_current_number(name), "predict")[[x[0] for x in ball_name]]
            if mini_args.cq == 0:
                predict_features = spider(name, last_current_year + max_times, get_current_number(name), "predict", windows_size)
            else:
                predict_features = spider_cq(name, last_current_year + max_times, get_current_number(name), "predict", windows_size)
            # time.sleep(np.random.random(1).tolist()[0])
            max_times -= 1
        return predict_features
    return predict_features

# def get_final_result(name, mode=0):
#     """" 最终预测函数
#     """
#     m_args = model_args[name]["model_args"]
#     windows_size = model_args[name]["model_args"]["windows_size"]
#     current_number = get_current_number(mini_args.name)
#     logger.info("正在创建【{}】数据集...".format(name_path[name]["name"]))
#     red_data = create_train_data(name, windows_size, 1, "red")
#     blue_data = create_train_data(name, windows_size, 1, "blue")
#     logger.info("【{}】预测期号：{} 窗口大小:{}".format(name_path[name]["name"], int(current_number) + 1, windows_size))
#     if name == "ssq":
#         red_pred, red_name_list = get_red_ball_predict_result(red_data, m_args["sequence_len"], m_args["windows_size"])
#         blue_pred = get_blue_ball_predict_result(name, blue_data, 0, m_args["windows_size"])
#         ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list] + [ball_name[1][mode]]
#         pred_result_list = red_pred[0].tolist() + blue_pred.tolist()
#         return {
#             b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
#         }
#     elif name == "dlt":
#         red_pred, red_name_list = get_red_ball_predict_result(red_data, m_args["red_sequence_len"], m_args["windows_size"])
#         blue_pred, blue_name_list = get_blue_ball_predict_result(name, blue_data, m_args["blue_sequence_len"], m_args["windows_size"])
#         ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list] + ["{}_{}".format(name[mode], i) for name, i in blue_name_list]
#         pred_result_list = red_pred[0].tolist() + blue_pred[0].tolist()
#         return {
#             b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
#         }
#     elif name == "pls":
#         red_pred, red_name_list = get_red_ball_predict_result(red_data, m_args["red_sequence_len"], m_args["windows_size"])
#         ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list]
#         pred_result_list = red_pred[0].tolist()
#         return {
#             b_name: int(res) for b_name, res in zip(ball_name_list, pred_result_list)
#         }
#     elif name == "kl8":
#         red_pred, red_name_list = get_red_ball_predict_result(red_data, m_args["red_sequence_len"], m_args["windows_size"])
#         ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list]
#         pred_result_list = red_pred[0].tolist()
#         return {
#             b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
#         }

# def predict_run(name):
#     global filedata, filetitle
#     windows_size = model_args[name]["model_args"]["windows_size"]
#     diff_number = windows_size - 1
#     # logger.info("预测结果：{}".format(get_final_result(name, predict_features_)))
#     predict_dict = get_final_result(name)
#     ans = ""
#     _data = []
#     _title = []
#     for item in predict_dict:
#         if (item == "红球_1" or item == "红球"):
#             ans += "红球："
#         if (item == "蓝球_1" or item == "蓝球"):
#             ans += "蓝球："
#         ans += str(predict_dict[item]) + " "
#         _data.append(int(predict_dict[item]))
#         _title.append(item)
#     logger.info("预测结果：{}".format(ans))
#     filedata.append(_data.copy())
#     filetitle = _title.copy()
#     return filedata, filetitle


# if __name__ == "__main__":
#     spider_cq("kl8", "20180101", "20180110", "train")