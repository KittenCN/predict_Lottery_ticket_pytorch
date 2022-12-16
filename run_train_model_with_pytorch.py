import torch
import time
import json
import argparse
import numpy as np
import pandas as pd
import warnings
from config import *
from model import *
from loguru import logger

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="dlt", type=str, help="选择训练数据: 双色球/大乐透")
parser.add_argument('--windows_size', default='3', type=str, help="训练窗口大小,如有多个，用'，'隔开")
parser.add_argument('--red_epochs', default=1, type=int, help="红球训练轮数")
parser.add_argument('--blue_epochs', default=1, type=int, help="蓝球训练轮数")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pred_key = {}


def create_train_data(name, windows):
    """ 创建训练数据
    :param name: 玩法，双色球/大乐透
    :param windows: 训练窗口
    :return:
    """
    data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
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
    for i in range(len(data) - windows - 1):
        sub_data = data[i:(i+windows+1), :]
        x_data.append(sub_data[1:])
        y_data.append(sub_data[0])

    cut_num = 6 if name == "ssq" else 5
    return {
        "red": {
            "x_data": np.array(x_data)[:, :, :cut_num], "y_data": np.array(y_data)[:, :cut_num]
        },
        "blue": {
            "x_data": np.array(x_data)[:, :, cut_num:], "y_data": np.array(y_data)[:, cut_num:]
        }
    }

def train_red_ball_model(name, x_data, y_data):
    # self, batch_size, n_class, ball_num, w_size, embedding_size, words_size, hidden_size, layer_size
    m_args = model_args[name]
    if name not in ["pls"]:
        x_data = x_data - 1
        y_data = y_data - 1
    data_len = x_data.shape[0]
    logger.info("特征数据维度: {}".format(x_data.shape))
    logger.info("标签数据维度: {}".format(y_data.shape))
    
    model = LstmWithCRFModel(m_args["model_args"]["batch_size"], m_args["model_args"]["red_n_class"], m_args["model_args"]["sequence_len"],
                                m_args["model_args"]["windows_size"], m_args["model_args"]["red_embedding_size"], m_args["model_args"]["red_n_class"],
                                m_args["model_args"]["red_hidden_size"], m_args["model_args"]["red_layer_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=m_args["train_args"]["red_learning_rate"])
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_data).long(), torch.from_numpy(y_data).long())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=m_args["model_args"]["batch_size"], shuffle=True)
    for epoch in range(m_args["model_args"]["red_epochs"]):
        start_time = time.time()
        logger.info("开始训练第{}轮".format(epoch+1))
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs, loss, pred_sequence, viterbi_score = model(x, y, m_args["model_args"]["sequence_len"])
            loss.backward()
            optimizer.step()
        logger.info("第{}轮训练结束，耗时: {}s".format(epoch+1, time.time()-start_time))
        logger.info("第{}轮训练损失: {}".format(epoch+1, loss.item()))


def train_blue_ball_model(name, x_data, y_data):
    m_args = model_args[name]
    x_data = x_data - 1
    y_data = y_data - 1
    batch_size = m_args["model_args"]["batch_size"]
    n_iters = int(len(x_data) / batch_size)
    input_dim = m_args["model_args"]["blue_n_class"]
    time_step = m_args["model_args"]["windows_size"]
    hidden_dim = m_args["model_args"]["hidden_dim"]
    num_layers = m_args["model_args"]["num_layers"]
    output_dim = m_args["model_args"]["blue_n_class"]
    lr = m_args["model_args"]["lr"]
    num_epochs = m_args["model_args"]["blue_epochs"]
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for i in range(n_iters):
            start = i * batch_size
            end = start + batch_size
            inputs = torch.from_numpy(x_data[start:end]).to(device)
            labels = torch.from_numpy(y_data[start:end]).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                logger.info('Epoch: {}/{}, Step: {}/{}, Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, n_iters, loss.item()))
    torch.save(model.state_dict(), "{}{}_blue.pth".format(model_path, name))
    logger.info("蓝球模型训练完成！")

def train_model(name):
    data = create_train_data(name, model_args[name]["model_args"]["windows_size"])
    train_red_ball_model(name, data["red"]["x_data"], data["red"]["y_data"])
    # train_blue_ball_model(name, data["blue"]["x_data"], data["blue"]["y_data"])

if __name__ == "__main__":
    train_model("ssq")
