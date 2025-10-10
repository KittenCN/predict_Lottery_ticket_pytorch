# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""
import urllib3
import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import torch
import torch.nn.functional as F
import datetime
import numpy as np
from src import modeling
from torch.utils.data import DataLoader
from bs4 import BeautifulSoup
from loguru import logger
from torch import nn
from src.config import *

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_http_session(retries=2, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    """Create a requests.Session with retry/backoff configured."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def get_http_session_with_backoff(url, retries=3, backoff_base=0.5, allowlist=None, timeout=10):
    """Make a GET request to url with simple exponential backoff and optional domain allowlist.

    Returns the requests.Response on success or raises the last exception.
    """
    from urllib.parse import urlparse
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    # default allowlist from config
    try:
        from src.config import HTTP_ALLOWLIST, HTTP_CACHE_DIR
    except Exception:
        HTTP_ALLOWLIST = None
        HTTP_CACHE_DIR = None

    if allowlist is None:
        allowlist = HTTP_ALLOWLIST
    if allowlist is not None and hostname not in allowlist:
        raise ValueError(f"Host {hostname} not in allowlist")

    session = requests.Session()
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, verify=False, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            sleep_time = backoff_base * (2 ** (attempt - 1))
            logger.warning(f"Request to {url} failed (attempt {attempt}/{retries}): {e}; retrying in {sleep_time}s")
            import time
            time.sleep(sleep_time)
    # all attempts failed -> try cache fallback if configured
    logger.error(f"All retries failed for {url}: {last_exc}")
    try:
        if HTTP_CACHE_DIR:
            # create cache filename based on hostname + path
            import hashlib
            key = hashlib.sha256(url.encode('utf-8')).hexdigest()
            cache_path = os.path.join(HTTP_CACHE_DIR, f"{key}.cache")
            if os.path.exists(cache_path):
                logger.warning(f"Using cached response for {url} from {cache_path}")
                class CachedResp:
                    def __init__(self, text):
                        self.text = text
                        self.status_code = 200

                with open(cache_path, 'r', encoding='utf-8') as fh:
                    txt = fh.read()
                return CachedResp(txt)
    except Exception:
        pass
    raise last_exc


def to_multi_hot(y_values, num_classes: int, device=None):
    """Convert a 2D tensor/array of number indices (1-based) to multi-hot float32 tensor.

    Accepts torch.Tensor or numpy array. Returns torch.FloatTensor on given device (or cpu).
    """
    import numpy as _np
    import torch as _torch

    if device is None:
        device = _torch.device('cpu')

    # convert to torch tensor
    if isinstance(y_values, _np.ndarray):
        y_tensor = _torch.from_numpy(y_values)
    else:
        y_tensor = y_values

    if y_tensor.ndim == 1:
        y_tensor = y_tensor.unsqueeze(0)

    batch = y_tensor.size(0)
    multi_hot = _torch.zeros((batch, num_classes), dtype=_torch.float32, device=device)
    # y_tensor expected shape [batch, k]
    for i in range(batch):
        row = y_tensor[i]
        if row.numel() == 0:
            continue
        indices = row.long() - 1
        valid_mask = (indices >= 0) & (indices < num_classes)
        if valid_mask.any():
            valid_indices = indices[valid_mask]
            multi_hot[i, valid_indices] = 1.0
    return multi_hot

# Note: module-level `ori_data` global is deprecated. Use src.pipeline.DEFAULT_PIPELINE.get_ori_data(name, cq)
filedata = []
filetitle = []
pred_key_d = {}
# mini_args has been removed; use src.pipeline.DEFAULT_PIPELINE.args or pass args explicitly

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

def create_train_data(name, windows, dataset=0, ball_type="red", cq=0, test_flag=0, test_begin=0, f_data=0, model="Transformer", num_classes=80, test_list=[]):
    """ 创建训练数据
    :param name: 玩法，双色球/大乐透
    :param windows: 训练窗口
    :return:
    """
    strflag = "训练" if test_flag == 0 else "测试"
    strball = "红球" if ball_type == "red" else "蓝球"
    # Prefer pipeline cached ori_data when available to avoid module-level globals
    try:
        from src.pipeline import DEFAULT_PIPELINE
        if DEFAULT_PIPELINE is not None:
            ori_df = DEFAULT_PIPELINE.get_ori_data(name, cq=cq)
        else:
            raise Exception()
    except Exception:
        if cq == 1 and name == "kl8":
            ori_df = pd.read_csv("{}{}".format(name_path[name]["path"], data_cq_file_name))
        else:
            ori_df = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
    data = ori_df.copy()
    if test_begin >= 0 and len(test_list) <= 0:
        if f_data == 0:
            if test_flag in [0, 2]:
                data = data[data['期数'] > test_begin]
            else:
                data = data[data['期数'] <= test_begin]
        else:
            data = data[data['期数'] <= f_data]
            data = data.head(windows + 1)
    # elif len(test_list) > 0:
    #     if test_flag == 0:
    #         data = data[~data['期数'].isin(test_list)]
    #     else:
    #         data = data[data['期数'].isin(test_list)]

    if not len(data):
        raise logger.error(" 请执行 get_data.py 进行数据下载！")
    else:
        # 创建模型文件夹
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        # logger.info(strball + strflag + "数据已加载! ")

    data = data.iloc[:, :].values
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
            dataset = modeling.MyDataset(data, windows, cut_num, model, num_classes, test_flag, test_list, f_data)
        else:
            dataset = modeling.MyDataset(data, windows, cut_num * -1, model, num_classes, test_flag, test_list, f_data)
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
    # add timeout and simple retry
    try:
        if name in ["qxc", "pls"]:
            r = get_http_session_with_backoff("{}{}".format(url, "inc/history.php"), timeout=10)
        elif name in ["ssq", "dlt"]:
            r = get_http_session_with_backoff("{}{}".format(url, "history.shtml"), timeout=10)
        elif name in ["kl8"]:
            r = get_http_session_with_backoff("{}{}".format(url, "newinc/jbzs_redblue.php"), timeout=10)
    except Exception as e:
        logger.warning(f"请求失败({e})，请检查网络或稍后重试")
        raise
    r.encoding = "gb2312"
    soup = BeautifulSoup(r.text, "lxml")
    if name in ["kl8"]:
        current_num = soup.find("div", class_="wrap_datachart").find("input", id="to")["value"]
    else:
        current_num = soup.find("div", class_="wrap_datachart").find("input", id="end")["value"]
    return current_num

def spider_cq(name="kl8", start=1, end=999999, mode="train", seq_len=0):
    syspath = name_path[name]["path"]
    if not os.path.exists(syspath):
        os.makedirs(syspath)
    if name == "kl8" and mode == "train":
        url = "https://data.917500.cn/kl81000_cq_asc.txt"
        # timeout and retry via session
        session = get_http_session()
        try:
            r = get_http_session_with_backoff(url, retries=3, backoff_base=0.5, timeout=20)
            # try to cache successful response
            try:
                from src.config import HTTP_CACHE_DIR
                import hashlib
                if HTTP_CACHE_DIR:
                    os.makedirs(HTTP_CACHE_DIR, exist_ok=True)
                    key = hashlib.sha256(url.encode('utf-8')).hexdigest()
                    cache_path = os.path.join(HTTP_CACHE_DIR, f"{key}.cache")
                    with open(cache_path, 'w', encoding='utf-8') as fh:
                        fh.write(r.text)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"请求 {url} 失败({e})，放弃")
            raise
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
        try:
            from src.pipeline import DEFAULT_PIPELINE
            if DEFAULT_PIPELINE is not None:
                ori_data = DEFAULT_PIPELINE.get_ori_data(name, cq=1)
            else:
                raise Exception()
        except Exception:
            ori_data = pd.read_csv("{}{}".format(syspath, data_cq_file_name))  
        data = []
        if seq_len > 0:
            ori_data = ori_data[0:seq_len]
        for i in range(len(ori_data)):
            item = dict()
            item[u"期数"] = ori_data.iloc[i, 1]
            for j in range(20):
                item[u"红球_{}".format(j+1)] = ori_data.iloc[i, j+2]
            data.append(item)
        return pd.DataFrame(data)
    else:
        spider(name, start, end, mode)

def spider(name="ssq", start=1, end=999999, mode="train", seq_len=0):
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
        try:
            r = get_http_session_with_backoff(url=url, retries=3, backoff_base=0.5, timeout=20)
            try:
                from src.config import HTTP_CACHE_DIR
                import hashlib
                if HTTP_CACHE_DIR:
                    os.makedirs(HTTP_CACHE_DIR, exist_ok=True)
                    key = hashlib.sha256(url.encode('utf-8')).hexdigest()
                    cache_path = os.path.join(HTTP_CACHE_DIR, f"{key}.cache")
                    with open(cache_path, 'w', encoding='utf-8') as fh:
                        fh.write(r.text)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"请求 {url} 失败({e})，放弃")
            raise
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
        try:
            from src.pipeline import DEFAULT_PIPELINE
            if DEFAULT_PIPELINE is not None:
                ori_data = DEFAULT_PIPELINE.get_ori_data(name, cq=0)
            else:
                raise Exception()
        except Exception:
            ori_data = pd.read_csv("{}{}".format(syspath, data_file_name))  
        data = []
        if seq_len > 0:
            ori_data = ori_data[0:seq_len]
        for i in range(len(ori_data)):
            item = dict()
            if (ori_data.iloc[i, 1] < int(start) or ori_data.iloc[i, 1] > int(end)) and seq_len == 0:
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

# NOTE: setMiniargs has been removed. Use src.pipeline.DEFAULT_PIPELINE.set_args(args) instead.

def init():
    global pred_key_d, filedata, filetitle
    filedata = []
    filetitle = []
    pred_key_d = {}

def predict_ball_model(name, dataset, num_classes, sub_name="红球", window_size=1, hidden_size=128, num_layers=8, num_heads=16, input_size=20, output_size=20, model_name="Transformer", args=None, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), embedding_dim=50):
    """预测指定球种的下一期结果

    This function prefers an explicit `args` object; if `args` is None it will attempt
    to delegate to a LotteryPipeline instance that has args set. Otherwise it will use
    the global `mini_args` for backward compatibility.
    """
    # Prefer explicit args, otherwise prefer pipeline singleton args. Do not fall back to legacy global.
    if args is None:
        try:
            from src.pipeline import DEFAULT_PIPELINE
            if DEFAULT_PIPELINE is not None and DEFAULT_PIPELINE.args is not None:
                cur_args = DEFAULT_PIPELINE.args
            else:
                raise ValueError("No args provided: call src.pipeline.DEFAULT_PIPELINE.set_args(args) or pass args explicitly")
        except Exception:
            raise
    else:
        cur_args = args

    # If no explicit args provided and a pipeline exists with args, delegate to it
    if args is None:
        try:
            from src.pipeline import LotteryPipeline

            pipeline = LotteryPipeline()
            if pipeline.args is not None:
                return pipeline.predict_ball_model(name=name, dataset=dataset, num_classes=num_classes, sub_name=sub_name, window_size=window_size, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, input_size=input_size, output_size=output_size, model_name=model_name, device=device, embedding_dim=embedding_dim)
        except Exception:
            # If delegation isn't possible, continue with cur_args (which may be empty)
            pass

    global last_save_time
    sub_name_eng = "red" if sub_name == "红球" else "blue"
    ball_model_name = red_ball_model_name if sub_name == "红球" else blue_ball_model_name
    m_args = model_args[name]
    ball_index = 0 if sub_name == "红球" else 1
    name_list = [(ball_name[ball_index], i + 1) for i in range(num_classes)]

    # safe-guard: ensure model path exists
    syspath = model_path + model_args[cur_args.name]["pathname"]['name'] + str(window_size) + model_args[cur_args.name]["subpath"][sub_name_eng]
    if not os.path.exists(syspath):
        os.makedirs(syspath)
    logger.info("标签数据维度: {}".format(dataset.data.shape))

    dataset = [dataset[0]]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if model_name == "Transformer":
        _model = modeling.Transformer_Model
    elif model_name == "LSTM":
        _model = modeling.LSTM_Model
    else:
        raise ValueError("暂不支持的模型类型: {}".format(model_name))

    checkpoint_path = f"{syspath}{sub_name_eng}_ball_model_pytorch_{model_name}.ckpt"
    input_dim = input_size

    def build_model():
        # use cur_args (which may be the injected args or the global mini_args)
        use_args = cur_args
        if model_name == "Transformer":
            return _model(input_size=input_dim,
                          output_size=output_size,
                          hidden_size=use_args.hidden_size,
                          num_layers=use_args.num_layers,
                          num_heads=use_args.num_heads,
                          dropout=0.5,
                          num_embeddings=m_args["model_args"]["{}_n_class".format(sub_name_eng)],
                          embedding_dim=embedding_dim,
                          seq_len=int(use_args.seq_len)).to(device)
        return _model(input_size=input_dim,
                      output_size=output_size,
                      hidden_size=use_args.hidden_size,
                      num_layers=use_args.num_layers,
                      num_heads=use_args.num_heads,
                      dropout=0.5,
                      num_embeddings=m_args["model_args"]["{}_n_class".format(sub_name_eng)],
                      embedding_dim=embedding_dim,
                      seq_len=int(use_args.seq_len)).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # restore extra_classes if saved in checkpoint
        if 'extra_classes' in checkpoint:
            try:
                modeling.extra_classes = int(checkpoint['extra_classes'])
                logger.info(f"Restored extra_classes={modeling.extra_classes} from checkpoint")
            except Exception:
                pass
        if cur_args is not None and {'seq_len', 'hidden_size', 'num_layers', 'num_heads'}.issubset(checkpoint.keys()):
            if checkpoint['seq_len'] != cur_args.seq_len or checkpoint['hidden_size'] != cur_args.hidden_size or checkpoint['num_layers'] != cur_args.num_layers or checkpoint['num_heads'] != cur_args.num_heads:
                logger.info("当前为预测模式，已自动使用模型保存时的参数")
                # modify cur_args in-place if possible
                try:
                    cur_args.seq_len = checkpoint['seq_len']
                    cur_args.hidden_size = checkpoint['hidden_size']
                    cur_args.num_layers = checkpoint['num_layers']
                    cur_args.num_heads = checkpoint['num_heads']
                except Exception:
                    pass
        model = build_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("已加载{}模型".format(sub_name))
    else:
        model = build_model()

    model.eval()
    y_pred_cpu = None
    y_target_cpu = None
    for batch in dataloader:
        x, y = batch
        x = x.float().to(device)
        y_pred = model(x)
        y_pred_cpu = y_pred.detach().cpu()
        y_target_cpu = y.detach().cpu()
    return y_pred_cpu, name_list, y_target_cpu

def run_predict(window_size, hidden_size=128, num_layers=8, num_heads=16, f_data=0, model="Transformer", args=None, test_mode=0):
    global pred_key_d
    # prefer explicit args, otherwise prefer pipeline singleton args. Do not fall back to legacy global.
    if args is None:
        from src.pipeline import DEFAULT_PIPELINE
        if DEFAULT_PIPELINE is not None and DEFAULT_PIPELINE.args is not None:
            cur_args = DEFAULT_PIPELINE.args
        else:
            raise ValueError("No args provided: call src.pipeline.DEFAULT_PIPELINE.set_args(args) or pass args explicitly")
    else:
        cur_args = args
    balls = ['red', 'blue'] if cur_args.name not in ["pls", "kl8"] else ['red']
    for sub_name_eng in balls:
        sub_name = "红球" if sub_name_eng == "red" else "蓝球"
        slot_count = model_args[cur_args.name]["model_args"]["{}_sequence_len".format(sub_name_eng)]
        num_classes = model_args[cur_args.name]["model_args"]["{}_n_class".format(sub_name_eng)]
        if window_size != 0:
            model_args[cur_args.name]["model_args"]["seq_len"] = window_size
        syspath = model_path + model_args[cur_args.name]["pathname"]['name'] + str(cur_args.seq_len) + model_args[cur_args.name]["subpath"][sub_name_eng]
        if os.path.exists("{}{}_ball_model_pytorch_{}.ckpt".format(syspath, sub_name_eng, model)):
            current_number = get_current_number(cur_args.name)
            logger.info("【{}】最新期号: {}".format(name_path[cur_args.name]["name"], current_number))
            logger.info("正在创建【{}】数据集...".format(name_path[cur_args.name]["name"]))
            data = create_train_data(cur_args.name, model_args[cur_args.name]["model_args"]["seq_len"], 1, sub_name_eng, cur_args.cq, f_data=f_data, model=model, test_flag=2)

            input_dim = slot_count + modeling.extra_classes
            output_dim = num_classes

            y_pred, name_list, y_target = predict_ball_model(name=cur_args.name,
                                                             dataset=data,
                                                             num_classes=output_dim,
                                                             sub_name=sub_name,
                                                             window_size=window_size,
                                                             hidden_size=hidden_size,
                                                             num_layers=num_layers,
                                                             num_heads=num_heads,
                                                             input_size=input_dim,
                                                             output_size=output_dim,
                                                             model_name=model,
                                                             args=cur_args,
                                                             embedding_dim=50)

            if y_pred is None:
                logger.warning("未获得{}的预测结果".format(sub_name))
                continue

            probabilities = torch.sigmoid(y_pred.squeeze(0))
            decoded = modeling.binary_decode_array(probabilities, threshold=0.25, top_k=output_dim)
            predicted_candidates = decoded[0] if len(decoded) > 0 else []
            unique_candidates = list(dict.fromkeys(predicted_candidates))
            top_predictions = unique_candidates[:slot_count]

            if test_mode == 0 or f_data == 0:
                logger.info("{}候选号码: {}".format(sub_name, predicted_candidates))
            else:
                logger.info("{}候选号码: {}".format(sub_name, top_predictions))

            result_strings = []
            result_strings.append("------------Predict Datetime: {}------------".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            result_strings.append("候选号码(阈值0.25): {}".format(predicted_candidates))
            result_strings.append("前{}位候选号码: {}".format(slot_count, top_predictions))
            result_strings.append("前{}位候选排序: {}".format(slot_count, sorted(top_predictions)))

            correct_nums = 0
            total_nums = 0
            if y_target is not None:
                target_array = y_target[:, :, :slot_count].reshape(-1).tolist()
                target_numbers = [int(value) for value in target_array if 1 <= int(value) <= num_classes]
                target_unique = list(dict.fromkeys(target_numbers))
                y_target_set = set(target_unique)
                total_nums += len(target_unique)
                correct_nums += len(set(top_predictions) & y_target_set)

            if test_mode != 0 and f_data > 0:
                accuracy = correct_nums / (total_nums if total_nums > 0 else 1)
                logger.info("预测{}准确率: {:.2f}%".format(sub_name, accuracy * 100))
                result_strings.append("预测{}准确率: {:.2f}%".format(sub_name, accuracy * 100))
            result_strings.append("------------------------------------------------")
            write_strings_to_file(result_path, result_strings)
        else:
            logger.warning("抱歉，没有找到{}模型".format(sub_name))
            exit(0)

def get_year():
    """ 截取年份
    eg：2020-->20, 2021-->21
    :return:
    """
    return int(str(datetime.datetime.now().year)[-2:])


def try_error(name, predict_features, seq_len, args=None):
    """Handle missing/irregular predict_features by fetching more history.

    Prefer an explicit `args` (or pipeline.args via delegation). Falls back to module-level
    `mini_args` for backward compatibility.
    """
    # prefer explicit args, otherwise prefer pipeline singleton args. Do not fall back to legacy global.
    if args is None:
        from src.pipeline import DEFAULT_PIPELINE
        if DEFAULT_PIPELINE is not None and DEFAULT_PIPELINE.args is not None:
            cur_args = DEFAULT_PIPELINE.args
        else:
            raise ValueError("No args provided: call src.pipeline.DEFAULT_PIPELINE.set_args(args) or pass args explicitly")
    else:
        cur_args = args
    # if args not provided, try to delegate to pipeline to obtain args
    if args is None:
        try:
            from src.pipeline import LotteryPipeline
            pipeline = LotteryPipeline()
            if pipeline.args is not None:
                cur_args = pipeline.args
        except Exception:
            pass

    if len(predict_features) != seq_len:
        logger.warning("期号出现跳期，期号不连续！开始查找最近上一期期号！本期预测时间较久！")
        last_current_year = (get_year() - 1) * 1000
        max_times = 160
        while len(predict_features) != seq_len:
            if getattr(cur_args, 'cq', 0) == 0:
                predict_features = spider(name, last_current_year + max_times, get_current_number(name), "predict", seq_len)
            else:
                predict_features = spider_cq(name, last_current_year + max_times, get_current_number(name), "predict", seq_len)
            max_times -= 1
        return predict_features
    return predict_features

# def get_final_result(name, mode=0):
#     """" 最终预测函数
#     """
#     m_args = model_args[name]["model_args"]
#     seq_len = model_args[name]["model_args"]["seq_len"]
#     current_number = get_current_number(mini_args.name)
#     logger.info("正在创建【{}】数据集...".format(name_path[name]["name"]))
#     red_data = create_train_data(name, seq_len, 1, "red")
#     blue_data = create_train_data(name, seq_len, 1, "blue")
#     logger.info("【{}】预测期号：{} 窗口大小:{}".format(name_path[name]["name"], int(current_number) + 1, seq_len))
#     if name == "ssq":
#         red_pred, red_name_list = get_red_ball_predict_result(red_data, m_args["sequence_len"], m_args["seq_len"])
#         blue_pred = get_blue_ball_predict_result(name, blue_data, 0, m_args["seq_len"])
#         ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list] + [ball_name[1][mode]]
#         pred_result_list = red_pred[0].tolist() + blue_pred.tolist()
#         return {
#             b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
#         }
#     elif name == "dlt":
#         red_pred, red_name_list = get_red_ball_predict_result(red_data, m_args["red_sequence_len"], m_args["seq_len"])
#         blue_pred, blue_name_list = get_blue_ball_predict_result(name, blue_data, m_args["blue_sequence_len"], m_args["seq_len"])
#         ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list] + ["{}_{}".format(name[mode], i) for name, i in blue_name_list]
#         pred_result_list = red_pred[0].tolist() + blue_pred[0].tolist()
#         return {
#             b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
#         }
#     elif name == "pls":
#         red_pred, red_name_list = get_red_ball_predict_result(red_data, m_args["red_sequence_len"], m_args["seq_len"])
#         ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list]
#         pred_result_list = red_pred[0].tolist()
#         return {
#             b_name: int(res) for b_name, res in zip(ball_name_list, pred_result_list)
#         }
#     elif name == "kl8":
#         red_pred, red_name_list = get_red_ball_predict_result(red_data, m_args["red_sequence_len"], m_args["seq_len"])
#         ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list]
#         pred_result_list = red_pred[0].tolist()
#         return {
#             b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
#         }

# def predict_run(name):
#     global filedata, filetitle
#     seq_len = model_args[name]["model_args"]["seq_len"]
#     diff_number = seq_len - 1
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

def write_strings_to_file(folder, strings):
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Format the current date and time as a string
    datetime_string = current_datetime.strftime('%Y%m%d')

    # Create the file path
    file_path = os.path.join(folder, datetime_string + '.txt')

    # Open the file in append mode and write the strings
    with open(file_path, 'a') as file:
        for string in strings:
            file.write(string + '\n')
        file.write('\n')
    file.close()

# if __name__ == "__main__":
#     spider_cq("kl8", "20180101", "20180110", "train")