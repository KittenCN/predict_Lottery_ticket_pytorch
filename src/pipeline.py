# -*- coding:utf-8 -*-
"""
Lightweight pipeline wrapper to encapsulate common operations and reduce global state usage.
This is intentionally non-intrusive and uses existing functions in src.common and src.modeling.
"""
from typing import Optional, Dict, Any
import os
import torch
import threading
from src import common as _common

import pandas as pd
from src.config import name_path, data_file_name, data_cq_file_name

# bring into this module names used by older code paths
model_args = _common.model_args
model_path = _common.model_path
ball_name = _common.ball_name
modeling = _common.modeling
from src import modeling
from src.common import init, create_train_data
from src.common import run_predict as common_run_predict, predict_ball_model as common_predict_ball_model


class LotteryPipeline:
    """A small pipeline helper to group common operations.

    Responsibilities (minimal, non-destructive):
    - reset module-level globals via `init()`
    - hold args for convenience via `set_args`
    - create dataset with consistent handling of modeling.extra_classes
    - save/load checkpoint helpers that include extra_classes metadata
    """

    def __init__(self, args: Optional[Any] = None):
        self.args = args
        # ensure globals reset on creation
        init()
        # cache raw dataframes keyed by (name, cq)
        # cache format: key -> { 'df': DataFrame, 'ts': float }
        self._ori_data = {}
        # per-key lock to avoid thundering herd
        import threading
        self._ori_locks = {}
        # default TTL for cached ori_data (seconds) - can be overridden via config
        try:
            from src.config import ORI_DATA_TTL
            self._ori_ttl = int(ORI_DATA_TTL)
        except Exception:
            self._ori_ttl = 300

    def set_args(self, args: Any):
        self.args = args

    def reset(self):
        init()
        # clear cached original data
        self._ori_data.clear()

    def get_ori_data(self, name: str, cq: int = 0, seq_len: int | None = None, refresh: bool = False):
        """Return the original DataFrame for `name` (cached). If not cached, load from disk.

        seq_len: optional, if provided and >0, return only the first seq_len rows.
        refresh: if True, force reloading from disk even if cached and TTL not expired.
        """
        key = (name, int(cq))
        import time
        now = time.time()
        entry = self._ori_data.get(key)
        # create per-key lock lazily
        if key not in self._ori_locks:
            self._ori_locks[key] = threading.Lock()

        with self._ori_locks[key]:
            entry = self._ori_data.get(key)
            expired = True
            if entry is not None and not refresh:
                ts = entry.get('ts', 0)
                expired = (now - ts) > self._ori_ttl
            if entry is None or expired or refresh:
                syspath = name_path[name]["path"]
                if int(cq) == 1 and name == "kl8":
                    df = pd.read_csv(f"{syspath}{data_cq_file_name}")
                else:
                    df = pd.read_csv(f"{syspath}{data_file_name}")
                self._ori_data[key] = {'df': df, 'ts': now}
            entry = self._ori_data[key]
        df = entry['df']
        if seq_len is not None and seq_len > 0:
            return df.head(seq_len)
        return df

    def create_dataset(self, name: str, windows: int, dataset=1, ball_type="red", cq=0, test_flag=0, test_begin=0, f_data=0, model="Transformer", num_classes=80, test_list=[]):
        """Create and return a modeling.MyDataset instance using create_train_data.
        This call will update modeling.extra_classes as a side-effect (same as existing codepath).
        """
        ds = create_train_data(name=name, windows=windows, dataset=dataset, ball_type=ball_type, cq=cq, test_flag=test_flag, test_begin=test_begin, f_data=f_data, model=model, num_classes=num_classes, test_list=test_list)
        return ds

    def save_checkpoint(self, path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, lr_scheduler=None, scaler=None, epoch:int = 0, extra: Dict[str,Any]=None):
        sd = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'extra_classes': modeling.extra_classes,
        }
        if optimizer is not None:
            sd['optimizer_state_dict'] = optimizer.state_dict()
        if lr_scheduler is not None:
            try:
                sd['scheduler_state_dict'] = lr_scheduler.state_dict()
            except Exception:
                pass
        if scaler is not None:
            try:
                sd['scaler_state_dict'] = scaler.state_dict()
            except Exception:
                pass
        if extra:
            sd.update(extra)
        torch.save(sd, path)

    def load_checkpoint(self, path: str, map_location='cpu') -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ck = torch.load(path, map_location=map_location)
        if 'extra_classes' in ck:
            try:
                modeling.extra_classes = int(ck['extra_classes'])
            except Exception:
                pass
        return ck

    def run_predict(self, window_size, hidden_size=128, num_layers=8, num_heads=16, f_data=0, model="Transformer", test_mode=0):
        """Wrapper that calls common.run_predict using the pipeline's args.

        This keeps backward compatibility while centralizing args injection.
        """
        if self.args is None:
            raise ValueError("Pipeline.args is not set. Call set_args(args) first.")
        # delegate to common implementation, passing self.args
        return common_run_predict(window_size=window_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  num_heads=num_heads,
                                  f_data=f_data,
                                  model=model,
                                  args=self.args,
                                  test_mode=test_mode)

    def predict_ball_model(self, name, dataset, num_classes, sub_name="红球", window_size=1, hidden_size=128, num_layers=8, num_heads=16, input_size=20, output_size=20, model_name="Transformer", device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), embedding_dim=50):
        """Delegate wrapper for predict_ball_model that injects pipeline args if available."""
        args = self.args if self.args is not None else None
        # If pipeline has args set, run an internal implementation that avoids global mini_args
        use_args = args
        # replicate the logic from src.common.predict_ball_model but using self.args
        from torch.utils.data import DataLoader
        sub_name_eng = "red" if sub_name == "红球" else "blue"
        m_args = model_args[name]
        ball_index = 0 if sub_name == "红球" else 1
        name_list = [(ball_name[ball_index], i + 1) for i in range(num_classes)]
        if use_args is None:
            raise ValueError("predict_ball_model requires args to be set on the pipeline or passed in")
        syspath = model_path + model_args[use_args.name]["pathname"]['name'] + str(window_size) + model_args[use_args.name]["subpath"][sub_name_eng]
        if not os.path.exists(syspath):
            os.makedirs(syspath)

        dataset = [dataset[0]]
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        if model_name == "Transformer":
            _model = modeling.Transformer_Model
        elif model_name == "LSTM":
            _model = modeling.LSTM_Model
        else:
            raise ValueError(f"暂不支持的模型类型: {model_name}")

        checkpoint_path = f"{syspath}{sub_name_eng}_ball_model_pytorch_{model_name}.ckpt"
        input_dim = input_size

        def build_model():
            use = use_args
            if model_name == "Transformer":
                return _model(input_size=input_dim,
                              output_size=output_size,
                              hidden_size=use.hidden_size,
                              num_layers=use.num_layers,
                              num_heads=use.num_heads,
                              dropout=0.5,
                              num_embeddings=m_args["model_args"]["{}_n_class".format(sub_name_eng)],
                              embedding_dim=embedding_dim,
                              seq_len=int(use.seq_len)).to(device)
            return _model(input_size=input_dim,
                          output_size=output_size,
                          hidden_size=use.hidden_size,
                          num_layers=use.num_layers,
                          num_heads=use.num_heads,
                          dropout=0.5,
                          num_embeddings=m_args["model_args"]["{}_n_class".format(sub_name_eng)],
                          embedding_dim=embedding_dim,
                          seq_len=int(use.seq_len)).to(device)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'extra_classes' in checkpoint:
                try:
                    modeling.extra_classes = int(checkpoint['extra_classes'])
                except Exception:
                    pass
            if use_args is not None and {'seq_len', 'hidden_size', 'num_layers', 'num_heads'}.issubset(checkpoint.keys()):
                if checkpoint['seq_len'] != use_args.seq_len or checkpoint['hidden_size'] != use_args.hidden_size or checkpoint['num_layers'] != use_args.num_layers or checkpoint['num_heads'] != use_args.num_heads:
                    try:
                        use_args.seq_len = checkpoint['seq_len']
                        use_args.hidden_size = checkpoint['hidden_size']
                        use_args.num_layers = checkpoint['num_layers']
                        use_args.num_heads = checkpoint['num_heads']
                    except Exception:
                        pass
            model = build_model()
            model.load_state_dict(checkpoint['model_state_dict'])
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


# Default singleton pipeline for backwards compatible global-style usage.
# Code elsewhere (legacy helpers) may call setMiniargs or expect a global context;
# setting args on DEFAULT_PIPELINE keeps behaviour consistent while centralizing state.
DEFAULT_PIPELINE = LotteryPipeline()
