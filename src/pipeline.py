# -*- coding:utf-8 -*-
"""
Lightweight pipeline wrapper to encapsulate common operations and reduce global state usage.
This is intentionally non-intrusive and uses existing functions in src.common and src.modeling.
"""
from typing import Optional, Dict, Any
import os
import torch
from src import modeling
from src.common import init, setMiniargs, create_train_data


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

    def set_args(self, args: Any):
        self.args = args
        setMiniargs(args)

    def reset(self):
        init()

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
