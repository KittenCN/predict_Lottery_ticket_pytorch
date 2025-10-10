import os
import sys
import torch
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import LotteryPipeline
from src import modeling


def test_load_checkpoint_without_extra(tmp_path):
    pipe = LotteryPipeline()
    # make sure modeling.extra_classes has a known value
    modeling.extra_classes = 0
    # create a checkpoint without extra_classes
    model = torch.nn.Linear(4, 2)
    ck_path = tmp_path / 'no_extra.ckpt'
    torch.save({'model_state_dict': model.state_dict(), 'epoch': 1}, str(ck_path))
    loaded = pipe.load_checkpoint(str(ck_path))
    # extra_classes not present, modeling.extra_classes should remain unchanged
    assert modeling.extra_classes == 0


def test_save_checkpoint_includes_extra(tmp_path):
    pipe = LotteryPipeline()
    modeling.extra_classes = 7
    model = torch.nn.Linear(6, 3)
    ck_path = tmp_path / 'has_extra.ckpt'
    pipe.save_checkpoint(str(ck_path), model, epoch=3)
    loaded = pipe.load_checkpoint(str(ck_path))
    assert 'extra_classes' in loaded
    assert int(loaded['extra_classes']) == 7
