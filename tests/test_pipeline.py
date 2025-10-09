import os
import sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import LotteryPipeline
from src import modeling


def test_pipeline_save_load(tmp_path):
    pipe = LotteryPipeline()
    modeling.extra_classes = 5
    # create a tiny model
    model = torch.nn.Linear(10, 3)
    ck = tmp_path / 'p.ckpt'
    pipe.save_checkpoint(str(ck), model, epoch=2, extra={'note':'test'})
    loaded = pipe.load_checkpoint(str(ck))
    assert 'extra_classes' in loaded
    assert int(loaded['extra_classes']) == 5
    assert loaded['epoch'] == 2


if __name__ == '__main__':
    test_pipeline_save_load(os.getcwd())
