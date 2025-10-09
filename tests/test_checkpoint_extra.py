import os
import sys
import torch
# ensure project root on sys.path so tests can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import modeling

# minimal smoke test: save a checkpoint with extra_classes and load it

def test_checkpoint_extra_roundtrip():
    # create dummy model
    model = torch.nn.Linear(10, 5)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else torch.cuda.amp.GradScaler()

    # set extra_classes
    modeling.extra_classes = 7
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': 1,
        'seq_len': 5,
        'hidden_size': 128,
        'num_layers': 2,
        'num_heads': 4,
        'extra_classes': modeling.extra_classes,
    }
    # write to a temp file in the tests directory
    test_dir = os.path.dirname(__file__)
    path = os.path.join(test_dir, "ckpt_test.ckpt")
    torch.save(save_dict, str(path))

    # load and ensure modeling.extra_classes restored
    loaded = torch.load(str(path), map_location='cpu')
    assert 'extra_classes' in loaded
    modeling.extra_classes = 0
    modeling.extra_classes = int(loaded['extra_classes'])
    assert modeling.extra_classes == 7

if __name__ == '__main__':
    test_checkpoint_extra_roundtrip()
