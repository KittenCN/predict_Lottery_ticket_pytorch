import os
import sys
import torch
import tempfile
# ensure project root on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import modeling


def test_smoke_train_roundtrip():
    # configuration for smoke run
    seq_len = 5
    extra = 3
    modeling.extra_classes = extra
    num_classes = 10
    batch_size = 4
    input_dim = seq_len + modeling.extra_classes

    # build a simple Transformer model
    model = modeling.Transformer_Model(input_size=input_dim, output_size=num_classes, hidden_size=64, num_layers=1, num_heads=2, seq_len=seq_len)
    device = torch.device('cpu')
    model.to(device)

    # synthetic dataset: x shape [batch, seq_len, input_features]
    # note: modeling.Transformer_Model expects input shaped accordingly in current implementation
    x = torch.randn(batch_size, seq_len, input_dim, dtype=torch.float32).to(device)
    # labels: multi-hot targets (numbers from 1..num_classes) -> build multi-hot
    # create for each batch a random set of 2 target indices
    targets = torch.zeros(batch_size, num_classes, dtype=torch.float32)
    for i in range(batch_size):
        idx = torch.randperm(num_classes)[:2]
        targets[i, idx] = 1.0

    criterion = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # forward
    model.train()
    optim.zero_grad()
    out = model(x)
    assert out.shape == (batch_size, num_classes), f"unexpected out shape: {out.shape}"
    loss = criterion(out, targets)
    loss.backward()
    optim.step()

    # save checkpoint containing extra_classes
    tmpdir = tempfile.mkdtemp()
    ckpt_path = os.path.join(tmpdir, 'smoke_ckpt.ckpt')
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'epoch': 0,
        'seq_len': seq_len,
        'hidden_size': 64,
        'num_layers': 1,
        'num_heads': 2,
        'extra_classes': modeling.extra_classes,
    }
    torch.save(save_dict, ckpt_path)

    # reset and load: ensure modeling.extra_classes restored
    modeling.extra_classes = 0
    loaded = torch.load(ckpt_path, map_location='cpu')
    assert 'extra_classes' in loaded
    modeling.extra_classes = int(loaded['extra_classes'])
    assert modeling.extra_classes == extra

    # try to build new model using restored extra
    input_dim2 = seq_len + modeling.extra_classes
    model2 = modeling.Transformer_Model(input_size=input_dim2, output_size=num_classes, hidden_size=64, num_layers=1, num_heads=2, seq_len=seq_len)
    model2.load_state_dict(loaded['model_state_dict'], strict=False)

    print('smoke test ok')


if __name__ == '__main__':
    test_smoke_train_roundtrip()
