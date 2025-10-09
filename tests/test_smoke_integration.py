import os
import tempfile
import torch
from src.pipeline import LotteryPipeline
from src import modeling


def make_dummy_dataset(num_samples=8, seq_len=5, num_classes=10):
    # construct data shaped like modeling.MyDataset would provide: small numpy array
    import numpy as np
    # columns:期数 + red features (num_classes)
    rows = []
    for i in range(num_samples):
        row = [i+1]
        # pad with sequential small ints for features
        for j in range(num_classes):
            row.append(((i + j) % num_classes) + 1)
        rows.append(row)
    data = np.array(rows)
    # modeling.MyDataset expects full-style values; but we'll bypass and create a dataset wrapper
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, data, seq_len, num_classes):
            self.data = data
            self.seq_len = seq_len
            self.num_classes = num_classes

        def __len__(self):
            return len(self.data) - self.seq_len

        def __getitem__(self, idx):
            # x: [seq_len, num_features], y: [1, num_classes]
            x = torch.tensor(self.data[idx:idx + self.seq_len, 1:1 + self.num_classes], dtype=torch.float32)
            y = torch.tensor(self.data[idx + self.seq_len, 1:1 + self.num_classes], dtype=torch.float32).unsqueeze(0)
            return x, y

    return DummyDataset(data, seq_len, num_classes)


def test_smoke_integration_roundtrip(tmp_path):
    pipeline = LotteryPipeline()
    # create small dummy dataset
    ds = make_dummy_dataset(num_samples=12, seq_len=3, num_classes=8)

    # simple model: take flattened input and output num_classes
    input_dim = 3 + modeling.extra_classes
    output_dim = 8
    model = modeling.Transformer_Model(input_size=input_dim, output_size=output_dim, hidden_size=32, num_layers=1, num_heads=2, dropout=0.1, num_embeddings=output_dim, embedding_dim=8, seq_len=3)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    dataloader = torch.utils.data.DataLoader(ds, batch_size=2)

    model.train()
    for x, y in dataloader:
        x = x.float()
        # ensure shape matches model expectation
        preds = model(x)
        loss = criterion(preds, y.squeeze(1))
        loss.backward()
        optimizer.step()
        break

    # save checkpoint
    ck_path = tmp_path / "smoke.ckpt"
    pipeline.save_checkpoint(str(ck_path), model, optimizer=optimizer, epoch=1)

    # load checkpoint and verify extra_classes present and model_state exists
    ck = pipeline.load_checkpoint(str(ck_path), map_location='cpu')
    assert 'model_state_dict' in ck
    assert 'extra_classes' in ck
