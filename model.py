import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        # batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))  # output(5, 30, 64)
        pred = self.linear(output)  # (5, 30, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred

class simpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(simpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # 初始化hidden和memory cell参数
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward propagate lstm
        out, (h_n, h_c) = self.lstm(x, (h0, c0))

        # 选取最后一个时刻的输出
        out = self.fc(out[:, -1, :])
        return out

class LstmWithCRFModel(nn.Module):
    def __init__(self, batch_size, n_class, ball_num, w_size, embedding_size, words_size, hidden_size, layer_size):
        super().__init__()

        self.batch_size = batch_size
        self.n_class = n_class
        self.ball_num = ball_num
        self.w_size = w_size
        self.embedding_size = embedding_size
        self.words_size = words_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size

        self.embedding = nn.Embedding(words_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=layer_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, n_class)
        self.crf = CRF(n_class, batch_first=True)

    def forward(self, inputs, tag_indices, sequence_length):
        # Extract features using the LSTM network
        x = self.embedding(inputs)
        x, _ = self.lstm(x)

        # Compute the CRF loss
        log_likelihood = self.crf(x, tag_indices, mask=sequence_length, reduction="mean")
        loss = -log_likelihood

        # Decode the CRF layer to get the predicted sequences
        pred_sequence = self.crf.decode(x, mask=sequence_length)

        return loss, pred_sequence