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
        super(LstmWithCRFModel, self).__init__()
        self.batch_size = batch_size
        self.n_class = n_class
        self.ball_num = ball_num
        self.w_size = w_size
        self.embedding_size = embedding_size
        self.words_size = words_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size

        self.embedding = nn.Embedding(words_size, embedding_size)
        # self.first_lstm = nn.ModuleList([nn.LSTM(embedding_size, hidden_size) for _ in range(ball_num)])
        self.first_lstm = nn.LSTM(embedding_size, hidden_size, layer_size, batch_first=True)
        self.second_lstm = nn.LSTM(hidden_size, hidden_size, layer_size, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(hidden_size * 2, n_class)

    def log_sum_exp_matrix(self, matrix, axis):
        # """ Calculates the log of the sum of the exponentiated elements of a matrix along a given axis.
        # Args:
        #     matrix: Matrix tensor
        #     axis: Axis along which the sum is calculated
        # Returns:
        #     log_sum_exp: Log of the sum of the exponentiated elements
        #     indices: Indices of the maximum element along the given axis
        # """
        max_val, indices = torch.max(matrix, dim=axis)
        max_val_broadcast = max_val.unsqueeze(axis)
        matrix_exp = torch.exp(matrix - max_val_broadcast)
        matrix_exp_sum = torch.sum(matrix_exp, dim=axis)
        log_sum_exp = torch.log(matrix_exp_sum) + max_val

        return log_sum_exp, indices
    
    def viterbi_decode(self, outputs, transition_params):
        # """ Decodes the predicted sequence using the Viterbi algorithm.
        # Args:
        #     outputs: Output tensor of shape (w_size, num_tags)
        #     transition_params: Transition parameters tensor of shape (num_tags, num_tags)
        # Returns:
        #     viterbi_score: Viterbi score scalar
        #     viterbi_sequence: Viterbi sequence tensor of shape (w_size)
        # """
        w_size = outputs.size(0)
        num_tags = outputs.size(1)

        viterbi = torch.empty(w_size, num_tags, dtype=torch.float)
        backpointers = torch.empty(w_size, num_tags, dtype=torch.int)
        viterbi[0] = outputs[0]
        for i in range(1, w_size):
            viterbi[i], backpointers[i] = self.log_sum_exp_matrix(viterbi[i-1].unsqueeze(1), transition_params) + outputs[i]

        viterbi_score, best_tag = torch.max(viterbi[w_size-1], dim=0)
        viterbi_sequence = torch.empty(w_size, dtype=torch.int)
        viterbi_sequence[w_size-1] = best_tag
        for i in range(w_size-2, -1, -1):
            viterbi_sequence[i] = backpointers[i+1, viterbi_sequence[i+1]]

        return viterbi_score, viterbi_sequence
    
    def crf_decode(self, outputs, transition_params, sequence_length):
        # """ Decodes the predicted sequence for the CRF.
        # Args:
        #     outputs: Output tensor of shape (batch_size, w_size, num_tags)
        #     transition_params: Transition parameters tensor of shape (num_tags, num_tags)
        #     sequence_length: Length of the sequences in the input data
        # Returns:
        #     pred_sequence: Predicted sequence tensor of shape (batch_size, w_size)
        #     viterbi_score: Viterbi score tensor of shape (batch_size)
        # """
        batch_size = outputs.size(0)
        w_size = outputs.size(1)
        num_tags = outputs.size(2)

        pred_sequence = torch.empty(batch_size, w_size, dtype=torch.int)
        viterbi_score = torch.empty(batch_size, dtype=torch.float)
        for i in range(batch_size):
            viterbi_score[i], pred_sequence[i, :sequence_length[i]] = self.viterbi_decode(outputs[i, :sequence_length[i]], transition_params)

        return pred_sequence, viterbi_score
    
    def crf_sequence_score(self, outputs, tag_indices, transition_params):
        # """ Calculates the score of the predicted sequence for the CRF.
        # Args:
        #     outputs: Output tensor of shape (w_size, num_tags)
        #     tag_indices: Tag indices tensor of shape (w_size)
        #     transition_params: Transition parameters tensor of shape (num_tags, num_tags)
        # Returns:
        #     score: Score of the predicted sequence scalar
        # """
        w_size = outputs.size(0)
        num_tags = outputs.size(1)

        score = outputs[0, tag_indices[0]]
        for i in range(1, w_size):
            score += outputs[i, tag_indices[i]] + transition_params[tag_indices[i-1], tag_indices[i]]

        return score
    
    def crf_forward(self, outputs, transition_params):
        # """ Calculates the normalization factor for the CRF.
        # Args:
        #     outputs: Output tensor of shape (w_size, num_tags)
        #     transition_params: Transition parameters tensor of shape (num_tags, num_tags)
        # Returns:
        #     log_norm: Normalization factor scalar
        # """
        w_size = outputs.size(0)
        num_tags = outputs.size(1)

        alpha = outputs[0]
        for i in range(1, w_size):
            log_sum_exp = self.log_sum_exp_matrix(alpha.unsqueeze(1), transition_params) + outputs[i]
            alpha = log_sum_exp
        log_norm = torch.logsumexp(alpha, dim=0)

        return log_norm
    
    def crf_log_likelihood_single(self, outputs, tag_indices, transition_params):
        # """ Calculates the negative log likelihood of the CRF for a single example.
        # Args:
        #     outputs: Output tensor of shape (w_size, num_tags)
        #     tag_indices: Tag indices tensor of shape (w_size)
        #     transition_params: Transition parameters tensor of shape (num_tags, num_tags)
        # Returns:
        #     log_likelihood: Negative log likelihood scalar
        # """
        log_norm = self.crf_forward(outputs, transition_params)
        log_likelihood = -self.crf_sequence_score(outputs, tag_indices, transition_params) + log_norm
        return log_likelihood
    
    def crf_log_likelihood(self, outputs, tag_indices, sequence_length, batch_size, num_tags):
        # """ Calculates the negative log likelihood of the CRF.
        # Args:
        #     outputs: Output tensor of shape (batch_size, w_size, num_tags)
        #     tag_indices: Tag indices tensor of shape (batch_size, w_size)
        #     sequence_length: Length of the sequences in the input data
        #     batch_size: Batch size
        #     num_tags: Number of tags
        # Returns:
        #     log_likelihood: Negative log likelihood tensor of shape (batch_size)
        #     transition_params: Transition parameters tensor of shape (num_tags, num_tags)
        # """
        # Obtain transition parameters
        transition_params = torch.empty(num_tags, num_tags, dtype=torch.float)
        nn.init.uniform_(transition_params)

        # Calculate log likelihood
        log_likelihood = torch.empty(batch_size, dtype=torch.float)
        for i in range(batch_size):
            log_likelihood[i] = self.crf_log_likelihood_single(outputs[i, :sequence_length[i]], tag_indices[i, :sequence_length[i]], transition_params)

        return log_likelihood, transition_params

    def forward(self, inputs, tag_indices, sequence_length):
        # Extract features
        embedding = self.embedding(inputs)
        first_lstm = torch.stack([self.first_lstm(embedding[:, :, i, :])[0] for i in range(self.ball_num)], dim=0)
        second_lstm, _ = self.second_lstm(first_lstm)

        # Obtain output
        outputs = self.dense(second_lstm)

        # Calculate loss
        log_likelihood, transition_params = self.crf_log_likelihood(outputs, tag_indices, sequence_length, self.batch_size, self.n_class)
        loss = -torch.sum(log_likelihood)

        # Decode predictions
        pred_sequence, viterbi_score = self.crf_decode(outputs, transition_params, sequence_length)

        return outputs, loss, pred_sequence, viterbi_score

