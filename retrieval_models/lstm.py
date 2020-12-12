import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence


class LSTMModel(nn.Module):
    def __init__(self, cfg):
        super(LSTMModel, self).__init__()
        cfg_lstm = cfg.get('lstm', {})
        input_dim = cfg.get('embedding_dim', 100)
        hidden_dim = cfg_lstm.get('hidden_dim', 100)
        output_dim = input_dim
        layer_dim = cfg_lstm.get('num_layers', 1)
        
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        self.batch_size = cfg.get('batch_size', 32)
        
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not cfg.get('use_cuda', True):
            self.device = torch.device('cpu')
    
    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # Initialize hidden state
        # max_length = int(torch.max(x.batch_sizes).data)
        h0 = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim).to(self.device)
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim).to(self.device)
        
        x = x.to(self.device)
        # One time step
        out, (hn, cn) = self.lstm(x, (h0,c0))
        # out, output_lengths = pad_packed_sequence(out, batch_first=True)
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out