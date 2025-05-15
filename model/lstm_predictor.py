import torch.nn as nn
import torch

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_output = out[:, -1, :]
        return self.fc(last_output)
