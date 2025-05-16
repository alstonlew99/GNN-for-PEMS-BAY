import torch
import torch.nn as nn

class MLPForecast(nn.Module):
    def __init__(self, input_len=12, num_nodes=325, hidden_dim=512):
        super(MLPForecast, self).__init__()
        self.input_size = input_len * num_nodes

        self.model = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        out = self.model(x)
        return out
