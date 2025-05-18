import torch
import torch.nn as nn
from gcn import GraphConvolution

class TGCN(nn.Module):
    def __init__(self, num_nodes, in_dim=1, gcn_hidden_dim=64, gru_hidden_dim=64, dropout_rate=0.3):
        super(TGCN, self).__init__()
        self.num_nodes = num_nodes
        self.gcn1 = GraphConvolution(in_dim, gcn_hidden_dim)
        self.gcn2 = GraphConvolution(gcn_hidden_dim, gcn_hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.gru_hidden_dim = gru_hidden_dim
        self.gru = nn.GRU(
            input_size=gcn_hidden_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.out = nn.Linear(gru_hidden_dim * 2, 1)

    def forward(self, x, adj):
        batch_size, seq_len, num_nodes = x.shape

        x = x.permute(1, 0, 2)
        gcn_outputs = []

        for t in range(seq_len):
            xt = x[t].unsqueeze(-1)
            h1 = self.gcn1(xt, adj)
            h1 = torch.relu(h1)
            h1 = self.dropout(h1)

            h2 = self.gcn2(h1, adj)
            h = torch.relu(h1 + h2)  #Residual connection

            gcn_outputs.append(h)

        gcn_seq = torch.stack(gcn_outputs, dim=1)
        gcn_seq = gcn_seq.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, -1)

        gru_out, _ = self.gru(gcn_seq)  # [B*N, seq_len, 2H]

        forward_last = gru_out[:, -1, :self.gru_hidden_dim]
        backward_last = gru_out[:, 0, self.gru_hidden_dim:]

        last_step = torch.cat([forward_last, backward_last], dim=1)  # [B*N, 2H]
        out = self.out(last_step)
        out = out.view(batch_size, num_nodes)

        return out
