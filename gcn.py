import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Simple GCN layer as used in Kipf & Welling (ICLR 2017).
    H' = ReLU(ÂHW)
    """
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        """
        x: Tensor, shape [batch_size, num_nodes, input_dim]
        adj: Tensor, shape [num_nodes, num_nodes]
        Returns:
            output: shape [batch_size, num_nodes, output_dim]
        """
        support = self.linear(x)  # shape: [batch, num_nodes, output_dim]
        output = torch.einsum('ij,bjd->bid', adj, support)  # matrix multiply Â * H
        return F.relu(output)
