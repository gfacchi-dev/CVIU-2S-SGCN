import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import LayerNorm


class Model_2S_SGCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, depth: int):
        super().__init__()
        assert depth >= 1, "Depth must be at least 1."

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth

        self.norm = LayerNorm(self.hidden_channels)

        self.blocks_L = torch.nn.ModuleList()
        self.blocks_G = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()

        self.blocks_L.append(GCNConv(in_channels, hidden_channels, improved=True))
        self.blocks_G.append(GCNConv(in_channels, hidden_channels, improved=True))
        self.linears.append(torch.nn.Linear(hidden_channels * 2, hidden_channels))
        for _ in range(depth):
            self.blocks_L.append(GCNConv(hidden_channels, hidden_channels, improved=True))
            self.blocks_G.append(GCNConv(hidden_channels, hidden_channels, improved=True))
            self.linears.append(torch.nn.Linear(hidden_channels * 2, hidden_channels))

        self.final = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, G_edge_index, L_edge_index):
        for i in range(self.depth):
            x_L = self.blocks_L[i](x, L_edge_index)
            x_L = self.norm(x_L)
            x_L = F.relu(x_L)
            x_G = self.blocks_G[i](x, G_edge_index)
            x_G = self.norm(x_G)
            x_G = F.relu(x_G)
            if i > 0:
                x_L = x_L + x
                x_G = x_G + x
            x = torch.cat([x_G, x_L], dim=1)
            x = self.linears[i](x)
        x_final = self.final(x)

        return x_final, x
