import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class StudentNet(nn.Module):
    def __init__(self, feature_dim):
        super(StudentNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, data):
        return self.mlp(data.pos)
