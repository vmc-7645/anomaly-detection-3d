import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class TeacherNet(nn.Module):
    def __init__(self, feature_dim):
        super(TeacherNet, self).__init__()
        self.mlp1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU())
        self.lfa1 = gnn.DynamicEdgeConv(nn.Sequential(nn.Linear(2 * 64, 128), nn.ReLU()), k=20)
        self.lfa2 = gnn.DynamicEdgeConv(nn.Sequential(nn.Linear(2 * 128, 256), nn.ReLU()), k=20)
        self.mlp2 = nn.Sequential(nn.Linear(256, feature_dim))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x = self.mlp1(pos)
        x = self.lfa1(x, batch)
        x = self.lfa2(x, batch)
        x = self.mlp2(x)
        return x
