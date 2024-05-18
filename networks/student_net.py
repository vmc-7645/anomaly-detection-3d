import torch.nn as nn

class StudentNet(nn.Module):
    def __init__(self, feature_dim):
        super(StudentNet, self).__init__()
        # Define the layers for the student network
        self.encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, data):
        x = data.pos
        return self.encoder(x)