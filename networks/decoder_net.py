import torch.nn as nn

class DecoderNet(nn.Module):
    def __init__(self, feature_dim):
        super(DecoderNet, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.decoder(x)