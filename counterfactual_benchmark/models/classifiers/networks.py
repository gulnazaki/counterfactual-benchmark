import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_inputs=1, width=32, num_outputs=1):
        super().__init__()
        activation = nn.LeakyReLU()
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, width, bias=False),
            nn.BatchNorm1d(width),
            activation,
            nn.Linear(width, width, bias=False),
            nn.BatchNorm1d(width),
            activation,
            nn.Linear(width, num_outputs),
        )

    def forward(self, x, y=None):
        return self.mlp(x)


class CNN(nn.Module):
    def __init__(self, in_shape=(1, 192, 192), width=16, num_outputs=1, context_dim=0):
        super().__init__()
        in_channels = in_shape[0]
        res = in_shape[1]
        s = 2 if res > 64 else 1
        activation = nn.LeakyReLU()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, width, 7, s, 3, bias=False),
            nn.BatchNorm2d(width),
            activation,
            (nn.MaxPool2d(2, 2) if res > 32 else nn.Identity()),
            nn.Conv2d(width, 2 * width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(2 * width),
            activation,
            nn.Conv2d(2 * width, 2 * width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2 * width),
            activation,
            nn.Conv2d(2 * width, 4 * width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(4 * width),
            activation,
            nn.Conv2d(4 * width, 4 * width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4 * width),
            activation,
            nn.Conv2d(4 * width, 8 * width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(8 * width),
            activation,
        )
        self.fc = nn.Sequential(
            nn.Linear(8 * width + context_dim, 8 * width, bias=False),
            nn.BatchNorm1d(8 * width),
            activation,
            nn.Linear(8 * width, num_outputs),
        )

    def forward(self, x, y=None):
        x = self.cnn(x)
        x = x.mean(dim=(-2, -1))  # avg pool
        if y is not None:
            x = torch.cat([x, y], dim=-1)
        return self.fc(x)
