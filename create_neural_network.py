import torch
import torch.nn as nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fcl1 = nn.Sequential(
            nn.Linear(in_features = 3*32*32, out_features = 256),
            nn.ReLU()
        )
        self.fcl2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU()
        )
        self.fcl3 = nn.Sequential(
            nn.Linear(in_features = 512, out_features = 1024),
            nn.ReLU()
        )
        self.fcl4 = nn.Sequential(
            nn.Linear(in_features = 1024, out_features = 512),
            nn.ReLU()
        )
        self.fcl5 = nn.Sequential(
            nn.Linear(in_features = 512, out_features = num_classes),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fcl1(x)
        x = self.fcl2(x)
        x = self.fcl3(x)
        x = self.fcl4(x)
        x = self.fcl5(x)
        return x

if __name__ == "__main__":
    fake_input = torch.rand(8, 3, 32, 32)
    model = SimpleNeuralNetwork()
    result = model(fake_input)
    print(result)