import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.conv1 = self.make_block(in_channels = 3, out_channels = 8)
        self.shortcut1 = self.make_shortcut(in_channels = 3, out_channels = 8)
        self.pool1 = self.make_small_block()

        self.conv2 = self.make_block(in_channels=8, out_channels=16)
        self.shortcut2 = self.make_shortcut(in_channels=8, out_channels=16)
        self.pool2 = self.make_small_block()

        self.conv3 = self.make_block(in_channels=16, out_channels=32)
        self.shortcut3 = self.make_shortcut(in_channels=16, out_channels=32)
        self.pool3 = self.make_small_block()

        self.conv4 = self.make_block(in_channels=32, out_channels=64)
        self.shortcut4 = self.make_shortcut(in_channels=32, out_channels=64)
        self.pool4 = self.make_small_block()

        self.conv5 = self.make_block(in_channels=64, out_channels=128)
        self.shortcut5 = self.make_shortcut(in_channels=64, out_channels=128)
        self.pool5 = self.make_small_block()

        self.fc1 = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(in_features = 6272, out_features = 512),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes),
        )


    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def make_shortcut(self, in_channels, out_channels):
        if in_channels != out_channels:
            return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding="same")
        else:
            return nn.Identity()

    def make_small_block(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x += self.shortcut1(identity)
        x = self.pool1(x)

        identity = x
        x = self.conv2(x)
        x += self.shortcut2(identity)
        x = self.pool2(x)

        identity = x
        x = self.conv3(x)
        x += self.shortcut3(identity)
        x = self.pool3(x)

        identity = x
        x = self.conv4(x)
        x += self.shortcut4(identity)
        x = self.pool4(x)

        identity = x
        x = self.conv5(x)
        x += self.shortcut5(identity)
        x = self.pool5(x)

        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # or use flatten
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = SimpleCNN()
    fake_input = torch.rand(8, 3, 224, 224)
    result = model(fake_input)
    print(result.shape)