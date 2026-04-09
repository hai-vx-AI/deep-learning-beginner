import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class DeepBall(nn.Module):
    def __init__(self):
        super(DeepBall, self).__init__()
        mobilenet = mobilenet_v3_small(weights = MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobilenet.features
        self.neck = nn.Sequential(
            # 8 - 16
            nn.ConvTranspose2d(in_channels = 576, out_channels = 256, kernel_size = 4, stride = 2,
                               padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            # 16 - 32
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2,
                                padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            # 32 - 64
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, 
                               padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )

        self.head = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 3, padding = 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x
    
if __name__ == "__main__":
    pass


