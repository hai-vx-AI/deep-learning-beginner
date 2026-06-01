import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class DeepBall(nn.Module):
    def __init__(self):
        super(DeepBall, self).__init__()
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        # --- CAN THIỆP ĐẦU VÀO: 3 channels -> 9 channels ---
        first_conv = mobilenet.features[0][0]
        new_first_conv = nn.Conv2d(
            in_channels=9,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        with torch.no_grad():
            # Nhân bản trọng số 3 lần, chia 3 để giữ biên độ tín hiệu
            new_first_conv.weight[:] = first_conv.weight.repeat(1, 3, 1, 1) / 3.0
            if first_conv.bias is not None:
                new_first_conv.bias[:] = first_conv.bias
        mobilenet.features[0][0] = new_first_conv

        self.backbone = mobilenet.features  # output: (B, 576, 8, 8) với input 256x256

        # Neck: giải mã không gian 8 -> 16 -> 32 -> 64
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(576, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Head: output là RAW LOGITS (không có Sigmoid ở đây)
        # BUG FIX: Đã xóa nn.Sigmoid(). focal_loss tự gọi torch.sigmoid() bên trong.
        # Nếu để Sigmoid ở đây, loss sẽ nhận xác suất và sigmoid lần 2 -> double sigmoid
        # -> mọi giá trị bị ép về [~0.37, ~0.63], mô hình không thể hội tụ.
        self.head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x  # raw logits, shape: (B, 1, 64, 64)