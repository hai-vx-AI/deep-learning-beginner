import torch
import torch.nn as nn
import timm

class DeepBallMobileOne(nn.Module):
    def __init__(self):
        super(DeepBallMobileOne, self).__init__()
        # 1. KHỞI TẠO BACKBONE MOBILEONE TỪ TIMM
        # - Bản 'mobileone_s0': Tương đương số tham số của MNv3-Small nhưng nhanh hơn.
        # - in_chans=9: Hệ thống timm sẽ TỰ ĐỘNG nhân bản trọng số pretrained 3 lần và chia 3. 
        #   (Triệt tiêu hoàn toàn đoạn code can thiệp thủ công dài dòng cũ).
        # - features_only=True: Tự động cắt bỏ Head phân loại, chỉ giữ lại các feature maps.

        self.backbone = timm.create_model(
            'mobileone_s0',
            pretrained=True,
            in_chans=9,
            features_only=True
        )

        # Lấy tự động số channels của feature map cuối cùng (Với s0, con số này là 1024)
        backbone_out_channels = self.backbone.feature_info[-1]['num_chs']

        # Neck: giải mã không gian 8 -> 16 -> 32 -> 64
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(backbone_out_channels, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        x = features[-1]
        x = self.neck(x)
        x = self.head(x)
        return x  # raw logits, shape: (B, 1, 64, 64)
    
    def switch_to_deploy(self):
        """
        Gộp các nhánh MobileOne trong backbone nếu module hỗ trợ reparameterize().
        Lưu ý: neck và head hiện tại không được fuse bởi hàm này.
        """
        self.eval()

        fused_count = 0

        for module in self.modules():
            if hasattr(module, "reparameterize") and callable(module.reparameterize):
                module.reparameterize()
                fused_count += 1

        print(f"Trạng thái: Đã gọi reparameterize() cho {fused_count} module.")

        if fused_count == 0:
            print("CẢNH BÁO: Không tìm thấy module nào có reparameterize(). Có thể backbone timm không hỗ trợ fuse theo cách này.")
        else:
            print("Model đã được chuyển sang dạng deploy/fused.")