import torch
import torch.nn as nn
from torch.nn import Module


class UNet(Module):
    def __init__(self, class_num):
        super(UNet, self).__init__()

        self.down_stage1 = self._down_layer(3, 64, down_sample=False)
        self.down_stage2 = self._down_layer(64, 128, down_sample=True)
        self.down_stage3 = self._down_layer(128, 256, down_sample=True)
        self.down_stage4 = self._down_layer(256, 512, down_sample=True)
        self.down_stage5 = self._down_layer(512, 1024, down_sample=True)

        self.up_sample1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_stage1 = self._conv_layer(1024, 512)
        self.up_sample2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_stage2 = self._conv_layer(512, 256)
        self.up_sample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_stage3 = self._conv_layer(256, 128)
        self.up_sample4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_stage4 = self._conv_layer(128, 64)

        self.out_layer = nn.Sequential(
            nn.Conv2d(64, class_num, kernel_size=1),
            nn.BatchNorm2d(class_num),
            nn.ReLU(),
        )

    def forward(self, x):
        down_feature1 = self.down_stage1(x)
        down_feature2 = self.down_stage2(down_feature1)
        down_feature3 = self.down_stage3(down_feature2)
        down_feature4 = self.down_stage4(down_feature3)
        down_feature5 = self.down_stage5(down_feature4)

        out = self.up_stage1(
            torch.cat([down_feature4, self.up_sample1(down_feature5)], dim=1)
        )
        out = self.up_stage2(torch.cat([down_feature3, self.up_sample2(out)], dim=1))
        out = self.up_stage3(torch.cat([down_feature2, self.up_sample3(out)], dim=1))
        out = self.up_stage4(torch.cat([down_feature1, self.up_sample4(out)], dim=1))
        out = self.out_layer(out)

        return out

    def _conv_layer(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        return layer

    def _down_layer(self, in_channels, out_channels, down_sample=False):
        layer = []

        if down_sample:
            layer.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layer.append(
            self._conv_layer(in_channels=in_channels, out_channels=out_channels)
        )

        return nn.Sequential(*layer)


# test code
if __name__ == "__main__":
    model = UNet(21).cuda()

    img = torch.randn(8, 3, 320, 320)
    x = model(img.cuda())
    print(x.shape)
    print(x.dtype)
    _, pred_idx = torch.max(x, dim=1)
    print(pred_idx.shape)

    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
