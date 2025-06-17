import torch.nn as nn
import torch.nn.init as init
import torch

def double_conv(in_channels, out_channels,affine):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels,affine=affine),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels,affine=affine),
        nn.ReLU(inplace=True))



def conv_bn_relu(in_channels, out_channels, kernel_size,affine=False):
    layer = []
    layer.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False))
    layer.append(nn.BatchNorm2d(out_channels,affine=affine))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)


def conv_bn_relu_transpose(in_channels, out_channels, kernel_size,affine=False):
    layer = []
    layer.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, bias=False))
    layer.append(nn.BatchNorm2d(out_channels,affine=affine))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)

class FCRN(nn.Module):
    def __init__(self,in_channels=1, out_channels=32, kernel_size=3,sigmoid=False,affine=False):

        super(FCRN, self).__init__()
        self.add_sigmoid = sigmoid
        # Encoder
        self.conv1 = conv_bn_relu(in_channels, out_channels, kernel_size,affine=affine)
        self.conv2 = conv_bn_relu(out_channels, out_channels * 2, kernel_size,affine=affine)
        self.conv3 = conv_bn_relu(out_channels * 2, out_channels * 4, kernel_size,affine=affine)

        self.maxpool = nn.MaxPool2d(2, 2)

        # LatentSpace
        self.conv4 = conv_bn_relu(out_channels * 4, out_channels * 16, kernel_size,affine=affine)

        # Decoder
        self.conv5 = conv_bn_relu_transpose(out_channels * 16, out_channels * 4, 2,affine=affine)
        self.conv6 = conv_bn_relu_transpose(out_channels * 4, out_channels * 2, 2,affine=affine)
        self.conv7 = conv_bn_relu_transpose(out_channels * 2, out_channels, 2,affine=affine)
        self.conv8 = nn.Conv2d(out_channels, in_channels, 3, padding=1)
        if self.add_sigmoid:
            self.sigmoid = nn.Sigmoid()

        self._initialize_weights()
    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.maxpool(self.conv3(x))

        x = self.conv4(x)
        feature_dist = x

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        if self.add_sigmoid:
            out = self.sigmoid(x)
        else:
            out = x

        return out,feature_dist

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m,nn.ConvTranspose2d):
                init.normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    init.constant_(m.weight, 0.1)
                    init.constant_(m.bias, 0)


class UNet(nn.Module):

    def __init__(self, n_class,sigmoid=False,affine=False):
        super(UNet,self).__init__()

        #Encoder
        self.dconv_down1 = double_conv(1, 32,affine=affine)
        self.dconv_down2 = double_conv(32, 64, affine=affine)
        self.dconv_down3 = double_conv(64, 128,affine=affine)


        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #Decoder
        self.dconv_up2 = double_conv(64 + 128, 64,affine=affine)
        self.dconv_up1 = double_conv(32 + 64, 32, affine=affine)
        self.conv_last = nn.Conv2d(32, n_class, kernel_size=1)
        self.add_sigmoid = sigmoid
        if self.add_sigmoid:
            self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    init.constant_(m.weight, 0.1)
                    init.constant_(m.bias, 0)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        x = self.dconv_down3(x)

        feature_distill =x
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        if self.add_sigmoid:
            out = nn.Sigmoid()(x)
        else:
            out = x
        return out,feature_distill


