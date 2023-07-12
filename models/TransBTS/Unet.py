import torch.nn as nn
import torch.nn.functional as F
import torch

# adapt from https://github.com/MIC-DKFZ/BraTS2017


def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m



class InitConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=16, dropout=0.2):
        super(InitConv, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout3d(y, self.dropout)

        return y


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='gn'):
        super(EnBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y

class DnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm='gn'):
        super(DnBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.bn2 = normalization(out_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)

        return y


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y

class DeUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y

class SE(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)

        return x * y.expand_as(x)



class Inter(nn.Module):
    def __init__(self, scale, in_channels, out_channels):
        super(Inter, self).__init__()
        self.scale = scale
        self.se = SE(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.se(x)
        y = F.interpolate(y, scale_factor=self.scale)
        y = self.conv(y)
        y = self.Softmax(y)

        return y

class Unet(nn.Module):
    def __init__(self, in_channels=4, base_channels=16, num_classes=4):
        super(Unet, self).__init__()

        self.InitConv = InitConv(in_channels=in_channels, out_channels=base_channels, dropout=0.2)
        self.EnBlock1 = EnBlock(in_channels=base_channels)
        self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels*2)

        self.EnBlock2_1 = EnBlock(in_channels=base_channels*2)
        self.EnBlock2_2 = EnBlock(in_channels=base_channels*2)
        self.EnDown2 = EnDown(in_channels=base_channels*2, out_channels=base_channels*4)

        self.EnBlock3_1 = EnBlock(in_channels=base_channels * 4)
        self.EnBlock3_2 = EnBlock(in_channels=base_channels * 4)
        self.EnDown3 = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)

        self.EnBlock4_1 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_2 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_3 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_4 = EnBlock(in_channels=base_channels * 8)

        self.DeUp1 = DeUp(in_channels=base_channels * 8, out_channels=base_channels * 4)
        self.DnBlock1_1 = DnBlock(in_channels=base_channels * 8, out_channels=base_channels * 4)
        self.DnBlock1_2 = DnBlock(in_channels=base_channels * 4, out_channels=base_channels * 4)
        self.inter1 = Inter(4, 64, 4)

        self.DeUp2 = DeUp(in_channels=base_channels * 4, out_channels=base_channels * 2)
        self.DnBlock2_1 = DnBlock(in_channels=base_channels * 4, out_channels=base_channels * 2)
        self.DnBlock2_2 = DnBlock(in_channels=base_channels * 2, out_channels=base_channels * 2)
        self.inter2 = Inter(2, 32, 4)


        self.DeUp3 = DeUp(in_channels=base_channels * 2, out_channels=base_channels)
        self.DnBlock3_1 = DnBlock(in_channels=base_channels * 2, out_channels=base_channels)
        self.DnBlock3_2 = DnBlock(in_channels=base_channels, out_channels=base_channels)

        self.ouput = nn.Conv3d(in_channels=base_channels, out_channels=num_classes, kernel_size=1, stride=1)
        self.Softmax = nn.Softmax(dim=1)




    def forward(self, x):
        x = self.InitConv(x)       # (1, 16, 128, 128, 128)

        x1_1 = self.EnBlock1(x)
        x1_2 = self.EnDown1(x1_1)  # (1, 32, 64, 64, 64)

        x2_1 = self.EnBlock2_1(x1_2)
        x2_1 = self.EnBlock2_2(x2_1)
        x2_2 = self.EnDown2(x2_1)  # (1, 64, 32, 32, 32)

        x3_1 = self.EnBlock3_1(x2_2)
        x3_1 = self.EnBlock3_2(x3_1)
        x3_2 = self.EnDown3(x3_1)  # (1, 128, 16, 16, 16)

        x4_1 = self.EnBlock4_1(x3_2)
        x4_2 = self.EnBlock4_2(x4_1)
        x4_3 = self.EnBlock4_3(x4_2)
        x4_4 = self.EnBlock4_4(x4_3)  # (1, 128, 16, 16, 16)

        x5_1 = self.DeUp1(x4_4)  # (1, 64, 32, 32, 32)
        x5_1 = torch.cat([x3_1, x5_1], 1)  # (1, 128, 32, 32, 32)
        x5_1 = self.DnBlock1_1(x5_1)
        x5_1 = self.DnBlock1_2(x5_1)  # (1, 64, 32, 32, 32)
        inter1 = self.inter1(x5_1)

        x6_1 = self.DeUp2(x5_1)  # (1, 32, 64, 64, 64)
        x6_1 = torch.cat([x2_1, x6_1], 1)  # (1, 64, 64, 64, 64)
        x6_1 = self.DnBlock2_1(x6_1)
        x6_1 = self.DnBlock2_2(x6_1)  # (1, 32, 64, 64, 64)
        inter2 = self.inter2(x6_1)

        x7_1 = self.DeUp3(x6_1)  # (1, 16, 128, 128, 128)
        x7_1 = torch.cat([x1_1, x7_1], 1)  # (1, 32, 128, 128, 128)
        x7_1 = self.DnBlock3_1(x7_1)
        x7_1 = self.DnBlock3_2(x7_1)  # (1, 16, 128, 128, 128)

        output = self.ouput(x7_1)
        output = self.Softmax(output)


        return x1_1,x2_1,x3_1,output
        # return output, inter1, inter2


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 4, 128, 128, 128), device=cuda0)
        # model = Unet1(in_channels=4, base_channels=16, num_classes=4)
        model = Unet(in_channels=4, base_channels=16, num_classes=4)
        model.cuda()
        _, _, _, output = model(x)
        print('output:', output.shape)
