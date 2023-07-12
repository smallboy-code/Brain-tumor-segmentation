import torch.nn as nn
import torch.nn.functional as F
import torch



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
        self.bn = normalization(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        y = F.dropout3d(y, self.dropout)

        return y



class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='gn'):
        super(EnBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        y = self.conv2(x1)
        y = self.bn2(y)
        y = self.relu2(y)
        y = y + x

        return y

class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y

class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y

class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1

class conv_project(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_project, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = normalization(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class Inter(nn.Module):
    def __init__(self, scale, in_channels, out_channels):
        super(Inter, self).__init__()
        self.scale = scale
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.conv(x)
        y = F.interpolate(y, scale_factor=self.scale)
        y = self.Softmax(y)

        return y

class Vnet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Vnet, self).__init__()
        self.num_classes = num_classes

        self.Initcov = InitConv(in_channels=in_channels, out_channels=16)
        self.EnBlock1 = EnBlock(in_channels=16)
        self.EnDown1 = EnDown(in_channels=16, out_channels=32)

        self.EnBlock2_1 = EnBlock(in_channels=32)
        self.EnBlock2_2 = EnBlock(in_channels=32)
        self.EnDown2 = EnDown(in_channels=32, out_channels=64)

        self.EnBlock3_1 = EnBlock(in_channels=64)
        self.EnBlock3_2 = EnBlock(in_channels=64)
        self.EnDown3 = EnDown(in_channels=64, out_channels=128)

        self.EnBlock4_1 = EnBlock(in_channels=128)
        self.EnBlock4_2 = EnBlock(in_channels=128)
        self.EnBlock4_3 = EnBlock(in_channels=128)
        self.EnBlock4_4 = EnBlock(in_channels=128)

        self.DeUp1 = DeUp_Cat(in_channels=128, out_channels=64)
        self.DeBlock1 = DeBlock(in_channels=64)

        self.DeUp2 = DeUp_Cat(in_channels=64, out_channels=32)
        self.DeBlock2 = DeBlock(in_channels=32)

        self.DeUp3 = DeUp_Cat(in_channels=32, out_channels=16)
        self.DeBlock3 = DeBlock(in_channels=16)

        self.endconv = nn.Conv3d(in_channels=16, out_channels=num_classes, kernel_size=1)
        self.Softmax = nn.Softmax(dim=1)


    def forward(self, x):
        output = {}
        x = self.Initcov(x) # [1, 16, 128, 128, 128]
        x = self.EnBlock1(x)
        output['prev1'] = x
        x = self.EnDown1(x)

        x = self.EnBlock2_1(x) # [1, 32, 64, 64, 64]
        x = self.EnBlock2_2(x)
        output['prev2'] = x
        x = self.EnDown2(x)

        x = self.EnBlock3_1(x)  # [1, 64, 32, 32, 32]
        x = self.EnBlock3_2(x)
        output['prev3'] = x
        x = self.EnDown3(x)

        x = self.EnBlock4_1(x)
        x = self.EnBlock4_2(x)
        x = self.EnBlock4_3(x)
        x = self.EnBlock4_4(x)

        x = self.DeUp1(x, output['prev3'])  # [1, 64, 32, 32, 32]
        x = self.DeBlock1(x)

        x = self.DeUp2(x, output['prev2'])  # [1, 32, 64, 64, 64]
        x = self.DeBlock2(x)

        x = self.DeUp3(x, output['prev1'])  # [1, 16, 128, 128, 128]
        x = self.DeBlock3(x)

        x = self.endconv(x)
        x = self.Softmax(x)

        return x


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 4, 128, 128, 128), device=cuda0)
        model = Vnet(in_channels=4, num_classes=4)
        model.cuda()
        output = model(x)
        print('output:', output.shape)
