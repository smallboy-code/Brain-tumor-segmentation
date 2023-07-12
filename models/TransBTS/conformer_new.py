import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, in_channels):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, 4096, in_channels)) #8x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings

class FCUDown(nn.Module):

    def __init__(self, in_channels, dw_stride):
        super(FCUDown, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        # self.sample_pooling = nn.AvgPool3d(kernel_size=dw_stride, stride=dw_stride)
        self.sample_pooling = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=dw_stride, padding=1)
        self.position_embeddings = LearnedPositionalEncoding(in_channels)
        self.ln = nn.LayerNorm(in_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.position_embeddings(x)
        x = self.ln(x)
        x = self.act(x)


        return x

class FCUUp(nn.Module):

    def __init__(self, in_channels, dw_stride):
        super(FCUUp, self).__init__()
        self.dw_stride = dw_stride
        self.conv_project = nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        self.bn = normalization(in_channels)
        self.act = nn.ReLU()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        self.bn1 = normalization(in_channels)


    def forward(self, x, H, W, Z):
        B, _, C = x.shape
        x_r = x.transpose(1, 2).reshape(B, C, H, W, Z)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        x_r = F.interpolate(x_r, size=(H * self.dw_stride, W * self.dw_stride, Z * self.dw_stride))

        x_r = self.act(self.bn1(self.conv1(x_r)))

        return x_r

class Block(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class ConvBlock(nn.Module):

    def __init__(self, in_channels, dw_stride):
        super(ConvBlock, self).__init__()
        self.dw_stride = dw_stride
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        self.bn1 = normalization(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = normalization(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        self.bn3 = normalization(in_channels)
        self.relu3 = nn.ReLU(inplace=True)

        self.FCUDown = FCUDown(in_channels, dw_stride)
        self.Block = Block(in_channels)
        self.FCUUp = FCUUp(in_channels, dw_stride)

        self.conv4 = nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        self.bn4 = normalization(in_channels)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn5 = normalization(in_channels)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        self.bn6 = normalization(in_channels)
        self.relu6 = nn.ReLU(inplace=True)


    def forward(self, x):
        output = {}

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        output['x2'] = y # [1, 16, 128, 128, 128]
        _, _, H, W, Z = y.shape
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu3(y)
        y = y + x
        output['state1'] = y # [1, 16, 128, 128, 128]

        y = self.FCUDown(output['x2'])
        output['fcudown'] = y # [1, 4096, 16]
        y = self.Block(y)
        output['block'] = y # [1, 4096, 16]
        y = self.FCUUp(y, H // self.dw_stride, W // self.dw_stride, Z // self.dw_stride)
        output['fcuup'] = y # [1, 16, 128, 128, 128]

        y = self.conv4(y)
        y = self.bn4(y)
        y = self.relu4(y)
        y = y + output['fcuup']

        y = self.conv5(y)
        y = self.bn5(y)
        y = self.relu5(y)

        y = self.conv6(y)
        y = self.bn6(y)
        y = self.relu6(y)

        y = y + output['state1']
        output['out'] = y
        return output


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 16, 128, 128, 128), device=cuda0)
        model = ConvBlock(in_channels=16, dw_stride=8)
        model.cuda()
        output = model(x)

        print('output:', output['out'].shape)
