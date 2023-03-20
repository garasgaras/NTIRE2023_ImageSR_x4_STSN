import torch
import torch.nn as nn
#import common
#import block6 as B
import torch.nn.functional as F
from torch import Tensor
from ptflops import get_model_complexity_info
from collections import OrderedDict
#from model import attention
from torch.nn.modules.batchnorm import _BatchNorm
import math
class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return nn.ReLU()(input)  # F.gelu(input)
class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out

def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)
def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        #self.norm =nn.BatchNorm2d(dim, eps=1e-5)
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return   x  
class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
      
        self.a = nn.Sequential(
                nn.Conv2d(dim, dim,1),
                nn.GELU(),
                nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim,3,1,1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.norm(x)   
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        return x        
            
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=2., drop_path=0.):
        super().__init__()
        
        self.attn = ConvMod(dim)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6           
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x        

class stsn(nn.Module):
    def __init__(self,  conv=default_conv):
        super(stsn, self).__init__()
        in_nc=3
        nf = 150
        kernel_size = 3
        num_modules=6 
        out_nc=3
        scale = 4
        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)
        # self.patch_emb = PatchEmbeddings(nf, 7, nf)
        self.conv_mixer_layers0 = Block(nf, 2)
        self.conv_mixer_layers1 = Block(nf, 2)
        self.conv_mixer_layers2 = Block(nf, 2)
        self.conv_mixer_layers3 = Block(nf, 2)
        self.conv1 = conv_layer(nf, nf, kernel_size=3)
        self.esa1 = ESA(nf, nn.Conv2d)
        self.conv_mixer_layers4 = Block(nf, 2)
        self.conv_mixer_layers5 = Block(nf, 2)
        self.conv_mixer_layers6 = Block(nf, 2)
        self.conv_mixer_layers7 = Block(nf, 2)
        self.conv2 = conv_layer(nf, nf, kernel_size=3)
        self.esa2 = ESA(nf, nn.Conv2d)
        self.conv_mixer_layers8 = Block(nf, 2)
        self.conv_mixer_layers9 = Block(nf, 2)
        
        self.conv_mixer_layers10 = Block(nf, 2)
        self.conv_mixer_layers11 = Block(nf, 2)
        self.conv3 = conv_layer(nf, nf, kernel_size=3)
        self.esa3 = ESA(nf, nn.Conv2d)  
        
        self.conv_mixer_layers12 = Block(nf, 2)
        self.conv_mixer_layers13 = Block(nf, 2)
        self.conv_mixer_layers14 = Block(nf, 2)
        self.conv_mixer_layers15 = Block(nf, 2)
        self.conv4 = conv_layer(nf, nf, kernel_size=3)
        self.esa4 = ESA(nf, nn.Conv2d)
        self.conv_mixer_layers16 = Block(nf, 2)
        self.conv_mixer_layers17 = Block(nf, 2)
        self.conv_mixer_layers18 = Block(nf, 2)
        self.conv_mixer_layers19 = Block(nf, 2)
        self.conv5 = conv_layer(nf, nf, kernel_size=3)
        self.esa5 = ESA(nf, nn.Conv2d)

        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type="lrelu")

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=scale)
        self.scale_idx = 0

    def forward(self, input):
        out_fea = self.fea_conv(input)
  
        out_B1 = self.conv_mixer_layers0(out_fea)
        out_B2 = self.conv_mixer_layers1(out_B1)
        out_B3 = self.conv_mixer_layers2(out_B2)
        out_B4 = self.conv_mixer_layers3(out_B3)
        out_B4=self.conv1(out_B4)
        out_B4 = self.esa1(out_B4)
        out_B5 = self.conv_mixer_layers4(out_B4)
        out_B6 = self.conv_mixer_layers5(out_B5)
        out_B7 = self.conv_mixer_layers6(out_B6)
        out_B8 = self.conv_mixer_layers7(out_B7)
        out_B8=self.conv2(out_B8)
        out_B8 = self.esa2(out_B8)
        
        out_B9 = self.conv_mixer_layers8(out_B8)
        out_B10 = self.conv_mixer_layers9(out_B9)

        out_B11 = self.conv_mixer_layers10(out_B10)
        out_B12 = self.conv_mixer_layers11(out_B11)
        out_B12=self.conv3(out_B12)
        out_B12 = self.esa2(out_B12)
        out_B13 = self.conv_mixer_layers12(out_B12)
        out_B14 = self.conv_mixer_layers13(out_B13)
        out_B15 = self.conv_mixer_layers14(out_B14)
        out_B16 = self.conv_mixer_layers15(out_B15)
        out_B16=self.conv4(out_B16)
        out_B16 = self.esa4(out_B16)
        out_B17 = self.conv_mixer_layers16(out_B16)
        out_B18 = self.conv_mixer_layers17(out_B17)
        out_B19 = self.conv_mixer_layers18(out_B18)
        out_B20 = self.conv_mixer_layers19(out_B19)          
        out_B20=self.conv5(out_B20)
        out_B20 = self.esa5(out_B20)

        
        out_B = self.c(
            torch.cat(
                [   out_fea,
                    out_B4,
                    out_B8,
                    out_B12,
                    out_B16,
                    out_B20
                ],
                dim=1,
            )
        )
 
        out_lr = self.LR_conv(out_B) + out_fea
          
        output = self.upsampler(out_lr)
 
        return output  
                
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('msa') or name.find('a') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('msa') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
