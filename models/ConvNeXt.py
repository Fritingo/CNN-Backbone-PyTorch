# @Article{liu2022convnet,
#   author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
#   title   = {A ConvNet for the 2020s},
#   journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#   year    = {2022},
# }

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from thop import profile


class Block(nn.Module):
    #DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)

    def __init__(self, dim, drop_path=0., norm_size=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.pwconv1 = nn.Conv2d(dim, 4*dim, 1, stride=1, padding=0) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4*dim, dim, 1, stride=1, padding=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = nn.LayerNorm(norm_size, eps=1e-6)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x
    
class ConvNeXt(nn.Module):
    
    def __init__(self, in_chans=3, input_size=224, class_num=100, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., head_init_scale=1.,):
        super().__init__()
        
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.stem_conv = nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4)

        norm_size_list = []
        # compute shape by doing a forward pass
        with torch.no_grad():
            fake_input = torch.randn(1, in_chans, input_size, input_size)
            out        = self.stem_conv(fake_input)
            norm_size  = out.size()[1:]
            norm_size_list.append(norm_size)
            for i in range(3):
                norm_size_list.append([dims[i+1], norm_size[1]//(2**(i+1)), norm_size[2]//(2**(i+1))])

        
        stem = nn.Sequential(
            self.stem_conv,
            nn.LayerNorm(norm_size, eps=1e-6)
        )

        self.downsample_layers.append(stem)

        for i in range(3):
            
            downsample_layer = nn.Sequential(
                    nn.LayerNorm(norm_size_list[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], norm_size=norm_size_list[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], class_num)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    

def convnext_tiny(input_size=224, class_num=100, **kwargs):
    model = ConvNeXt(input_size=input_size, class_num=class_num, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


if __name__ == '__main__':
    input = torch.randn(16, 3, 224, 224)
    model = convnext_tiny(224, 100)
    print(model)
    y = model(input)
    print('output shape:', y.size())

    # time.sleep(10)
    flops, params = profile(model, inputs=(input, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

