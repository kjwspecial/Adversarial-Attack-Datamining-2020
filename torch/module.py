import torch
import torch.nn as nn
from torch.nn import init

from filter.guided_filter import FastGuidedFilter

# 모델 및 가중치 초기화
def weights_init_identity(m):
    classname = m.__class__.__name__
    #Conv weight 초기화
    if classname.find('Conv') != -1:
        n_out, n_in, h, w = m.weight.data.size()
        # Last Layer
        if n_out < n_in:
            init.xavier_uniform(m.weight.data)
            return

        # Last Layer 제외
        m.weight.data.zero_()
        ch, cw = h // 2, w // 2
        for i in range(n_in):
            m.weight.data[i, i, ch, cw] = 1.0
            
    #batch_norm 초기화
    elif classname.find('BatchNorm2d') != -1:
        init.constant(m.weight.data, 1.0)
        init.constant(m.bias.data,   0.0)
        
class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)
    
# FCNN  - 원래 목적은 주변 픽셀들 보간, layer가 너무 깊어도 학습이 어려움
def build_lr_net(norm=AdaptiveNorm, layer=5):
    # 사이즈 그대로 유지
    layers = [
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(32),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    #사이즈 그대로 유지
    for l in range(1, layer):
        layers += [nn.Conv2d(32,  32, kernel_size=3, stride=1, padding=2**l,  dilation=2**l,  bias=False),
                   norm(32),
                   nn.LeakyReLU(0.2, inplace=True)]
    #사이즈 그대로 유지
    layers += [
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(32),
        nn.LeakyReLU(0.2, inplace=True),
    # fully connected layer 대신 1d conv를 이용하여 위치 정보 저장
        nn.Conv2d(32,  3, kernel_size=1, stride=1, padding=0, dilation=1)
    ]

    net = nn.Sequential(*layers)

    net.apply(weights_init_identity)

    return net

class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super(DeepGuidedFilter, self).__init__()
        self.lr = build_lr_net()
        self.gf = FastGuidedFilter(radius, eps)
    # x_lr : 입력 저해상도 이미지
    # x_hr : 입력 고해상도 이미지
    # self.lr(x_lr) : 출력 저해상도 이미지
    def forward(self, x_lr, x_hr):
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path), strict=False)