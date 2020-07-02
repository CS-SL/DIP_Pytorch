
import torch
import torch.nn as nn
import torch.nn.functional as F 

################
### Sparsity ###
################
# 1. Obtain a sparse vector of LR image
class SoftThreshold(nn.Module):
    def __init__(self, theta=0.1):
        super(SoftThreshold, self).__init__()

        self.theta = nn.Parameter(torch.tensor(theta), requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_ = torch.abs(x) - self.theta
        x = torch.sign(x) * self.relu(x_)
        return x

class SparseBlock(nn.Module):
    def __init__(self, in_feat=3, out_feat=32):
        super(SparseBlock, self).__init__()
        self.g = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, dilation=5, padding=5, bias=False)
        self.s0 = SoftThreshold(theta=0.2)

        self.v1 = nn.Conv2d(out_feat, in_feat, kernel_size=3, stride=1, dilation=5, padding=5, bias=False)
        self.t1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, dilation=5, padding=5, bias=False)
        self.s1 = SoftThreshold(theta=0.2)
        
        self.v2 = nn.Conv2d(out_feat, in_feat, kernel_size=3, stride=1, dilation=5, padding=5, bias=False)
        self.t2 = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, dilation=5, padding=5, bias=False)
        self.s2 = SoftThreshold(theta=0.2)

    def forward(self, x):
        g = self.g(x)
        s0 = self.s0(g)

        v1 = self.v1(s0)
        t1 = self.t1(v1)
        s1 = s0 - t1 + g
        s1 = self.s1(s1)

        v2 = self.v2(s1)
        t2 = self.t2(v2)
        s2 = s1 - t2 + g
        s2 = self.s2(s2)

        return s2

# 2. Embedding the Sparsity prior to the module        
class SFTLayer(nn.Module):
    def __init__(self, in_feat=32, out_feat=64):
        super(SFTLayer, self).__init__()
        self.scale_conv0 = nn.Conv2d(in_feat, in_feat, kernel_size=3, stride=1, padding=1, bias=False)
        self.scale_conv1 = nn.Conv2d(in_feat, in_feat, kernel_size=3, stride=1, padding=1, bias=False)

        self.scale_conv2 = nn.Conv2d(in_feat, in_feat, kernel_size=1, stride=1, padding=0, bias=False)
        self.scale_conv3 = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=1, padding=0, bias=False)

        self.shift_conv0 = nn.Conv2d(in_feat, in_feat, kernel_size=1, stride=1, padding=0, bias=False)
        self.shift_conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, side_x):
        theta = self.scale_conv1(F.leaky_relu(self.scale_conv0(side_x), 0.1, inplace=True))
        gamma = self.scale_conv3(F.leaky_relu(self.scale_conv2(theta), 0.1, inplace=True))
        beta = self.shift_conv1(F.leaky_relu(self.shift_conv0(theta), 0.1, inplace=True))
        return gamma * x + beta
