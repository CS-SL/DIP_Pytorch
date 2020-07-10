import torch
import torch.nn as nn
import torch.nn.functional as F 
import sparsity

def pixel_unshuffle(input, downscale_factor):
    '''
    input: n, c, k*h, k*w
    downscale_factor: k
    n, c, k*w * k*h -> n, k*k*c, h, w
    '''
    c = input.shape[1]
    kernel = torch.zeros(size = [downscale_factor * downscale_factor * c, 1, downscale_factor, downscale_factor],
                        device = input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnShuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.downscale_factor = downscale_factor
    def forward(self, input):
        return pixel_unshuffle(input, self.downscale_factor)

class UpSampling(nn.Module):
    '''feature upsampling by pixel_shuffle'''
    def __init__(self, scale=4, n_feat=32, out_feat=3):
        super(UpSampling, self).__init__()
        self.scale = scale
        
        if self.scale == 4:
            self.up_conv_1 = nn.Conv2d(n_feat, n_feat*4, kernel_size=3, stride=1, padding=1)
            self.up_conv_2 = nn.Conv2d(n_feat, n_feat*4, kernel_size=3, stride=1, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        else: # scale 2 or 3
            c_feat = n_feat * scale * scale
            self.up_conv_1 = nn.Conv2d(n_feat, c_feat, kernel_size=3, stride=1, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(scale)

        self.conv_last = nn.Conv2d(n_feat, out_feat, kernel_size=3, stride=1, padding=1) 
    
    def forward(self, x):
        if self.scale == 4:
            x = self.up_conv_1(x)
            x = F.relu(self.pixel_shuffle(x), inplace=True)
            x = self.up_conv_2(x)
            x = self.pixel_shuffle(x)
        else:
            x = self.up_conv_1(x)
            x = self.pixel_shuffle(x)

        return self.conv_last(x)


class Generator(nn.Module):
    def __init__(self, scale=2):
        super(Generator, self).__init__()
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.down1 = PixelUnShuffle(2) # 1,32,h,w -> 1,32*2*2,h/2, w/2

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(32*2*2, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.down2 = PixelUnShuffle(2) # 1,32,h,w -> 1,32*2*2,h/4, w/4

        self.down_conv3 = nn.Sequential(
            nn.Conv2d(32*2*2, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.down3 = PixelUnShuffle(2) # 1,32,h,w -> 1,32*2*2,h/8, w/8

        self.down_conv4 = nn.Sequential(
            nn.Conv2d(32*2*2, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(32, 32*2*2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2)
        )
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(64, 64*2*2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2)
        )
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(64+32, 64+32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64+32),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(96, 96*2*2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2)
        )
        self.up_conv3 = nn.Sequential(
            nn.Conv2d(96+32, 96+32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(96+32),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.conv = nn.Conv2d(96+32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.up_conv = UpSampling(scale=scale)

    def forward(self, noise):
        d1 = self.down_conv1(noise)
        
        d2 = self.down1(d1)
        d2 = self.down_conv2(d2)

        d3 = self.down2(d2)
        d3 = self.down_conv3(d3)

        d4 = self.down3(d3)
        d4 = self.down_conv4(d4)

        up1 = self.up1(d4)
        up1 = self.up_conv1(torch.cat([up1, d3], dim=1))

        up2 = self.up2(up1)
        up2 = self.up_conv2(torch.cat([up2, d2], dim=1))

        up3 = self.up3(up2)
        up3 = self.up_conv3(torch.cat([up3, d1], dim=1))

        up3 = self.conv(up3)
        up3 = self.up_conv(up3)
        return up3



class Downsampler(nn.Module):
    def __init__(self, scale):
        super(Downsampler, self).__init__()
        self.downsampler = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            PixelUnShuffle(scale),
            nn.Conv2d(32*scale*scale, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        '''
        x:n, c, h, w
        output:n, c, h/scale, w/scale
        '''
        return self.downsampler(x)

if __name__ == "__main__":
    x = torch.randn(1, 3, 128, 128)
    model = Generator()
    out_x = model(x)
    print(out_x, out_x.shape)

    down = Downsampler(scale=2)
    x_ = down(out_x)
    print(x_, x_.shape)
    l1 = nn.L1Loss()
    loss = l1(x, x_)
    print('====')
    print(loss)
    

        
        
