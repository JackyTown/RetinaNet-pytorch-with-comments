'''RetinaFPN in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

# 定义Bottleneck结构
class Bottleneck(nn.Module):
    expansion = 4

    # 该block的输入通道数为in_planes, 输出通道数为self.expansion * planes
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes: # 修正通道数
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x): # 若stride = 1, 则该block不改变输入的分辨率
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out))) 
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

# torchsummary, 可查看网络的输出结构
class FPN(nn.Module):
    def __init__(self, block, num_blocks): # block == Bottleneck
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        # (x + 3 * 2 - 7) / 2 + 1 = ceil(x / 2)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        # 输入通道: 64 输出通道: 64 * 4 
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        # 输入通道: 256 输出通道: 128 * 4 分辨率减半
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # 输入通道: 512 输出通道: 256 * 4 分辨率减半
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # 输入通道: 1024 输出通道: 512 * 4 分辨率减半
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # 输入通道: 2048 输出通道: 256 分辨率减半
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        # 输入通道: 256 输出通道: 256 分辨率减半
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # 每一个layer中,第一次卷积步长为stride,之后(num_blocks-1)层步长为1
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride)) # layers中存有各层网络的调用地址
            self.in_planes = planes * block.expansion # 修改self.in_planes(下一层的输入通道数为该层的输出通道数)
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x))) # conv1 (300,300)
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1) # (150,150)
        c2 = self.layer1(c1) # layer1 (150,150)
        c3 = self.layer2(c2) # layer2 (75,75)
        c4 = self.layer3(c3) # layer3 (38,38)
        c5 = self.layer4(c4) # layer4 (19,19)
        p6 = self.conv6(c5) # (10,10)
        p7 = self.conv7(F.relu(p6)) # (5,5)
        # Top-down
        p5 = self.latlayer1(c5) # 将通道数降为256, 上采样后叠加 (19,19)
        p4 = self._upsample_add(p5, self.latlayer2(c4)) # 通道数降为256,和上采样后的p5叠加
        p4 = self.toplayer1(p4) # (38,38)
        p3 = self._upsample_add(p4, self.latlayer3(c3)) # 通道数降为256,和上采样后的p4叠加
        p3 = self.toplayer2(p3) #(75,75)
        return p3, p4, p5, p6, p7 #(75,75), (38,38), (19,19), (10,10), (5,5)


def FPN50():
    return FPN(Bottleneck, [3,4,6,3])

def FPN101():
    return FPN(Bottleneck, [2,4,23,3])


def test():
    net = FPN50()
    fms = net(Variable(torch.randn(1,3,600,300))) # 返回1 * 3 * 600 * 300的标准正态分布
    # 想计算导数, 可以在 Variable 上调用 backward() 方法
    for fm in fms:
        print(fm.size())

# test()
