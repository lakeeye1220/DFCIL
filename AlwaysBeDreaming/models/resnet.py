
'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

#__all__ = ['ResNet','resnet32']
__all__ = ['ResNet','resnet32','ResNet18','ResNet34','ResNet50','ResNet101','ResNet152']


def _weights_init(m):
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.last = nn.Linear(64, num_classes)

        self.apply(_weights_init)

        self.middle1= nn.Linear(16,16) 
        self.middle2= nn.Linear(32,32)
        self.middle3= nn.Linear(64,64)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, middle=False, pen=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = F.avg_pool2d(out3, out3.size()[3])
        out_pen = out.view(out.size(0), -1)
        if pen:
            return out_pen

        if middle:
            #out1 = F.adaptive_avg_pool2d(out1,(1,1))
            #out1 = out1.view(out1.size(0),-1)
            #out1_m = self.middle1(out1)

            #out2 = F.adaptive_avg_pool2d(out2,(1,1))
            #out2 = out2.view(out2.size(0),-1)
            #out2_m = self.middle2(out2)

            #out3 = F.adaptive_avg_pool2d(out3,(1,1))
            #out3 = out3.view(out3.size(0),-1)
            #out3_m = self.middle3(out3)


            #return out, out1_m, out2_m, out3_m
            return out, out_pen, out1,out2,out3

        else:
            out = self.last(out_pen)
            return out


class ResNet_ori(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,p=0.0):
        super(ResNet_ori, self).__init__()
        self.in_planes = 64
        self.p=p
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.last = nn.Linear(512*block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, pen=False):
        x = self.conv1(x)
        x = self.bn1(x)
        out0 = F.relu(x)
        out = self.layer1(out0)
        out1 = self.layer2(out)
        out2 = self.layer3(out1)
        out3 = self.layer4(out2)
        out4 = self.avgpool(out3)
        feature = out4.view(out4.size(0), -1)
        #img = self.last(feature)
        '''
        if out_feature == False:
            #return img, out3, out2, out1, out, out0
            return img
        else:
            return img, feature, out3, out2, out1, out, out0
        '''
        if pen:
            return feature
        else:
            img=self.last(feature)
            return img

        
def ResNet18(out_dim):
    return ResNet_ori(BasicBlock, [2,2,2,2], num_classes=out_dim)
    
    
def ResNet34(out_dim):
    return ResNet_ori(BasicBlock, [3,4,6,3], num_classes=out_dim)
    
def ResNet50(out_dim):
    return ResNet_ori(Bottleneck, [3,4,6,3], num_classes=out_dim)


def resnet32(out_dim):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=out_dim)

def resnet18(out_dim):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=out_dim)

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))
    
    def forward(self, x):
        return self.alpha * x + self.beta
    
    def printParam(self, i):
        print(i, self.alpha.item(), self.beta.item())
