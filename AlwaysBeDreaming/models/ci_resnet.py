
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
import numpy as np

__all__ = ['ResNet','resnet32','resnet18']

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

class WALinear(nn.Module):
    def __init__(self,in_features,out_features,task_num):
        super(WALinear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.task_num = task_num
        self.sub_num_classes = self.out_features//self.task_num
        self.WA_linears = nn.ModuleList()
        self.WA_linears.extend([nn.Linear(self.in_features,self.sub_num_classes, bias=False ) for i in range(self.task_num)])

    def forward(self,x):
        out_list=[]
        for i in range(self.task_num):
            out = self.WA_linears[i](x)
            out_list.append(out)
        return torch.cat(out_list,dim=1)


    def align_norms(self,step_b):
        new_layer = self.WA_linears[step_b]
        old_layer = self.WA_linears[:step_b]
        #print(old_layer[0].weight)
        new_weight = new_layer.weight.cpu().detach().numpy()
        for i in range(step_b):
            old_weight = np.concatenate([old_layer[i].weight.cpu().detach().numpy() for i in range(step_b)])
        print("old weight's shape is :",old_weight.shape)
        print("new weight's shape is " ,new_weight.shape)

        Norm_of_new = np.linalg.norm(new_weight,axis=1)
        Norm_of_old = np.linalg.norm(old_weight,axis=1)

        assert(len(Norm_of_new)==self.task_num)
        assert(len(Norm_of_old)==self.task_num*step_b)

        gamma = np.mean(Norm_of_new)/np.mean(Norm_of_old)
        print("Gamma : ",gamma)

        update_new_weight = torch.Tensor(gamma*new_weight).cuda()
        #print("new weight : ",new_weight[0])
        #print("update new weight : ",update_new_weight[0])
        #print("old weight : ",old_layer[0].weight)
        self.WA_linears[step_b].weight = torch.nn.Parameter(update_new_weight)



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
    def __init__(self, block, num_blocks, num_classes=10,task_num=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        #self.last = nn.Linear(64, num_classes)
        self.last = WALinear(64,num_classes,task_num)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, pen=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        if pen:
            return out
        else:
            out = self.last(out)
            return out

    def weight_align(self,step_b):
        self.last.align_norms(step_b)

def resnet32(out_dim,task_num):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=out_dim,task_num=task_num)

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
