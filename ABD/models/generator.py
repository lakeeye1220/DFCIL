import torch
import torch.nn as nn

"""
@article{fang2019datafree,
    title={Data-Free Adversarial Distillation},	
    author={Gongfan Fang and Jie Song and Chengchao Shen and Xinchao Wang and Da Chen and Mingli Song},	  
    journal={arXiv preprint arXiv:1912.11006},	
    year={2019}
}
"""

class Generator(nn.Module):
    def __init__(self, zdim, in_channel, img_sz):
        super(Generator, self).__init__()
        self.z_dim = zdim

        self.init_size = img_sz // 4
        self.l1 = nn.Sequential(nn.Linear(zdim, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img

    def sample(self, size):
        
        # sample z
        z = torch.randn(size, self.z_dim)
        z = z.cuda()
        X = self.forward(z)
        return X

class GeneratorMed(nn.Module):
    def __init__(self, zdim, in_channel, img_sz):
        super(GeneratorMed, self).__init__()
        self.z_dim = zdim

        self.init_size = img_sz // 8
        self.l1 = nn.Sequential(nn.Linear(zdim, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks3(img)
        return img

    def sample(self, size):
        
        # sample z
        z = torch.randn(size, self.z_dim)
        z = z.cuda()
        X = self.forward(z)
        return X

class GeneratorBig(nn.Module):
    def __init__(self, zdim, in_channel, img_sz):
        super(GeneratorBig, self).__init__()
        self.z_dim = zdim

        self.init_size = img_sz // (2**5)
        self.l1 = nn.Sequential(nn.Linear(zdim, 64*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(64),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks6 = nn.Sequential(
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 64, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks3(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks4(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks5(img)
        img = self.conv_blocks6(img)
        return img

    def sample(self, size):
        
        # sample z
        z = torch.randn(size, self.z_dim)
        z = z.cuda()
        X = self.forward(z)
        return X


class NIGenerator(nn.Module):
    def __init__(self, zdim, in_channel,image_size, num_classes=100, initial=True):
        super(NIGenerator, self).__init__()
        self.init_size = image_size//4
        self.image_size = image_size
        self.l3 = nn.Linear(zdim+num_classes, 128 * self.init_size ** 2)
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(128, 128, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 3, 3, stride = 1, padding = 1),
        )

        self.zdim=zdim
        self.in_channel=in_channel
        self.initial=initial
        self.num_classes=num_classes
    
        if initial:
            self.init()

    def init(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight.data, 0.0, 0.02)

    def forward(self, z):
        out1 = self.l3(z)
        out = out1.view(out1.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_block(out)
        return img 

    def reset(self,num_classes):
        self=self.__init__(self.zdim, self.in_channel, self.image_size, num_classes, self.initial)


class Feature_Decoder(nn.Module):
    def __init__(self,feature_block_num=4):
        super(Feature_Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor = 2)
        self.feature_block_num = feature_block_num
        if feature_block_num==4:
            self.conv1 = nn.Conv2d(512, 256, 1, stride = 1, padding = 0)
            self.conv2 = nn.Conv2d(256, 128, 1, stride = 1, padding = 0)
            self.conv3 = nn.Conv2d(128, 64, 1, stride = 1, padding = 0)
            self.conv4 = nn.Conv2d(64, 3, 1, stride = 1, padding = 0)
            self.conv5 = nn.Conv2d(3, 3, 1, stride = 1, padding = 0)
            self.conv_31 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
            self.conv_32 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
            self.conv_33 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        elif feature_block_num==3:
            self.conv1 = nn.Conv2d(64, 32, 1, stride = 1, padding = 0)
            self.conv2 = nn.Conv2d(32, 16, 1, stride = 1, padding = 0)
            self.conv3 = nn.Conv2d(16, 3, 1, stride = 1, padding = 0)
            self.conv4 = nn.Conv2d(3, 3, 1, stride = 1, padding = 0)
            self.conv_31 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
            self.conv_32 = nn.Conv2d(16, 16, 3, stride=1, padding=1)


    def forward(self, x, features):
        if self.feature_block_num==4:
            out = self.conv1(self.upsample(features[-2]))
            out = self.conv_31(out + features[-3])
            
            out = self.conv2(self.upsample(out))
            out = self.conv_32(out + features[-4])
            
            out = self.conv3(self.upsample(out))
            out = self.conv_33(out + features[-5])
            
            out_ = self.conv4(out)
            out = (x + out_)
            out = self.conv5(out)
        elif self.feature_block_num==3:
            out = self.conv1(self.upsample(features[-2]))
            out = self.conv_31(out + features[-3])
            
            out = self.conv2(self.upsample(out))
            out = self.conv_32(out + features[-4])
            
            out_ = self.conv3(out)
            out = (x + out_)
            out = self.conv4(out)
        out = torch.tanh(out)
        
        return out, out_
    
    def reset(self):
        self.__init__(self.feature_block_num)

def CIFAR_GEN(bn = False):
    return Generator(zdim=1000, in_channel=3, img_sz=32)
def CIFAR_GEN_NI(bn = False):
    return NIGenerator(zdim=1000, in_channel=3, image_size=32)

def TINYIMNET_GEN(bn = False):
   return GeneratorMed(zdim=1000, in_channel=3, img_sz=64)

def IMNET_GEN(bn = False):
    return GeneratorBig(zdim=1000, in_channel=3, img_sz=224)