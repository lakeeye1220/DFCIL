import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, numclass, initial=True):
        super(Generator, self).__init__()
        self.init_size = image_size//4
        self.embed=nn.Embedding(numclass,int(latent_dim/2))
        self.l3 = nn.Linear(latent_dim, 128 * self.init_size ** 2)
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
        latent_vec=self.embed(z[1])
        out1 = self.l3(torch.cat([z[0],latent_vec],1))
        out = out1.view(out1.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_block(out)
        return img 


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