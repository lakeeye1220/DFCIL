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


class CDISCGenerator(nn.Module):
    def __init__(self, zdim, in_channel, img_sz,num_classes=10):
        super(CDISCGenerator, self).__init__()
        self.z_dim = zdim
        self.embeddings=nn.Embedding(num_classes,zdim)
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
        self.num_classes=num_classes

    def forward(self, z, c):

        cls_z=self.embeddings(c)
        z=z*0.05+cls_z
        # z=torch.cat((z,cls_z),1)
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img
    
    def update_num_classes(self,num_classes):
        self.num_classes=num_classes
        self.embeddings=nn.Embedding(num_classes,self.z_dim).cuda()

    def sample(self, size):
        
        c=torch.randint(0,self.num_classes,(size,))
        # sample z
        z = torch.randn(size, self.z_dim)
        z = z.cuda()
        c=c.cuda()
        X = self.forward(z,c)
        return X,c

# latent based CGAN
class CLATENTGenerator(nn.Module):
    def __init__(self, zdim, in_channel, img_sz,num_classes=10):
        super(CLATENTGenerator, self).__init__()
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
        self.num_classes=num_classes

        # perclass
        self.perclass_mean=nn.Parameter(torch.ones(num_classes).view(-1,1))
        self.perclass_mean.requires_grad=True
        self.register_parameter('perclass_mean',self.perclass_mean)

        self.perclass_std=nn.Parameter(torch.ones(num_classes).view(-1,1))
        self.perclass_std.requires_grad=True
        self.register_parameter('perclass_std',self.perclass_std)

    def update_num_classes(self,num_classes):
        self.num_classes=num_classes
        self.perclass_mean=nn.Parameter(torch.ones(num_classes).cuda().view(-1,1))
        self.perclass_mean.requires_grad=True
        self.register_parameter('perclass_mean',self.perclass_mean)
        self.perclass_std=nn.Parameter(torch.ones(num_classes).cuda().view(-1,1))
        self.perclass_std.requires_grad=True
        self.register_parameter('perclass_std',self.perclass_std)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img
    

    def sample(self, size, model):
        # sample z
        z = torch.randn(size, model.last.weight.shape[1]).cuda()
        # sample class
        class_index=torch.randint(0,self.num_classes,(size,),dtype=torch.long).cuda()
        # sample perclass
        z = z * self.perclass_std[class_index] + self.perclass_mean[class_index]
        with torch.no_grad():
            y=model.last(z)[:,:self.num_classes].argmax(dim=1)
        z = torch.cat((z,torch.randn(size,self.z_dim-z.shape[1]).cuda()),dim=1)
        X = self.forward(z)
        return X, y, z

def CIFAR_GEN(bn = False, cgan=None, num_classes=10):
    if cgan is not None and 'disc' in cgan:
        return CDISCGenerator(zdim=1000, in_channel=3, img_sz=32, num_classes=num_classes)
    elif cgan == 'latent':
        return CLATENTGenerator(zdim=1000, in_channel=3, img_sz=32, num_classes=num_classes)
    else:
        return Generator(zdim=1000, in_channel=3, img_sz=32)

def TINYIMNET_GEN(bn = False):
   return GeneratorMed(zdim=1000, in_channel=3, img_sz=64)

def IMNET_GEN(bn = False):
    return GeneratorBig(zdim=1000, in_channel=3, img_sz=224)
