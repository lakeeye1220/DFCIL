import torch
import torch.nn as nn
import torch.nn.functional as F
import random


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
        self.embedding = nn.Embedding(100,1)
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

    def sample(self,config,task_idx,size,v,m):
        #gt_idx = sum(config['gt_idx'],[]) #100
        if self.z_dim ==1000:
            z = torch.randn(size, self.z_dim).cuda()
            if v is not None and m is not None:
                z = z*v + m
                #print("generator variation: ",v,"mean ",m)
            z = z.cuda()
            X = self.forward(z)
            return X
        else:
            cls_idx_list = [i for i in gt_idx[:task_idx]] #task idx: 20 40 60 80 100
            rand_pick = config['batch_size']%task_idx
            rand_picked_cls = random.sample(cls_idx_list,rand_pick)
            picked = config['batch_size']-rand_pick
            fake_targets = cls_idx_list*int(picked/len(cls_idx_list)) + rand_picked_cls
            fake_targets = torch.LongTensor(fake_targets)
            oh_targets = F.one_hot(fake_targets,config['num_classes']).cuda()
            
            #em_targets = self.embedding(oh_targets)
            #em_targets = em_targets.view(em_targets.shape[0],-1).cuda()
            #print("em targets shape : ",type(em_targets))
            #print("oh_ tarhget shape : ",type(oh_targets))
            z = torch.randn(size,self.z_dim-config['num_classes']).cuda()
            z = torch.cat((z,oh_targets),dim=1)
            z = z.cuda()
            X = self.forward(z)

            #return X, oh_targets,fake_targets
            return X, fake_targets 
            '''
            cls_idx_list = [i for i in gt_idx[:task_idx]] #task idx: 20 40 60 80 100
            #print(cls_idx_list)
            rand_pick = config['batch_size']%task_idx
            rand_picked_cls = random.sample(cls_idx_list,rand_pick)
            #print("random images : ",rand_picked_cls)
            #print("rand pick : ",rand_pick) 8
            picked = config['batch_size']-rand_pick
            #print("picked : ",picked) 120
            fake_targets = cls_idx_list*int(picked/len(cls_idx_list)) + rand_picked_cls
            #print("fake targets : ",fake_targets)
            #print("len of fake targets : ",len(fake_targets))
            #print("fake targest : ",fake_targets)
            fake_targets = torch.LongTensor(fake_targets)
            #print("fake targets generator 64.py : ",fake_targets)
            oh_targets = F.one_hot(fake_targets,config['num_classes']).cuda()
            z = torch.randn(size,self.z_dim-config['num_classes']).cuda()
            z = torch.cat((z,oh_targets),dim=1)
            z = z.cuda()
            X = self.forward(z)

            #return X, oh_targets,fake_targets
            return X 
            '''
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

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3,64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def CIFAR_GEN(bn = False,cgan=False):
    if cgan==False:
        return Generator(zdim=1000, in_channel=3, img_sz=32)
    else:
        return Generator(zdim=1100, in_channel=3, img_sz=32)

def CIFAR_DEC(bn = False):
    return Discriminator()

def TINYIMNET_GEN(bn = False):
   return GeneratorMed(zdim=1000, in_channel=3, img_sz=64)

def IMNET_GEN(bn = False):
    return GeneratorBig(zdim=1000, in_channel=3, img_sz=224)