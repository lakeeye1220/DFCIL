from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import math
import numbers
import numpy as np
import os
import glob
import collections
from PIL import Image
import sys 
import pickle
import os
import itertools
from tqdm import tqdm
from NaturalInversion.network import Generator, Feature_Decoder

sys.path.insert(0,'..')
from iCaRL.ResNet import resnet34_cbam as ResNet34
#from models.resnet import ResNet34
#import learners

sys.path.insert(0,os.path.abspath('..'))

NUM_CLASSES = 100
ALPHA=1.0
image_list=[]
target_list=[]

debug_output = False
debug_output = True

# To fix Seed
def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


class NaturalInversionFeatureHook():
    def __init__(self, module, rs, mean=None, var= None):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.rs = rs
        self.rankmean = mean
        self.rankvar=var

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type())  - mean, 2)

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def make_grid(tensor, nrow, padding = 2, pad_value : int = 0):
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


def get_inversion_images(net, 
                num_classes=10,
                task=0,
                bs=256,
                filename='model.pth',
                targets = None,
                epochs=2000, 
                prefix=None, 
                global_iteration=0, 
                bn_reg_scale=10,
                g_lr=0.001,
                d_lr=0.0005,
                a_lr=0.05,
                var_scale=0.001, 
                l2_coeff=0.00001,
                num_generate_images=2000,
            ):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    best_cost = 1e6
   
    net.eval()

    best_inputs_list=[]
    best_targets_list = []
    for iters_ in range(int(num_generate_images/bs)):
        generator = Generator(8,1024,3).to(device)
        optimizer_g = optim.Adam(generator.parameters(), lr=g_lr)

        #### Feature_Map Decoder
        feature_decoder = Feature_Decoder().to(device)
        optimizer_f = torch.optim.Adam(feature_decoder.parameters(), lr=d_lr)

        # Learnable Scale Parameter
        alpha = torch.empty((bs,3,1,1), requires_grad=True, device=device)
        torch.nn.init.normal_(alpha, 5.0, 1)
        optimizer_alpha = torch.optim.Adam([alpha], lr=a_lr)

        # set up criteria for optimization
        criterion = nn.CrossEntropyLoss()
        optimizer_g.state = collections.defaultdict(dict)
        optimizer_f.state = collections.defaultdict(dict)  # Reset state of optimizer
        optimizer_alpha.state = collections.defaultdict(dict)
        print("----------------------------------------num_classes[0] : ",num_classes[0])
        np_targets = np.random.choice(num_classes[0],bs)
        targets = torch.LongTensor(np_targets).to('cuda')
        z = torch.randn((bs, 1024)).to(device)
        
        loss_r_feature_layers = []
        count = 0
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(NaturalInversionFeatureHook(module, 0))
    
        lim_0, lim_1 = 2, 2

        for epoch in tqdm(range(epochs), leave=False, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            inputs_jit = generator(z)
        
            # apply random jitter offsets
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1,off2), dims=(2,3))
        
            # Apply random flip 
            flip = random.random() > 0.5
            if flip:
                inputs_jit = torch.flip(inputs_jit, dims = (3,))
        
            ##### step2
            input_for_f = inputs_jit.clone().detach()
            with torch.no_grad():
                _, features = net(input_for_f)
        
            inputs_jit, addition = feature_decoder(inputs_jit, features)

            ##### step3
            inputs_jit = inputs_jit * alpha
            inputs_for_save = inputs_jit.data.clone()

            # apply random jitter offsets
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1,off2), dims=(2,3))
        
            # Apply random flip 
            flip = random.random() > 0.5
            if flip:
                inputs_jit = torch.flip(inputs_jit, dims = (3,))
            outputs, features = net(inputs_jit)
        
            loss_target = criterion(outputs, targets)
            loss = loss_target

            # apply total variation regularization
            diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
            diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
            diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
            diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
            loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
            loss = loss + var_scale*loss_var

            # R_feature loss
            loss_distr = sum([mod.r_feature for idx, mod in enumerate(loss_r_feature_layers)])
            loss = loss + bn_reg_scale*loss_distr # best for noise before BN

            # l2 loss
            loss = loss + l2_coeff * torch.norm(inputs_jit, 2)

            if debug_output and epoch % 100==0:
                print("It {}\t Losses: total: {:.3f},\ttarget: {:.3f} \tR_feature_loss unscaled:\t {:.3f}\tstyle_loss : {:.3f}".format(epoch, loss.item(),loss_target,loss_distr.item(), 0))
                nchs = inputs_jit.shape[1]

                save_pth = os.path.join(prefix, 'inversion_images', 'task{}'.format(task))
                if not os.path.exists(save_pth):
                    os.makedirs(save_pth)
    
                vutils.save_image(inputs_jit.data.clone(),os.path.join(save_pth,'generator_{}.png'.format(str(epoch//100).zfill(2))),normalize=True, scale_each=True, nrow=10)
                vutils.save_image(inputs_for_save,os.path.join(save_pth,'save_{}.png'.format(str(epoch//100).zfill(2))),normalize=True, scale_each=True, nrow=10)

            if best_cost > loss.item():
                best_cost = loss.item()
                with torch.no_grad():
                    best_inputs = generator(z)
                    _, features = net(best_inputs)
                    best_inputs, addition = feature_decoder(best_inputs, features)
                    best_inputs *= alpha
        
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            optimizer_alpha.zero_grad()

            # backward pass
            loss.backward()

            optimizer_g.step()
            optimizer_f.step()
            optimizer_alpha.step()

        best_inputs_list.append(best_inputs.cpu().detach().numpy())
        best_targets_list.append(targets.cpu().detach().numpy())
    
    return best_inputs_list, best_targets_list


def save_finalimages(images, targets, num_generations, prefix, exp_descr):
    # method to store generated images locally
    local_rank = torch.cuda.current_device()
    images = images.data.clone()

    for id in range(images.shape[0]):
        class_id = str(targets[id].item()).zfill(2)
        image = images[id].reshape(3,32,32)
        image_np = images[id].data.cpu().numpy()
        pil_image = torch.from_numpy(image_np)

        save_pth = os.path.join(prefix, 'final_images/s{}'.format(class_id))
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        vutils.save_image(image, os.path.join(prefix, 'final_images/s{}/{}_output_{}_'.format(class_id, num_generations, id)) + exp_descr + '.png', normalize=True, scale_each=True, nrow=1)