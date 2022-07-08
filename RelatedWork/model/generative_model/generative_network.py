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
import torchvision.utils as vutils

import math
import numpy as np
import os
import collections
import sys 
import os
from tqdm import tqdm
import torch.nn.functional as F
NUM_CLASSES = 100
ALPHA=1.0
image_list=[]
target_list=[]

debug_output = False
debug_output = True

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
                feature_block_num=4,
                latent_dim=100,
                configs=None,
                device='cuda',

            ):

    kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    best_cost = 1e6
   
    net.eval()
    minimum_per_class=num_generate_images//num_classes[0]
    num_cls_targets=np.zeros(num_classes[0])

    best_inputs_list=[]
    best_targets_list = []
    def softmax(x,axis=0):
        exp_x=np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    if configs['network_ver']==1:
        from model.generative_model.network_v1 import Generator,Feature_Decoder
        generator_class = Generator
        #### Feature_Map Decoder
        feature_decoder_class = Feature_Decoder
    elif configs['network_ver']==2:
        from model.generative_model.network_v2 import Generator,Feature_Decoder
        generator_class = Generator
        #### Feature_Map Decoder
        feature_decoder_class = Feature_Decoder
    elif configs['network_ver']==3:
        from model.generative_model.network_v3 import Generator,Feature_Decoder
        generator_class = Generator
        feature_decoder_class = Feature_Decoder
    while np.count_nonzero(num_cls_targets>=minimum_per_class)<num_classes[0]:
        generator = generator_class(8,latent_dim+num_classes[0],3).to(device)
        feature_decoder = feature_decoder_class(feature_block_num).to(device)
        generator.train()
        feature_decoder.train()
        optimizer_g = optim.Adam(generator.parameters(), lr=g_lr)
        optimizer_f = torch.optim.Adam(feature_decoder.parameters(), lr=d_lr)


        # set up criteria for optimization
        criterion = nn.CrossEntropyLoss()
        optimizer_g.state = collections.defaultdict(dict)
        optimizer_f.state = collections.defaultdict(dict)  # Reset state of optimizer

        np_targets=np.random.choice(num_classes[0],bs)
        targets = torch.LongTensor(np_targets).to(device)
        onehot_targets= F.one_hot(targets, num_classes[0]).float().to(device)

        z = torch.randn((bs, latent_dim)).to(device)
        z = torch.cat((z,onehot_targets), dim = 1)
        
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

            ##### step3 ACS
            # inputs_jit = inputs_jit * alpha
            inputs_for_save = inputs_jit.data.clone()

            # apply random jitter offsets
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1,off2), dims=(2,3))
        
            # Apply random flip 
            flip = random.random() > 0.5
            if flip:
                inputs_jit = torch.flip(inputs_jit, dims = (3,))
            with torch.no_grad():
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

            if debug_output and epoch % int(epochs/2)==0:
                print("\rIt {:4d}\tLosses: total: {:7.3f},\ttarget: {:5.3f} \tR_feature_loss unscaled:\t {:6.3f}".format(epoch, loss.item(),loss_target,loss_distr.item()),end='')

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
                    # best_inputs *= alpha
        
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            # optimizer_alpha.zero_grad()

            # backward pass
            loss.backward()

            optimizer_g.step()
            optimizer_f.step()
            # optimizer_alpha.step()

        best_inputs_list.append(best_inputs.cpu().detach().numpy())
        best_targets_list.append(targets.cpu().detach().numpy())
        for mod in loss_r_feature_layers:
            mod.close()
        for target in targets.cpu().detach().numpy():
            num_cls_targets[target]+=1
        print("Goal num_cls_targets : {} / Current [{:.2f}%] {}".format(minimum_per_class, 100.*min(num_cls_targets)/minimum_per_class,num_cls_targets))
        # optimizer_alpha.zero_grad(set_to_none=True)
        optimizer_f.zero_grad(set_to_none=True)
        optimizer_g.zero_grad(set_to_none=True)
        del generator
        del feature_decoder
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



