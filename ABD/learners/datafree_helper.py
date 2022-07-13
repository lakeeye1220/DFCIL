import PIL
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import random
import collections
import torch.optim as optim
import torchvision.utils as vutils
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from dataloaders.utils import get_transform
from models.generator import Feature_Decoder

from PIL import Image
"""
Some content adapted from the following:
@article{fang2019datafree,
    title={Data-Free Adversarial Distillation},	
    author={Gongfan Fang and Jie Song and Chengchao Shen and Xinchao Wang and Da Chen and Mingli Song},	  
    journal={arXiv preprint arXiv:1912.11006},	
    year={2019}
}
@inproceedings{yin2020dreaming,
	title = {Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion},
	author = {Yin, Hongxu and Molchanov, Pavlo and Alvarez, Jose M. and Li, Zhizhong and Mallya, Arun and Hoiem, Derek and Jha, Niraj K and Kautz, Jan},
	booktitle = {The IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)},
	month = June,
	year = {2020}
}
"""

class Teacher(nn.Module):

    def __init__(self, solver, generator, gen_opt, img_shape, iters, class_idx, deep_inv_params, train = True, config=None):

        super().__init__()
        self.solver = solver
        self.generator = generator
        self.gen_opt = gen_opt
        self.solver.eval()
        self.generator.eval()
        self.img_shape = img_shape
        self.iters = iters
        self.config = config

        # hyperparameters
        self.di_lr = deep_inv_params[0]
        self.r_feature_weight = deep_inv_params[1]
        self.di_var_scale = deep_inv_params[2]
        self.content_temp = deep_inv_params[3]
        self.content_weight = deep_inv_params[4]
        

        # get class keys
        self.class_idx = list(class_idx)
        self.num_k = len(self.class_idx)

        # first time?
        self.first_time = train

        # set up criteria for optimization
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduction="none").cuda()
        self.smoothing = Gaussiansmoothing(3,5,1)

        # Create hooks for feature statistics catching
        loss_r_feature_layers = []
        for module in self.solver.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module, 0, self.r_feature_weight))
        self.loss_r_feature_layers = loss_r_feature_layers


    def sample(self, size, device, return_scores=False):
        
        # make sure solver is eval mode
        self.solver.eval()

        # train if first time
        self.generator.train()
        if self.first_time:
            self.first_time = False
            self.get_images(bs=size, epochs=self.iters, idx=-1)
        
        # sample images
        self.generator.eval()
        with torch.no_grad():
            x_i = self.generator.sample(size)

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x_i)
        y_hat = y_hat[:, self.class_idx]

        # get predicted class-labels (indexed according to each class' position in [self.class_idx]!)
        _, y = torch.max(y_hat, dim=1)

        return (x_i, y, y_hat) if return_scores else (x_i, y)

    def generate_scores(self, x, allowed_predictions=None, return_label=False):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
        _, y = torch.max(y_hat, dim=1)

        return (y, y_hat) if return_label else y_hat


    def generate_scores_pen(self, x):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)

        return y_hat

    def get_images(self, bs=256, epochs=1000, idx=-1):

        # clear cuda cache
        torch.cuda.empty_cache()

        self.generator.train()
        for epoch in range(epochs):

            # sample from generator
            inputs = self.generator.sample(bs)

            # forward with images
            self.gen_opt.zero_grad()
            self.solver.zero_grad()

            # content
            outputs = self.solver(inputs)[:,:self.num_k]
            loss = self.criterion(outputs / self.content_temp, torch.argmax(outputs, dim=1)) * self.content_weight

            # class balance
            softmax_o_T = F.softmax(outputs, dim = 1).mean(dim = 0)
            loss += (1.0 + (softmax_o_T * torch.log(softmax_o_T) / math.log(self.num_k)).sum())

            # R_feature loss
            for mod in self.loss_r_feature_layers: 
                loss_distr = mod.r_feature * self.r_feature_weight / len(self.loss_r_feature_layers)
                if len(self.config['gpuid']) > 1:
                    loss_distr = loss_distr.to(device=torch.device('cuda:'+str(self.config['gpuid'][0])))
                loss = loss + loss_distr

            # image prior
            inputs_smooth = self.smoothing(F.pad(inputs, (2, 2, 2, 2), mode='reflect'))
            loss_var = self.mse_loss(inputs, inputs_smooth).mean()
            loss = loss + self.di_var_scale * loss_var

            # backward pass
            loss.backward()

            self.gen_opt.step()

        # clear cuda cache
        torch.cuda.empty_cache()
        self.generator.eval()


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

class NITeacher(Teacher):
    def __init__(self, solver, generator, gen_opt, img_shape, iters, class_idx, deep_inv_params,device='cuda',train=True, config=None):
        super().__init__(solver, generator, gen_opt, img_shape, iters, class_idx, deep_inv_params, train, config)
        self.device=device
        self.loader=None

    def sample(self, size, device, return_scores=False):
        # train if first time
        if self.first_time:
            self.first_time=False
            self.transform_from_cifar=get_transform('cifar100','train')
        
        idx= np.random.randchoice(len(self.inv_labels), size)
        xs_i=self.inv_labels[idx]
        ys_i=self.inv_labels_y[idx]

        xs=[]
        for x in xs_i:
            x_tmp=PIL.Image.fromarray(x)
            xs.append(self.transform_from_cifar(x_tmp))
        x_i=torch.stack(xs,dim=0).to(device)
        y=torch.from_numpy(ys_i).long().to(device)
        # make sure solver is eval mode
        self.solver.eval()
        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x_i)
        y_hat = y_hat[:, self.class_idx]

        return (x_i, y, y_hat) if return_scores else (x_i, y)

    def get_images(self, num_generate_images=2000, epochs=2000, idx=-1):
        torch.cuda.empty_cache()
        self.solver.eval()
        feature_decoder=Feature_Decoder(3)

        bn_reg_scale=10
        g_lr=0.001
        d_lr=0.0005
        a_lr=0.05
        var_scale=0.001 
        l2_coeff=0.00001
        print("num generate images",num_generate_images)
        print("class idx",self.class_idx)

        num_classes=len(self.class_idx)
        minimum_per_class=num_generate_images//num_classes
        num_cls_targets=np.zeros(num_classes)

        best_inputs_list=[]
        best_targets_list = []
        batch_size=256
        best_cost=1e6
        while np.count_nonzero(num_cls_targets>=minimum_per_class)<num_classes:

            self.generator.reset(num_classes)
            self.generator=self.generator.to(self.device)
            self.generator.train()
            feature_decoder.reset()
            feature_decoder.to(self.device)
            feature_decoder.train()
            optimizer_g = optim.Adam(self.generator.parameters(), lr=g_lr)
            optimizer_f = torch.optim.Adam(feature_decoder.parameters(), lr=d_lr)

            # Learnable Scale Parameter
            alpha = torch.empty((batch_size,3,1,1), requires_grad=True, device=self.device)
            torch.nn.init.normal_(alpha, 5.0, 1)
            optimizer_alpha = torch.optim.Adam([alpha], lr=a_lr)

            # set up criteria for optimization
            criterion = nn.CrossEntropyLoss()
            optimizer_g.state = collections.defaultdict(dict)
            optimizer_f.state = collections.defaultdict(dict)  # Reset state of optimizer
            optimizer_alpha.state = collections.defaultdict(dict)

            np_targets=np.random.choice(num_classes,batch_size)
            targets = torch.LongTensor(np_targets).to(self.device)
            onehot_targets=F.one_hot(targets,num_classes).float().to(self.device)
            z = torch.randn((batch_size, 1000)).to(self.device)
            z = torch.cat((z,onehot_targets), dim = 1)
            
            loss_r_feature_layers = []
            count = 0
            for module in self.solver.modules():
                if isinstance(module, nn.BatchNorm2d):
                    loss_r_feature_layers.append(NaturalInversionFeatureHook(module, 0))
        
            lim_0, lim_1 = 2, 2
            best_cost=1e6
            for epoch in range(epochs):
                inputs_jit = self.generator(z)
            
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
                    _, features = self.solver.forward(input_for_f,feature=True)
            
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
                outputs, features = self.solver.forward(inputs_jit,feature=True)
            
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

                if (epoch+1) % int(epochs/5)==0:
                    print("\rIt {:4d}\tLosses: total: {:7.3f},\ttarget: {:5.3f} \tR_feature_loss unscaled:\t {:6.3f}".format(epoch, loss.item(),loss_target,loss_distr.item()),end='')
                    nchs = inputs_jit.shape[1]

                    save_pth = os.path.join('inversion_images', 'task{}'.format(idx))
                    if not os.path.exists(save_pth):
                        os.makedirs(save_pth)
        
                    vutils.save_image(inputs_jit.data.clone(),os.path.join(save_pth,'generator_{}.png'.format(str(epoch//100).zfill(2))),normalize=True, scale_each=True, nrow=10)
                    vutils.save_image(inputs_for_save,os.path.join(save_pth,'save_{}.png'.format(str(epoch//100).zfill(2))),normalize=True, scale_each=True, nrow=10)

                if best_cost > loss.item():
                    best_cost = loss.item()
                    with torch.no_grad():
                        self.generator.eval()
                        best_inputs = self.generator(z)
                        _, features = self.solver.forward(best_inputs,feature=True)
                        best_inputs, addition = feature_decoder(best_inputs, features)
                        best_inputs *= alpha
                        self.generator.train()
            
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()
                optimizer_alpha.zero_grad()

                # backward pass
                loss.backward()

                optimizer_g.step()
                optimizer_f.step()
                optimizer_alpha.step()
            print("")
            best_inputs_list.append(best_inputs.cpu().detach().numpy())
            best_targets_list.append(targets.cpu().detach().numpy())
            for mod in loss_r_feature_layers:
                mod.close()
            for target in targets.cpu().detach().numpy():
                num_cls_targets[target]+=1
            print("Goal num_cls_targets : {} / Current [{:.2f}%] {}".format(minimum_per_class, 100.*min(num_cls_targets)/minimum_per_class,num_cls_targets))
            optimizer_alpha.zero_grad(set_to_none=True)
            optimizer_f.zero_grad(set_to_none=True)
            optimizer_g.zero_grad(set_to_none=True)
            # del generator
            # del feature_decoder
        inv_images= best_inputs_list
        inv_labels= best_targets_list
        datas, labels = [], []
        for lbl, img in zip(inv_labels, inv_images):
            for l, i in zip(lbl, img):

                data = np.transpose(np.array(i), (1, 2, 0))
                datas.append(data.astype(np.uint8))
                labels.append(np.array([l]))

        # list (np.array(32,32,3),...) -> stack (bsz,32,32,3)
        inv_images = np.stack(datas, axis=0)
        inv_labels = np.concatenate(labels, axis=0).reshape(-1)
        # indexing
        inv_filtered_images = []
        inv_filtered_labels = []
        size_of_exemplar = num_generate_images//idx
        for cls_idx in range(0, idx):
            # size of exemplar is from prev task_id.
            inv_filtered_images.append(
                inv_images[inv_labels == cls_idx][:size_of_exemplar])
            inv_filtered_labels.append(
                inv_labels[inv_labels == cls_idx][:size_of_exemplar])
        self.inv_images = np.concatenate(inv_filtered_images, axis=0)
        self.inv_labels = np.concatenate(
            inv_filtered_labels, axis=0).reshape(-1)
        print("Length of inv_images: {}".format(len(self.inv_images)))

            
class ImageDatasetFromData(Dataset):
    def __init__(self, images, labels=None, transform=None,target_transform=None, return_idx=False):
        self.X = images
        self.y = labels
        self.transform = transform
        self.target_transform = target_transform
        self.return_idx=return_idx
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data=Image.fromarray(self.X[i])
        
        if self.transform:
            data = self.transform(data)
        

        if self.y is not None:
            target=torch.tensor(self.y[i],dtype=torch.long)
            if self.target_transform:
                target = self.target_transform(target)
            if self.return_idx:
                return data, target, i
            return (data, target)
        else:
            if self.return_idx:
                return data, i
            return data

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, gram_matrix_weight, layer_weight):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.target = None
        self.gram_matrix_weight = gram_matrix_weight
        self.layer_weight = layer_weight

    def hook_fn(self, module, input, output):

        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False) + 1e-8
        r_feature = torch.log(var**(0.5) / (module.running_var.data.type(var.type()) + 1e-8)**(0.5)).mean() - 0.5 * (1.0 - (module.running_var.data.type(var.type()) + 1e-8 + (module.running_mean.data.type(var.type())-mean)**2)/var).mean()

        self.r_feature = r_feature

            
    def close(self):
        self.hook.remove()

class Gaussiansmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussiansmoothing, self).__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).cuda()

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
