import torch
from torch import nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import torchvision.utils as vutils
import os
import collections
import random

class Teacher(nn.Module):
    def __init__(self, solver,previous_teacher, generator,  gen_opt, proto, disc_opt, reparam_opt, m,v,img_shape, iters, class_idx, deep_inv_params, train = True, config=None):
        super().__init__()
        self.solver = solver #classifier
        self.previous_teacher = previous_teacher
        self.generator = generator #generator
        self.gen_opt = gen_opt
        self.disc_opt = disc_opt
        self.reparam_opt = reparam_opt
        self.m = m
        self.v = v
        self.solver.eval()
        self.generator.eval()
        self.img_shape = img_shape
        self.iters = iters
        self.config = config

        # hyperparameters for image synthesis
        self.di_lr = deep_inv_params[0]
        self.r_feature_weight = deep_inv_params[1]
        self.di_var_scale = deep_inv_params[2]
        self.content_temp = deep_inv_params[3]
        self.content_weight = deep_inv_params[4]

                
        # adversarial loss
        self.real_label = 1.
        self.fake_label = 0.
        # get class keys
        self.class_idx = list(class_idx)
        self.num_k = len(self.class_idx)

        self.first_time = True

        # set up criteria for optimization
        self.criterion = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.bceloss = nn.BCELoss()
        self.mse_loss = nn.MSELoss(reduction="none").cuda()
        self.smoothing = Gaussiansmoothing(3,5,1).cuda()

        # Create hooks for feature statistics catching
        loss_r_feature_layers = []
        self.proto = proto
        
        if self.config['cgan']:
            for module in self.solver.modules():
                #if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                if isinstance(module,nn.BatchNorm2d):
                    loss_r_feature_layers.append(original_DeepInversionFeatureHook(module))
            self.loss_r_feature_layers = loss_r_feature_layers

        else:
            for module in self.solver.modules():
                #if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                if isinstance(module,nn.BatchNorm2d):
                    loss_r_feature_layers.append(DeepInversionFeatureHook(module, 0, self.r_feature_weight))
            self.loss_r_feature_layers = loss_r_feature_layers


    def sample(self, size, device, return_scores=False):
        # make sure solver is eval mode
        self.solver.eval()
        # train if first time
        self.generator.train()
        if self.first_time:
            if self.config['cgan']:
                self.original_get_images(bs=size, epochs=self.iters, idx=-1)
            if self.config['prototype']:
                self.get_images(bs=size, epochs=self.iters, idx=-1)
            if self.config['gen_v2']:
                self.get_images(bs=size,epochs=self.iters,)
            self.first_time = False
        # sample images
        self.generator.eval()
        with torch.no_grad():
            if self.config['cgan']:
                x_i,_ = self.generator.sample(self.config, self.num_k, size)
            else:
                x_i = self.generator.sample(self.config, self.num_k, size,self.v,self.m)
                
        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x_i)
        #_, y = torch.max(y_hat, dim=1)
        # get predicted logit-scores
        #with torch.no_grad():
        #    y_hat = self.solver.forward(x_i)
        y_hat = y_hat[:, self.class_idx]

        # get predicted class-labels (indexed according to each class' position in [self.class_idx]!)
        _, y = torch.max(y_hat, dim=1)
        #print("y label datafee_helper.py 76: ",y)

        #return (x_i, y.cuda(), y_hat) if return_scores else (x_i, y)
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

    def off_diagonal(self,x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


    def generate_scores_pen(self, x):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)

        return y_hat

    def get_images(self, bs=256, epochs=1000, idx=-1):
        # synthesize old samples using model inversion
        # clear cuda cache
        torch.cuda.empty_cache()
        if self.config['reparam']:
            self.reparam_opt.state = collections.defaultdict(dict)

        self.generator.train()

        for epoch in tqdm(range(epochs)):
            # sample from generator
            #if self.config['cgan']==False:
            #    inputs = self.generator.sample(self.config,self.num_k,bs)
                #inputs,_,_ = self.generator.sample(self.config,self.num_k,bs)
            if self.previous_teacher is None:
                if self.config['cgan']:
                    inputs,fake_targets = self.generator.sample(self.config,self.num_k,bs)
                else:
                    inputs = self.generator.sample(self.config,self.num_k,bs,self.v,self.m)
                    #print("first time : ",inputs.shape)
            else:
                if self.config['cgan']:
                    inputs,fake_targets = self.generator.sample(self.config,self.num_k,bs)
                elif self.config['prototype']:
                    inputs = self.generator.sample(self.config,self.num_k,bs*2,self.v,self.m)
                    #print("second upper time : ",inputs.shape)
                else:
                    inputs = self.generator.sample(self.config,self.num_k,bs,self.v,self.m)
                #print("inputs : ",inputs.shape)
            

            # forward with images
            self.gen_opt.zero_grad()
            if self.config['reparam']:
                self.reparam_opt.zero_grad()
            self.solver.zero_grad()

            # data conetent loss
            if self.config['gen_v2']:
                #_,gm = self.solver.forward(inputs,middle=True)[:,:self.num_k]
                _,gm = self.solver.forward(inputs,middle=True)
                g3,g2,g1=gm
                inputs = self.generator.sample(self.config,self.num_k,bs,None,None,g1,g2,g3)

            outputs = self.solver(inputs)[:,:self.num_k]
            loss = self.criterion(outputs / self.content_temp, torch.argmax(outputs, dim=1)) * self.content_weight
            loss_cnt = loss.detach()

                
            softmax_o_T = F.softmax(outputs, dim = 1).mean(dim = 0)
            loss_ie= (1.0 + (softmax_o_T * torch.log(softmax_o_T) / math.log(self.num_k)).sum())
            loss = loss+ loss_ie

            if self.config['prototype']:
                proto_logits = random.choice(self.proto)
                #M = 0.5*(outputs+proto_logits[:,:self.num_k].detach())
                #print("outputs.shape ",outputs.shape)
                #print("proto type logit : ",proto_logits.shape)
                #loss_recon = 0.5*self.kl_loss(torch.log(outputs),M)+0.5*self.kl_loss(proto_logits[:,:self.num_k],M)
                loss_recon = self.kl_loss(torch.log(F.softmax(outputs,dim=1)),F.softmax(proto_logits[:,:self.num_k],dim=1).detach())
                loss = loss + loss_recon
                if epoch % 1000==0:
                    print("reconstruction loss", loss_recon.item())

            # Statstics alignment
            for mod in self.loss_r_feature_layers: 
                loss_distr = mod.r_feature * self.r_feature_weight / len(self.loss_r_feature_layers)
                if len(self.config['gpuid']) > 1:
                    loss_distr = loss_distr.to(device=torch.device('cuda:'+str(self.config['gpuid'][0])))
                loss = loss + loss_distr

            # image prior
            inputs_smooth = self.smoothing(F.pad(inputs, (2, 2, 2, 2), mode='reflect'))
            loss_var = self.mse_loss(inputs, inputs_smooth).mean()
            loss = loss + self.di_var_scale * loss_var

            if self.config['adversarial'] and self.previous_teacher:
                #for j in range(5):
                inputs = self.generator.sample(self.config,self.num_k,bs)
                outputs_p = self.previous_teacher.solver(inputs)[:,:self.num_k]
                outputs_c = self.solver(inputs)[:,:self.num_k]
                loss_G = -F.l1_loss(outputs_c,outputs_p.detach())
                loss = loss + loss_G
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f}'.format(
                    epoch, epoch, epochs, 100*float(epoch)/float(epochs), loss_G.item()))
            # backward pass - update the generator

            if self.config['confusion']:
                off_diag_lambda = 0.0051
                #norm_layer = nn.BatchNorm1d(sizes[-1],affine=False)
                outputs_N = F.normalize(outputs,p=1.0,dim=1)
                #print("outputs_N ",outputs_N)
                outputs_T = torch.transpose(outputs_N,0,1)
                sm_matrix = torch.matmul(outputs_N,outputs_T).cuda()
                #print("sm_matrix : ",sm_matrix.shape)
                #I_matrix = torch.eye(outputs_N.shape[0]).cuda()
                #pred_matrix = I_matrix - sm_matrix
                
                on_diag = torch.diagonal(sm_matrix).add_(-1).pow_(2).sum()
                off_diag = self.off_diagonal(sm_matrix).pow_(2).sum()

                loss_matrix = on_diag + off_diag_lambda*off_diag
                loss = loss + loss_matrix
                if epoch % 1000==0:
                    print("confusion loss", loss_matrix.item())

            loss.backward()
            self.gen_opt.step()
            if self.config['reparam']:
                self.reparam_opt.step()
            
            torch.cuda.empty_cache()
            self.generator.eval()

            if self.config['adversarial'] and self.previous_teacher:
                self.solver.train()
                for j in range(5):
                    self.disc_opt.zero_grad()
                    inputs = self.generator.sample(self.config,self.num_k,bs)
                    outputs_p = self.previous_teacher.solver(inputs)[:,:self.num_k]
                    outputs_c = self.solver(inputs)[:,:self.num_k]
                    loss_D = F.l1_loss(outputs_c,outputs_p.detach())
                    loss = loss+loss_D
                    loss_D.backward()
                    self.disc_opt.step()
                    #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tD_Loss: {:.6f}'.format(
                    #    epoch, epoch, epochs, 100*float(epoch)/float(epochs), loss_D.item()))
            
            if epoch % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttotal_Loss: {:.4f} cnt_loss: {:.4f} ie_Loss: {:.4f} bn_loss: {:.4f} smooth_Loss: {:.4f}'.format(
                    epoch, epoch, epochs, 100*float(epoch)/float(epochs), loss.item(), loss_cnt, loss_ie.item(), loss_distr.item(), loss_var.item()))
                #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f}'.format(
                    #epoch, epoch, epochs, 100*float(epoch)/float(epochs), loss_G.item()))
                
        # clear cuda cache
        #gen_image_path=os.path.join(self.log_dir,'gen_images')
        #vutils.save_image(inputs.data.clone(),'./{}/generator_img.png'.format(gen_image_path),normalize=True,scale_each=True,nrow=10)
        #self.generator.eval()

    def original_get_images(self, bs=256, epochs=1000, idx=-1):
        torch.cuda.empty_cache()
        var_scale = 6.0e-3
        l2_coeff = 1.5e-5

        self.generator.train()
        for epoch in tqdm(range(epochs)):
            # forward with images
            self.gen_opt.zero_grad()
            self.solver.zero_grad()
            inputs,fake_targets = self.generator.sample(self.config,self.num_k,bs)
            inputs, fake_targets = inputs.cuda(),fake_targets.cuda()
            #print("fake targets : ",fake_targets)
            outputs = self.solver(inputs)[:,:self.num_k]
            #print("torch.margmax : ",torch.argmax(outputs,dim=1))
            loss = self.criterion(outputs, torch.argmax(outputs, dim=1))

            loss_distr = sum([mod.r_feature for mod in self.loss_r_feature_layers])
            loss = loss + self.r_feature_weight*loss_distr # best for noise before BN

            # apply total variation regularization
            '''
            diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
            diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
            diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
            diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
            loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
            loss = loss + var_scale*loss_var

            loss = loss + l2_coeff * torch.norm(inputs_jit, 2)
            '''
            # image prior
            inputs_smooth = self.smoothing(F.pad(inputs, (2, 2, 2, 2), mode='reflect'))
            loss_var = self.mse_loss(inputs, inputs_smooth).mean()
            loss = loss + self.di_var_scale * loss_var
            
            loss.backward()
            self.gen_opt.step()

        # clear cuda cache
        torch.cuda.empty_cache()
        self.generator.eval()


class DeepInversionFeatureHook():
    #Implementation of the forward hook to track feature statistics and compute a loss on them.
    #Will compute mean and variance, and will use l2 as a loss

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

class original_DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

class Gaussiansmoothing(nn.Module):
    #Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input using a depthwise convolution.
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
        #Apply gaussian filter to input.
        #filtered (torch.Tensor): Filtered output.
        return self.conv(input, weight=self.weight, groups=self.groups)
