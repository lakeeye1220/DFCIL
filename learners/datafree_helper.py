import torch
from torch import nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import torchvision
import os
import wandb
import matplotlib.pyplot as plt
from utils.metric import AverageMeter
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

    def __init__(self, solver, generator, gen_opt, img_shape, iters, class_idx, deep_inv_params, train = True, task_num=-1, config=None):

        super().__init__()
        self.solver = solver #classifier
        self.generator = generator #generator
        self.gen_opt = gen_opt
        self.solver.eval()
        self.generator.eval()
        self.img_shape = img_shape
        self.iters = iters
        self.config = config
        self.task_num = task_num

        # hyperparameters for image synthesis
        self.di_lr = deep_inv_params[0]
        self.r_feature_weight = deep_inv_params[1]
        self.di_var_scale = deep_inv_params[2]
        self.content_temp = deep_inv_params[3]
        self.content_weight = deep_inv_params[4]
        

        # get class keys
        self.class_idx = list(class_idx)
        self.num_k = len(self.class_idx)

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
            if self.config['cgan']:
                x_i, y_i = self.generator.sample(size)
            else:
                x_i = self.generator.sample(size)

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x_i)
        y_hat = y_hat[:, self.class_idx]

        # get predicted class-labels (indexed according to each class' position in [self.class_idx]!)
        if self.config['cgan']:
            y=y_i
        else:
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
        save_images=[]
        
        # cgan setup
        if 'disc' in self.config['cgan']:
            self.generator.update_num_classes(self.num_k)
            if 'CIFAR' in self.config['dataset']:
                n_dim=64
            else:
                n_dim=512
            self.discriminator=nn.Sequential(
                nn.Linear(n_dim, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1),
                nn.Sigmoid())
            self.discriminator.cuda()
            self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.config['disc_lr'], betas=(0.5, 0.999))
            try:
                train_iter = iter(self.train_dataloader)
            except:
                raise NotImplementedError("No train dataloader provided for cgan")
            if self.config['wandb']:
                wandb.watch(self.discriminator, log='all', idx=2)

        def plot_save(lists, name):
            plt.plot(lists)
            plt.ylabel('{}task_'+name)
            plt.xlabel('step')
            plt.savefig(os.path.join(self.config['model_save_dir'],'{}.png'.format(name)))
            plt.close()
            plt.clf()
            plt.cla()
            if self.config['wandb']: # plot with wandb
                wandb.Image(os.path.join(self.config['model_save_dir'],'{}.png'.format(name)), caption='{}task_'+name)
        
        # training generator
        if self.config['cgan'] is None:
            # lists for storing losses
            loss_list = []
            cnt_loss_list=[]
            bnc_loss_list=[]
            distrs_loss_list=[]
            var_loss_list=[]
            for epoch in tqdm(range(epochs)):

                # sample from generator
                inputs = self.generator.sample(bs)

                # forward with images
                self.gen_opt.zero_grad()
                self.solver.zero_grad()

                # content
                outputs = self.solver(inputs)[:,:self.num_k]
                cnt_loss = self.criterion(outputs / self.content_temp, torch.argmax(outputs, dim=1)) * self.content_weight
                with torch.no_grad():
                    ce_loss = self.criterion(outputs,torch.argmax(outputs,dim=1))

                # class balance
                softmax_o_T = F.softmax(outputs, dim = 1).mean(dim = 0)
                bnc_loss= (1.0 + (softmax_o_T * torch.log(softmax_o_T) / math.log(self.num_k)).sum())

                # R_feature loss
                loss_distrs=0
                for mod in self.loss_r_feature_layers: 
                    loss_distr = mod.r_feature * self.r_feature_weight / len(self.loss_r_feature_layers)
                    if len(self.config['gpuid']) > 1:
                        loss_distr = loss_distr.to(device=torch.device('cuda:'+str(self.config['gpuid'][0])))
                    loss_distrs+=loss_distr

                # image prior
                inputs_smooth = self.smoothing(F.pad(inputs, (2, 2, 2, 2), mode='reflect'))
                var_loss = self.mse_loss(inputs, inputs_smooth).mean()
                var_loss = self.di_var_scale * var_loss

                # backward pass - update the generator
                loss=(cnt_loss + bnc_loss + loss_distrs + var_loss)
                loss.backward()
                self.gen_opt.step()
                if epoch % 1000 == 0:
                    print("Epoch: %d, Loss: %.3e, cnt_loss: %.3e (CE: %.3e), bnc_loss: %.3e, loss_distrs: %.3e, var_loss: %.3e" % (epoch, loss, cnt_loss,ce_loss, bnc_loss, loss_distrs, var_loss))
                    save_images.append(inputs.detach().cpu())
                    if self.config['wandb']:
                        # save images (var: inputs)
                        table_data=[]
                        for i in range(len(inputs)):
                            table_data.append([i, wandb.Image(inputs[i]),outputs[i],torch.argmax(outputs[i])])
                        wandb.log({"{}task {}iter images".format(self.task_num,epoch): wandb.Table(data=table_data, columns=["idx", "image", "logits", "label"])},commit=False)

                loss_list.append(loss.item())
                cnt_loss_list.append(cnt_loss.item())
                bnc_loss_list.append(bnc_loss.item())
                distrs_loss_list.append(loss_distrs.item())
                var_loss_list.append(var_loss.item())

            plot_save(loss_list, 'loss')
            plot_save(cnt_loss_list, 'cnt_loss')
            plot_save(bnc_loss_list, 'bnc_loss')
            plot_save(distrs_loss_list, 'distrs_loss')
            plot_save(var_loss_list, 'var_loss')

        elif 'disc' in self.config['cgan']:
            # loss list
            loss_list = []
            cnt_loss_list=[]
            # bnc_loss_list=[]
            distrs_loss_list=[]
            var_loss_list=[]
            disc_loss_list=[]

            for epoch in tqdm(range(epochs)):

                # train generator
                self.gen_opt.zero_grad()
                inputs, y_i = self.generator.sample(bs)
                out_pen = self.solver(inputs,pen=True)
                outputs = self.solver.last(out_pen)[:,:self.num_k]

                # discriminator loss measures
                y_hat = self.discriminator(out_pen)
                g_loss= F.mse_loss(y_hat, torch.ones_like(y_hat).cuda())
                # content loss
                cnt_loss = self.criterion(outputs / self.content_temp, y_i) * self.content_weight
                with torch.no_grad():
                    ce_loss=self.criterion(outputs, y_i).detach().clone()
                # Statstics alignment
                loss_distrs=0
                for mod in self.loss_r_feature_layers: 
                    loss_distr = mod.r_feature * self.r_feature_weight / len(self.loss_r_feature_layers)
                    if len(self.config['gpuid']) > 1:
                        loss_distr = loss_distr.to(device=torch.device('cuda:'+str(self.config['gpuid'][0])))
                    loss_distrs+=loss_distr

                # image prior
                inputs_smooth = self.smoothing(F.pad(inputs, (2, 2, 2, 2), mode='reflect'))
                var_loss = self.mse_loss(inputs, inputs_smooth).mean()
                var_loss = self.di_var_scale * var_loss
                loss=(cnt_loss + g_loss + loss_distrs + var_loss)
                loss.backward(retain_graph=True)
                self.gen_opt.step()

                # train discriminator
                self.discriminator_opt.zero_grad()
                try:
                    (x, y, task) = next(train_iter)
                except:
                    train_iter = iter(self.train_dataloader)
                    (x, y, task) = next(train_iter)

                x = x.cuda()
                y = y.cuda()
                out_real_pen = self.solver(x,pen=True)
                y_real_hat = self.discriminator(out_real_pen)
                d_real_loss = F.mse_loss(y_real_hat, torch.ones_like(y_real_hat).cuda())
                out_fake_pen = self.solver(inputs.detach().clone(),pen=True)
                y_fake_hat = self.discriminator(out_fake_pen)
                d_fake_loss = F.mse_loss(y_fake_hat, torch.zeros_like(y_fake_hat).cuda())
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.discriminator_opt.step()
                self.solver.zero_grad()
                if epoch % 1000 == 0:
                    print("Epoch: %d, g_loss: %.3e, cnt_loss: %.3e (CE: %.3e), d_loss: %.3e, loss_distrs: %.3e, var_loss: %.3e" % (epoch, loss, cnt_loss,ce_loss, d_loss, loss_distrs, var_loss))
                    save_images.append(inputs.detach().cpu())
                    if self.config['wandb']:
                        # save images (var: inputs)
                        table_data=[]
                        for i in range(len(inputs)):
                            table_data.append([i, wandb.Image(inputs[i]),outputs[i],torch.argmax(outputs[i])])
                        wandb.log({"{}task {}iter images".format(self.task_num,epoch): wandb.Table(data=table_data, columns=["idx", "image", "logits", "label"])},commit=False)
                loss_list.append(loss.item())
                cnt_loss_list.append(cnt_loss.item())
                # bnc_loss_list.append(bnc_loss.item())
                disc_loss_list.append(d_loss.item())
                distrs_loss_list.append(loss_distrs.item())
                var_loss_list.append(var_loss.item())
            plot_save(loss_list, 'generator_loss')
            plot_save(cnt_loss_list, 'cnt_loss')
            # plot_save(bnc_loss_list, 'bnc_loss')
            plot_save(distrs_loss_list, 'distrs_loss')
            plot_save(var_loss_list, 'var_loss')
            plot_save(disc_loss_list, 'discriminator_loss')
        # save images
        save_images = torch.cat(save_images, dim=0)
        grid=torchvision.utils.make_grid(save_images, nrow=bs, padding=1, normalize=True, range=None, scale_each=False, pad_value=0)
        torchvision.utils.save_image(grid, os.path.join(self.config['model_save_dir'],'iter_generated_images.png'.format(idx)), nrow=1, padding=0, normalize=False, range=None, scale_each=False, pad_value=0)

        # clear cuda cache
        torch.cuda.empty_cache()
        self.generator.eval()
        with torch.no_grad():
            if self.config['cgan']:                
                samples, y_i = self.generator.sample(self.num_k*10)
            else:
                samples = self.generator.sample(self.num_k*10)
            grid=torchvision.utils.make_grid(samples, nrow=self.num_k, padding=1, normalize=True, range=None, scale_each=False, pad_value=0)
            torchvision.utils.save_image(grid, os.path.join(self.config['model_save_dir'],'generated_images.png'.format(idx)), nrow=1, padding=0, normalize=False, range=None, scale_each=False, pad_value=0)
            if self.config['wandb']:
                wandb.log({"{}task generated images".format(self.task_num): [wandb.Image(grid)]},commit=True)

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
