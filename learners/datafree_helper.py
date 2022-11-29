import torch
from torch import nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import torchvision
import os
import wandb
import matplotlib.pyplot as plt
from learners.wgan.utils import cal_grad_penalty, d_wasserstein, g_wasserstein, sample_normal
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
        if self.config['cgan'] is None or 'wgan' not in self.config['cgan']:
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
                if 'disc' in self.config['cgan']:
                    x_i, y_i = self.generator.sample(size)
                elif 'latent' == self.config['cgan']:
                    x_i, y_i, z = self.generator.sample(size, self.solver)
                elif 'wgan' == self.config['cgan']:
                    y_fake = torch.randint(low=0, high=self.num_k, size=(size, ), dtype=torch.long, device='cuda')
                    zs = sample_normal(batch_size=size, z_dim=128, truncation_factor=-1, device='cuda')
                    xs = self.generator(zs,y_fake)
                    ys= self.solver(xs)[:,:self.num_k]
                    if self.config['gan_target'] in ['hard','soft']:
                        y_i=torch.argmax(ys,dim=1)
                        x_i=xs
                    else: # pseudo
                        condition=torch.nonzero(torch.softmax(ys ,dim=1).max(dim=1)[0]>0.8)
                        y_i=[ys.index_select(0,condition)]
                        x_i=[xs.index_select(0,condition)]
                        condition.tolist()
                        while len(condition)<y_fake.shape[0]:
                            y_fake = torch.randint(low=0, high=self.num_k, size=(size, ), dtype=torch.long, device='cuda')
                            zs = sample_normal(batch_size=size, z_dim=128, truncation_factor=-1, device='cuda')
                            xs = self.generator(zs,y_fake)
                            ys= self.solver(xs)[:,:self.num_k]
                            condition_list=torch.nonzero(torch.softmax(ys ,dim=1).max(dim=1)[0]>0.8).tolist()
                            condition+=condition_list
                            x_i.append(xs.index_select(0,condition_list))
                            y_i.append(ys.index_select(0,condition_list))
                        x_i=torch.cat(x_i,dim=0)
                        y_i=torch.cat(y_i,dim=0)
                        x_i=x_i[:size]
                        y_i=torch.argmax(y_i[:size],dim=1)

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
        print("Start learning teacher")
        # print(self.generator.parameters())
        # print(self.gen_opt.param_groups)

        # clear cuda cache
        torch.cuda.empty_cache()

        self.generator.train()
        save_images=[]
        
        # cgan setup
        if self.config['cgan'] and 'disc' in self.config['cgan']:
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
        elif self.config['cgan'] and 'wgan' == self.config['cgan']:
            from learners.wgan.resnet import Discriminator
            self.discriminator = Discriminator(32,64,False,False,["N/A"],"W/O","W/O","N/A",False,self.num_k,"ortho","N/A",False)
            self.discriminator.cuda()
            beta1=0.5
            beta2=0.999
            betas_g = [beta1, beta2]
            eps_ = 1e-6
            self.discriminator_opt = torch.optim.Adam(params=self.discriminator.parameters(),
                                                            lr=0.0002,
                                                            betas=betas_g,
                                                            weight_decay=0.0,
                                                            eps=eps_)

        def plot_save(lists, name):
            plt.plot(lists)
            plt.ylabel('{}task_'.format(self.task_num)+name)
            plt.xlabel('step')
            plt.savefig(os.path.join(self.config['model_save_dir'],'{}.png'.format(name)))
            plt.close()
            plt.clf()
            plt.cla()
            if self.config['wandb']: # plot with wandb
                wandb.Image(os.path.join(self.config['model_save_dir'],'{}.png'.format(name)), caption='{}task_'.format(self.task_num)+name)
        
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
                        #wandb.log({"{}task {}iter images".format(self.task_num,epoch): wandb.Table(data=table_data, columns=["idx", "image", "logits", "label"])},commit=False)
                        del table_data

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
                    #if self.config['wandb']:
                    #    # save images (var: inputs)
                    #    table_data=[]
                    #    for i in range(len(inputs)):
                    #        table_data.append([i, wandb.Image(inputs[i]),outputs[i],torch.argmax(outputs[i])])
                    #    #wandb.log({"{}task {}iter images".format(self.task_num, epoch): wandb.Table(data=table_data, columns=["idx", "image", "logits", "label"])},commit=True)
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

        elif self.config['cgan']=='latent':
            # loss list
            loss_list = []
            cnt_loss_list=[]
            mse_loss_list=[]
            uni_loss_list=[]
            distrs_loss_list=[]
            
            for epoch in tqdm(range(epochs)):
                # train generator
                inputs, y_i, z = self.generator.sample(bs, self.solver)
                out_pen = self.solver(inputs,pen=True)
                outputs = self.solver.last(out_pen)[:,:self.num_k]
                self.gen_opt.zero_grad()
                self.solver.zero_grad()

                # mse with out_pen and z
                loss_mse = self.mse_loss(out_pen, z[:,:out_pen.shape[1]]).mean()

                with torch.no_grad():
                    ce_loss=self.criterion(outputs, y_i).detach().clone()


                # content loss
                cnt_loss = self.criterion(outputs / self.content_temp, y_i) * self.content_weight

                # uniform loss
                # softmax_o_T = F.softmax(outputs, dim = 1).mean(dim = 0)
                # uni_loss= (1.0 + (softmax_o_T * torch.log(softmax_o_T) / math.log(self.num_k)).sum())
                uni_loss=torch.tensor(0,dtype=torch.float).view(-1).to(cnt_loss.device)

                # Statstics alignment
                loss_distrs=0
                for mod in self.loss_r_feature_layers: 
                    loss_distr = mod.r_feature * self.r_feature_weight / len(self.loss_r_feature_layers)
                    if len(self.config['gpuid']) > 1:
                        loss_distr = loss_distr.to(device=torch.device('cuda:'+str(self.config['gpuid'][0])))
                    loss_distrs+=loss_distr
                loss= (cnt_loss+loss_mse+loss_distrs+uni_loss)

                # # image prior
                # inputs_smooth = self.smoothing(F.pad(inputs, (2, 2, 2, 2), mode='reflect'))
                # loss_var = self.mse_loss(inputs, inputs_smooth).mean()
                # loss_var = self.di_var_scale * loss_var
                # loss=(cnt_loss + loss_distrs + loss_var)
                self.gen_opt.zero_grad()
                loss.backward()
                self.gen_opt.step()
                if epoch % 1000 == 0:
                    print(f"Epoch: {epoch:5d}, Loss: {loss:.3e} CNT_loss: {cnt_loss:.3e} (CE: {ce_loss:.3e}) MSE_loss: {loss_mse:.3e} loss_distrs: {loss_distrs:.3e}")
                    save_images.append(inputs.detach().cpu())
                loss_list.append(loss.item())
                cnt_loss_list.append(cnt_loss.item())
                mse_loss_list.append(loss_mse.item())
                uni_loss_list.append(uni_loss.item())
                distrs_loss_list.append(loss_distrs.item())
            plot_save(loss_list, 'generator_loss')
            plot_save(cnt_loss_list, 'cnt_loss')
            plot_save(mse_loss_list, 'mse_loss')
            plot_save(uni_loss_list, 'uni_loss')
            plot_save(distrs_loss_list, 'distrs_loss')
            
        elif self.config['cgan']=='wgan':
            self.generator.train()
            self.discriminator.train()
            wgan_bsz=64
            for epoch in tqdm(range(epochs)):
                # get real image
                # train discriminator
                for p in self.discriminator.parameters():
                    p.requires_grad = True
                for p in self.generator.parameters():
                    p.requires_grad = False
                for step_index in range(5): # d_updates_per_step
                    try:
                        (real_images, real_labels, task) = next(train_iter)
                    except:
                        train_iter = iter(self.train_dataloader)
                        (real_images, real_labels, task) = next(train_iter)
                    real_images = real_images.cuda()
                    real_labels = real_labels.cuda()
                    # train discriminator
                    self.discriminator_opt.zero_grad()
                    zs = sample_normal(batch_size=wgan_bsz, z_dim=128, truncation_factor=-1, device='cuda')
                    y_fake = torch.randint(low=0, high=self.num_k, size=(wgan_bsz, ), dtype=torch.long, device='cuda')
                    fake_images = self.generator(zs,y_fake, eval=False)
                    real_dict = self.discriminator(real_images, real_labels)
                    with torch.no_grad():
                        fake_labels= torch.argmax(self.solver(fake_images)[:,:self.num_k],dim=1)
                    fake_dict = self.discriminator(fake_images, fake_labels, adc_fake=False)
                    dis_loss = d_wasserstein(real_dict["adv_output"], fake_dict["adv_output"])

                    gp_loss = cal_grad_penalty(real_images=real_images,
                                                        real_labels=real_labels,
                                                        fake_images=fake_images,
                                                        discriminator=self.discriminator,
                                                        device='cuda')
                    dis_acml_loss=dis_loss + 10.0 * gp_loss
                    dis_acml_loss.backward()
                    self.discriminator_opt.step()
                
                for p in self.discriminator.parameters():
                    p.requires_grad = False
                for p in self.generator.parameters():
                    p.requires_grad = True
                # train generator
                self.generator.train()
                self.gen_opt.zero_grad()
                
                zs = sample_normal(batch_size=wgan_bsz, z_dim=128, truncation_factor=-1, device='cuda')
                y_fake = torch.randint(low=0, high=self.num_k, size=(wgan_bsz, ), dtype=torch.long, device='cuda')
                fake_images = self.generator(zs,y_fake)
                fake_labels= torch.argmax(self.solver(fake_images)[:,:self.num_k],dim=1)
                fake_dict = self.discriminator(fake_images, fake_labels)
                gen_acml_loss = g_wasserstein(fake_dict["adv_output"])
                gen_acml_loss.backward()
                self.gen_opt.step()
                self.discriminator_opt.zero_grad()
                if epoch % 1000 == 0:
                    print(f"Epoch: {epoch:5d}, G Loss: {gen_acml_loss:.3e} D gp_loss:{gp_loss:.3e} D Loss: {dis_loss:.3e}")
                    save_images.append(fake_images.detach().cpu())
                

        # save images
        save_images = torch.cat(save_images, dim=0)
        grid=torchvision.utils.make_grid(save_images, nrow=bs, padding=1, normalize=True, range=None, scale_each=False, pad_value=0)
        torchvision.utils.save_image(grid, os.path.join(self.config['model_save_dir'],'iter_generated_images.png'.format(idx)), nrow=1, padding=0, normalize=False, range=None, scale_each=False, pad_value=0)

        # clear cuda cache
        torch.cuda.empty_cache()
        self.generator.eval()
        if self.config['cgan'] is not None:
            self.discriminator.eval()
        self.gen_opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            if self.config['cgan']:
                if 'disc' in self.config['cgan']:                
                    samples, y_i = self.generator.sample(self.num_k*10)
                elif self.config['cgan']=='latent':
                    samples, y_i, z = self.generator.sample(self.num_k*10, self.solver)
                elif self.config['cgan']=='wgan':
                    zs = sample_normal(batch_size=self.num_k*10, z_dim=128, truncation_factor=-1, device='cuda')
                    y_fake = torch.randint(low=0, high=self.num_k, size=(self.num_k*10, ), dtype=torch.long, device='cuda')
                    samples = self.generator(zs,y_fake)
                    y_i= torch.argmax(self.solver(samples)[:,:self.num_k],dim=1)
                    
            else:
                samples = self.generator.sample(self.num_k*10)
                logits = self.solver(samples)
                logits = logits[:,:self.num_k]
                y_i = logits.argmax(dim=1)
            argsorted_logits=torch.argsort(y_i, dim=0, descending=True)
            samples=samples[argsorted_logits]
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
