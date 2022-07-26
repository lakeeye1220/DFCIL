from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
import models
from utils.metric import AverageMeter, Timer
import numpy as np
from .datafree_helper import Teacher
from .default import NormalNN, weight_reset, accumulate_acc, loss_fn_kd
import copy
from torch.optim import Adam
import matplotlib.pyplot as plt
import os
from learners.cc import CC
class SP(nn.Module):
    def __init__(self,reduction='mean'):
        super(SP,self).__init__()
        self.reduction=reduction

    def forward(self,fm_s,fm_t):
        fm_s = fm_s.view(fm_s.size(0),-1)
        G_s = torch.mm(fm_s,fm_s.t())
        norm_G_s =F.normalize(G_s,p=2,dim=1)

        fm_t = fm_t.view(fm_t.size(0),-1)
        G_t = torch.mm(fm_t,fm_t.t())
        norm_G_t = F.normalize(G_t,p=2,dim=1)
        if self.reduction == 'mean':
            loss = F.mse_loss(norm_G_s,norm_G_t)
        elif self.reduction == 'sum':
            loss = F.mse_loss(norm_G_s,norm_G_t,reduction='sum')
        elif self.reduction=='none':
            loss = F.mse_loss(norm_G_s,norm_G_t,reduction='none')
        else:
            raise NotImplementedError
        return loss

class DeepInversionGenBN(NormalNN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN, self).__init__(learner_config)
        self.inversion_replay = False
        self.previous_teacher = None
        self.dw = self.config['DW']
        self.device = 'cuda' if self.gpu else 'cpu'
        self.power_iters = self.config['power_iters']
        self.deep_inv_params = self.config['deep_inv_params']
        self.kd_criterion = nn.MSELoss(reduction="none")

        # gen parameters
        self.generator = self.create_generator()
        self.generator_optimizer = Adam(params=self.generator.parameters(), lr=self.deep_inv_params[0])
        self.beta = self.config['beta']
        # repeat call for generator network
        if self.gpu:
            self.cuda_gen()
    
        
    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        self.pre_steps()

        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if need_train:
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()

            # data weighting
            self.data_weighting(train_dataset)

            # Evaluate the performance of current task
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)
            num_meter=5
            losses = [AverageMeter() for i in range(num_meter)]
            acc = AverageMeter()
            accg = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            self.save_gen = False
            self.save_gen_later = False
            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch
                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task)  in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x =x.cuda()
                        y = y.cuda()

                    # data replay
                    if self.inversion_replay:
                        x_replay, y_replay, y_replay_hat = self.sample(self.previous_teacher, len(x), self.device)
                    else:
                        x_replay = None
                        y_replay = None
                    #if middle KD
                    #if self.middle and self.previous_teacher.solver is not None:
                        #print("previous teacher : ",self.previous_teacher)
                        #logits_middle,out1_m,out2_m,out3_m = self.model.forward(x, middle=True)
                        #logits_prev_middle,out1_pm, out2_pm, out3_pm = self.previous_teacher.solver.forward(x,middle=True)
                        #print("out1_m shape : ", out1_m.shape)
                        #print("out1_mp shape ",out1_pm.shape)

                        #print("out1_m ",out1_m)
                        #print("out1_mp : ",out1_pm)


                    if self.inversion_replay:
                        y_hat = self.previous_teacher.generate_scores(x, allowed_predictions=np.arange(self.last_valid_out_dim))
                        _, y_hat_com = self.combine_data(((x, y_hat),(x_replay, y_replay_hat)))
                    else:
                        y_hat_com = None

                    # combine inputs and generated samples for classification
                    if self.inversion_replay:
                        x_com, y_com = self.combine_data(((x, y),(x_replay, y_replay)))
                    else:
                        x_com, y_com = x, y

                    # sd data weighting (NOT online learning compatible)
                    if self.dw:
                        dw_cls = self.dw_k[y_com.long()]
                    else:
                        dw_cls = None

                    # model update
                    if self.inversion_replay:

                    #loss, loss_class, loss_kd, loss_middle, output= self.update_model(x, y,x_replay,y_replay, x_com, y_com, y_hat_com, dw_force = dw_cls, kd_index = np.arange(len(x), len(x_com)))
                        loss,loss_class,loss_kd,loss_middle,loss_balancing,output = self.update_model(x,y,x_replay,y_replay,x_com,y_com,y_hat_com, dw_force=dw_cls, kd_index = np.arange(len(x), len(x_com)))

                        #loss,loss_class,loss_hardKD,loss_middle,loss_balancing,output = self.update_model(x,y,x_replay,y_replay,x_com,y_com,y_hat_com, dw_force=dw_cls, kd_index = np.arange(len(x), len(x_com)))
                    else:
                        loss, loss_class, loss_kd, loss_middle,loss_balancing, output= self.update_model(x, y,None, None, x_com, y_com, y_hat_com, dw_force = dw_cls, kd_index = np.arange(len(x), len(x_com)))

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y_com = y_com.detach()
                    accumulate_acc(output[:self.batch_size], y_com[:self.batch_size], task, acc, topk=(self.top_k,))
                    if self.inversion_replay: accumulate_acc(output[self.batch_size:], y_com[self.batch_size:], task, accg, topk=(self.top_k,))
                    losses[0].update(loss,  y_com.size(0)) 
                    losses[1].update(loss_class,  y_com.size(0))
                    losses[2].update(loss_kd,  y_com.size(0))
                    #losses[2].update(loss_hardKD, y_com.size(0))
                    #losses[3].update(loss_middle,y_com.size(0))
                    losses[3].update(loss_middle,1)
                    losses[4].update(loss_balancing,1)
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3e} | CE Loss {lossb.avg:.3e} | KD Loss {lossc.avg:.3f}  | Middle loss {middle.avg:.3e} | Balancing loss {balancing.avg:.3e}'.format(loss=losses[0],lossb=losses[1],lossc=losses[2],middle=losses[3],balancing=losses[4]))
                self.log(' * Train Acc {acc.avg:.3f} | Train Acc Gen {accg.avg:.3f}'.format(acc=acc,accg=accg))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for i in range(num_meter)]
                acc = AverageMeter()
                accg = AverageMeter()

        self.model.eval()
        self.last_last_valid_out_dim = self.last_valid_out_dim
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher
        
        # new teacher
        if (self.out_dim == self.valid_out_dim): need_train = False
        self.previous_teacher = Teacher(solver=copy.deepcopy(self.model), generator=self.generator, gen_opt = self.generator_optimizer, img_shape = (-1, train_dataset.nch,train_dataset.im_size, train_dataset.im_size), iters = self.power_iters, deep_inv_params = self.deep_inv_params, class_idx = np.arange(self.valid_out_dim), train = need_train, config = self.config)
        self.sample(self.previous_teacher, self.batch_size, self.device, return_scores=False)
        if len(self.config['gpuid']) > 1:
            self.previous_linear = copy.deepcopy(self.model.module.last)
        else:
            self.previous_linear = copy.deepcopy(self.model.last)
        self.inversion_replay = True

        try:
            return batch_time.avg
        except:
            return None

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):

        loss_kd = torch.zeros((1,), requires_grad=True).cuda()

        if dw_force is not None:
            dw_cls = dw_force
        elif self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]

        # forward pass
        logits = self.forward(inputs)

        # classification 
        class_idx = np.arange(self.batch_size)
        loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD old
        if target_scores is not None:
            loss_kd = self.mu * loss_fn_kd(logits[class_idx], target_scores[class_idx], dw_cls[class_idx], np.arange(self.last_valid_out_dim).tolist(), self.DTemp)

        # KD new
        if target_scores is not None:
            target_scores = F.softmax(target_scores[:, :self.last_valid_out_dim] / self.DTemp, dim=1)
            target_scores = [target_scores]
            target_scores.append(torch.zeros((len(targets),self.valid_out_dim-self.last_valid_out_dim), requires_grad=True).cuda())
            target_scores = torch.cat(target_scores, dim=1)
            loss_kd += self.mu * loss_fn_kd(logits[kd_index], target_scores[kd_index], dw_cls[kd_index], np.arange(self.valid_out_dim).tolist(), self.DTemp, soft_t = True)

        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        loss_middle=torch.zeros((1,), requires_grad=True).cuda()
        loss_balancing=torch.zeros((1,), requires_grad=True).cuda()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), loss_middle.detach(), loss_balancing.detach(), logits
    ##########################################
    #             MODEL UTILS                #
    ##########################################

    def combine_data(self, data):
        x, y = [],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(data[i][1])
        x, y = torch.cat(x), torch.cat(y)
        return x, y

    def save_model(self, filename):
        
        model_state = self.generator.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving generator model to:', filename)
        torch.save(model_state, filename + 'generator.pth')
        super(DeepInversionGenBN, self).save_model(filename)

    def load_model(self, filename):
        self.generator.load_state_dict(torch.load(filename + 'generator.pth'))
        if self.gpu:
            self.generator = self.generator.cuda()
        self.generator.eval()
        super(DeepInversionGenBN, self).load_model(filename)

    def create_generator(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        generator = models.__dict__[cfg['gen_model_type']].__dict__[cfg['gen_model_name']]()
        return generator

    def print_model(self):
        super(DeepInversionGenBN, self).print_model()
        self.log(self.generator)
        self.log('#parameter of generator:', self.count_parameter_gen())
    
    def reset_model(self):
        super(DeepInversionGenBN, self).reset_model()
        self.generator.apply(weight_reset)

    def count_parameter_gen(self):
        return sum(p.numel() for p in self.generator.parameters())

    def count_memory(self, dataset_size):
        return self.count_parameter() + self.count_parameter_gen() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]

    def cuda_gen(self):
        self.generator = self.generator.cuda()
        return self

    def sample(self, teacher, dim, device, return_scores=True):
        return teacher.sample(dim, device, return_scores=return_scores)

class DeepInversionLWF(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(DeepInversionLWF, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, real_x,real_y,x_fake, y_fake, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()

        if dw_force is not None:
            dw_cls = dw_force
        elif self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]

        # forward pass
        logits = self.forward(inputs)

        # classification 
        class_idx = np.arange(self.batch_size)
        loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD
        if target_scores is not None:
            loss_kd = self.mu * loss_fn_kd(logits, target_scores, dw_cls, np.arange(self.last_valid_out_dim).tolist(), self.DTemp)

        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        loss_middle=torch.zeros((1,), requires_grad=True).cuda()
        loss_balancing=torch.zeros((1,), requires_grad=True).cuda()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), loss_middle.detach(), loss_balancing.detach(), logits

class AlwaysBeDreaming(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(AlwaysBeDreaming, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, real_x,real_y,x_fake, y_fake, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()
        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits_pen = self.model.forward(x=inputs, pen=True)
        if len(self.config['gpuid']) > 1:
            logits = self.model.module.last(logits_pen)
        else:
            logits = self.model.last(logits_pen)
        
        # classification 
        class_idx = np.arange(self.batch_size)
        if self.inversion_replay:

            # local classification
            loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) 

            # ft classification  
            with torch.no_grad():             
                feat_class = self.model.forward(x=inputs, pen=True).detach()
            if len(self.config['gpuid']) > 1:
                loss_class += self.criterion(self.model.module.last(feat_class), targets.long(), dw_cls)
            else:
                loss_class += self.criterion(self.model.last(feat_class), targets.long(), dw_cls)
            
        else: # start 
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])
            # loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], targets[class_idx].long(), dw_cls[class_idx]) # 211 no split logits

        # KD
        if target_scores is not None:

            # hard - linear
            kd_index = np.arange(2 * self.batch_size)
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            logits_KD = self.previous_linear(logits_pen[kd_index])[:,:self.last_valid_out_dim]
            logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(inputs[kd_index]))[:,:self.last_valid_out_dim]
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))
        else:
            loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        loss_middle=torch.zeros((1,), requires_grad=True).cuda()
            
        if self.previous_teacher is not None and self.config['balancing']:
            task_step=self.valid_out_dim-self.last_valid_out_dim
            task_weights=[]
            loss_balancing=torch.zeros((1,),requires_grad=True).cuda()
            if len(self.config['gpuid']) > 1:
                for i in range(self.valid_out_dim//task_step):
                    task_weights.append(self.model.module.last.weight[i*task_step:(i+1)*task_step,:])
            else:
                for i in range(self.valid_out_dim//task_step):
                    task_weights.append(self.model.last.weight[i*task_step:(i+1)*task_step,:])

            for i in range(len(task_weights)):
                if i==0:
                    oldest_task_weights=task_weights[i].detach().clone()
                else:
                    if self.config['balancing_loss_type']=='l1':
                        loss_balancing+=F.l1_loss(task_weights[i].norm(),oldest_task_weights.norm())/task_step
                    elif self.config['balancing_loss_type']=='l2':
                        loss_balancing+=F.mse_loss(task_weights[i].norm(),oldest_task_weights.norm())/task_step
            loss_balancing*=self.config['balancing_mu']
        else:
            loss_balancing=torch.zeros((1,),requires_grad=True).cuda()
        total_loss = loss_class + loss_kd + loss_middle + loss_balancing
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), loss_middle.detach(), loss_balancing.detach(), logits

class AlwaysBeDreamingBalancing(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(AlwaysBeDreamingBalancing, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.md_criterion = SP(reduction='none').cuda()
        if self.config['balancing_loss_type']=='l1':
            self.norm_type=1
        elif self.config['balancing_loss_type']=='l2':
            self.norm_type=2

        self.cc_criterion=CC(self.config['cc_gamma'],self.config['p_order'],reduction='none').cuda()

    def update_model(self, real_x,real_y,x_fake, y_fake, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        task_step=self.valid_out_dim-self.last_valid_out_dim
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()
        #print("ABD self.last_valid_out_dim shape : ",self.last_valid_out_dim) #5
        #print("ABD self.valid_out_dim", self.valid_out_dim) #10

        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        #print("rnt  :",rnt)
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits_pen = self.model.forward(x=inputs, pen=True)

        if len(self.config['gpuid']) > 1:
            logits = self.model.module.last(logits_pen)
        else:
            logits = self.model.last(logits_pen)
        
        # classification 
        class_idx = np.arange(self.batch_size)
        if self.inversion_replay:
            if self.config['classification_index']=='real':
                class_idx=class_idx
            elif self.config['classification_index']=='fake':
                class_idx=class_idx+self.batch_size
            else: # real_fake
                class_idx=np.arange(2*self.batch_size)

            # local classification
            if self.config['classification_type']=='local':
                loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) 
            elif self.config['classification_type']=='global':
                loss_class = self.criterion(logits,targets,dw_cls)
            # ft classification  
            if self.config['ft']:
                if len(self.config['gpuid']) > 1:
                    loss_class += self.criterion(self.model.module.last(logits_pen.detach()), targets.long(), dw_cls)
                else:
                    loss_class += self.criterion(self.model.last(logits_pen.detach()), targets.long(), dw_cls)
            
        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD
        # if self.config['dw_kd']:
        #     dw_kd=dw_cls[kd_index]
        # else:

        if target_scores is not None:
            if self.config['kd_index']=='real_fake':
                kd_index= np.arange(2*self.batch_size)
            elif self.config['kd_index']=='fake':
                kd_index= np.arange(self.batch_size)+self.batch_size
            elif self.config['kd_index']=='real':
                kd_index= np.arange(self.batch_size)
            else:
                raise ValueError("middle_index must be real, fake or real_fake")
            dw_kd = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            kd_inputs=inputs[kd_index]
            if self.config['kd_type']=='abd':
                # hard - linear
                logits_KD = self.previous_linear(logits_pen[kd_index])[:,:self.last_valid_out_dim]
                logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(kd_inputs))[:,:self.last_valid_out_dim]
                loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_kd).mean() / (logits_KD.size(1))
            elif self.config['kd_type']=='kd':
                with torch.no_grad():
                    logits_prev = self.previous_teacher.solver.forward(kd_inputs)[:,:self.last_valid_out_dim]
                loss_kd=(-F.log_softmax(logits[kd_index,:self.last_valid_out_dim]/self.config['temp'],dim=1)*logits_prev.softmax(dim=1)/self.config['temp'])
                loss_kd=(loss_kd.sum(dim=1)*dw_kd[kd_index]).mean()/ task_step * self.mu
            elif self.config['kd_type']=='hkd_yj':
                with torch.no_grad():
                    logits_prev = self.previous_teacher.solver.forward(kd_inputs)[:,:self.last_valid_out_dim]
                loss_kd=(F.mse_loss(logits[kd_index,:self.last_valid_out_dim],logits_prev,reduction='none')).mean()/ task_step * self.mu#*dw_kd
            else:
                raise ValueError("kd_type must be abd, kd or hkd_yj")
        else:
            loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        
        #Middle K.D
        if self.previous_teacher:
            if self.config['middle_index']=='real_fake':
                middle_index= np.arange(2*self.batch_size)
            elif self.config['middle_index']=='fake':
                middle_index= np.arange(self.batch_size,2*self.batch_size)
            elif self.config['middle_index']=='real':
                middle_index= np.arange(self.batch_size)
            else:
                raise ValueError("middle_index must be real, fake or real_fake")
            if self.config['middle_kd_type']=='sp':
                logits_pen,out1_m,out2_m,out3_m = self.model.forward(inputs, middle=True)
                with torch.no_grad():
                    logits_prev_middle,out1_pm, out2_pm, out3_pm = self.previous_teacher.solver.forward(inputs,middle=True)
                loss_middle = (self.md_criterion(out1_m[middle_index],out1_pm[middle_index])+self.md_criterion(out2_m[middle_index],out2_pm[middle_index])+self.md_criterion(out3_m[middle_index],out3_pm[middle_index]))
                # if self.config['dw_middle']:
                #     loss_middle*=dw_cls[middle_index]
                loss_middle = loss_middle.mean()*self.config['middle_mu']
            elif self.config['middle_kd_type']=='cc':
                with torch.no_grad():
                    last_logits_pen=self.previous_teacher.generate_scores_pen(inputs)[middle_index]
                loss_middle=self.cc_criterion(logits_pen[middle_index], last_logits_pen)
                # if self.config['dw_middle']:
                #     loss_middle*=(dw_cls[middle_index])#/dw_cls[middle_index].sum(keepdim=True))
                loss_middle=loss_middle.mean()*self.config['middle_mu']
            else:
                raise ValueError("middle_kd_type must be sp or cc")
        else:
            loss_middle=torch.zeros((1,), requires_grad=True).cuda()

        # balancing
        if self.previous_teacher and self.config['balancing']:
            task_weights=[]
            loss_balancing=torch.zeros((1,),requires_grad=True).cuda()
            if len(self.config['gpuid']) > 1:
                for i in range(self.valid_out_dim//task_step):
                    task_weights.append(self.model.module.last.weight[i*task_step:(i+1)*task_step,:])
            else:
                for i in range(self.valid_out_dim//task_step):
                    task_weights.append(self.model.last.weight[i*task_step:(i+1)*task_step,:])

            for i in range(len(task_weights)):
                if i==0:
                    oldest_task_weights=task_weights[i].detach().clone()
                else:
                    if self.config['balancing_loss_type']=='l1':
                        loss_balancing+=F.l1_loss(task_weights[i].norm(),oldest_task_weights.norm())/task_step
                    elif self.config['balancing_loss_type']=='l2':
                        loss_balancing+=F.mse_loss(task_weights[i].norm(),oldest_task_weights.norm())/task_step
            loss_balancing*=self.config['balancing_mu']
        else:
            loss_balancing=torch.zeros((1,),requires_grad=True).cuda()
        total_loss = loss_class + loss_kd + loss_middle + loss_balancing
        #total_loss = loss_class + loss_midfdle + 
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        #return total_loss.detach(), loss_class.detach(), loss_kd.detach(), loss_middle.detach(),logits
        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), loss_middle.detach(), loss_balancing.detach(), logits
