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
class SP(nn.Module):
    def __init__(self):
        super(SP,self).__init__()

    def forward(self,fm_s,fm_t):
        fm_s = fm_s.view(fm_s.size(0),-1)
        G_s = torch.mm(fm_s,fm_s.t())
        norm_G_s =F.normalize(G_s,p=2,dim=1)

        fm_t = fm_t.view(fm_t.size(0),-1)
        G_t = torch.mm(fm_t,fm_t.t())
        norm_G_t = F.normalize(G_t,p=2,dim=1)
        loss = F.mse_loss(norm_G_s,norm_G_t)
        return loss

class DeepInversionGenBN(NormalNN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN, self).__init__(learner_config)
        self.inversion_replay = False
        self.previous_teacher = None
        self.dw = self.config['DW']
        self.middle= self.config['middle']
        self.device = 'cuda' if self.gpu else 'cpu'
        self.power_iters = self.config['power_iters']
        self.deep_inv_params = self.config['deep_inv_params']
        self.kd_criterion = nn.MSELoss(reduction="none")

        # gen parameters
        self.generator = self.create_generator()
        self.generator_optimizer = Adam(params=self.generator.parameters(), lr=self.deep_inv_params[0])
        self.beta = self.config['beta']
        self.middle_mu = self.config['middle_mu']
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
                self.log(' * Loss {loss.avg:.3e} | CE Loss {lossb.avg:.3e} | hard KD Loss {lossc.avg:.3f}  | Middle loss {middle.avg:.3e} | Balancing loss {balancing.avg:.3e}'.format(loss=losses[0],lossb=losses[1],lossc=losses[2],middle=losses[3],balancing=losses[4]))
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

    def update_model(self, real_x,real_y,x_fake,y_fake,inputs, targets, target_scores = None, dw_force = None, kd_index = None):

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
            print("self.mu : ",self.mu)
        # KD new
        if target_scores is not None:
            target_scores = F.softmax(target_scores[:, :self.last_valid_out_dim] / self.DTemp, dim=1)
            target_scores = [target_scores]
            target_scores.append(torch.zeros((len(targets),self.valid_out_dim-self.last_valid_out_dim), requires_grad=True).cuda())
            target_scores = torch.cat(target_scores, dim=1)
            loss_kd += self.mu * loss_fn_kd(logits[kd_index], target_scores[kd_index], dw_cls[kd_index], np.arange(self.valid_out_dim).tolist(), self.DTemp, soft_t = True)
        
        loss_middle =torch.zeros((1,),requires_grad=True).cuda()
        if self.previous_teacher is not None:
            logits_middle,out1_m,out2_m,out3_m = self.model.forward(real_x, middle=True)
            logits_prev_middle,out1_pm, out2_pm, out3_pm = self.previous_teacher.solver.forward(real_x,middle=True)
            #print("layer 1 difference : ",torch.norm(out1_m,out1_pm,1))
            #print("layer 2 difference : ",torch.norm(out2_m, out2_pm,1))
            #print("layer 3 differnce : ", torch.norm(out3_m,out3_pm,1))
            loss_middle = self.mu*(torch.norm(out1_m - out1_pm,2)+torch.norm(out2_m-out2_pm,2)+torch.norm(out3_m-out3_pm,2)).mean()
        print("General middle layer distllation loss: ",loss_middle.detach())
        '''
        loss_hardKD = torch.zeros((1,),requires_grad=True).cuda()
        if x_fake is not None and self.previous_teacher:
            logits_old = self.model.forward(x_fake)
            logits_prev_old = self.previous_teacher.solver.forward(x_fake)
            loss_hardKD = torch.norm(logits_old[:self.last_valid_out_dim]-logits_prev_old,1).sum()*self.mu 
        print("General loss_hardKD : ",loss_hardKD)
        '''


        #total_loss = loss_class + loss_kd + loss_middle

        #total_loss = loss_class + loss_kd+loss_middle
        total_loss = loss_class + loss_kd + loss_middle
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        #return total_loss.detach(), loss_class.detach(), loss_kd.detach(),loss_middle.detach(),logits
        return total_loss.detach(),loss_class.detach(),loss_kd.detach(),loss_middle.detach(),logits
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

    def update_model(self, real_x,real_y, x_fake, y_fake, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        print("Welcome to DI_LWF")        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()

        if dw_force is not None:
            dw_cls = dw_force
        elif self.dw:
            dw_cls = self.dw_k[targets.long()]
            #print("self.dw_K : ",dw_k)
            #print("length of self.dw_k  :",len(self.dw_K))
            #print("dw_cls : ",dw_cls)
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
        
        loss_middle = torch.zeros((1,),requires_grad=True).cuda()
        if self.previous_teacher is not None:
            logits_middle,out1_m,out2_m,out3_m = self.model.forward(real_x, middle=True)
            logits_prev_middle,out1_pm, out2_pm, out3_pm = self.previous_teacher.solver.forward(real_x,middle=True)
            loss_middle = self.mu*(torch.norm(out1_m - out1_pm,2)+torch.norm(out2_m-out2_pm,2)+torch.norm(out3_m-out3_pm,2)).mean()

            print("DI layer 1 difference : ",torch.norm(out1_m-out1_pm,1), "l2 : ",torch.norm(out1_m-out1_pm,2))
            print("DI layer 2 difference : ",torch.norm(out2_m-out2_pm,1),"l2 : ",torch.norm(out2_m-out2_pm,2))
            print("DI layer 3 differnce : ", torch.norm(out3_m-out3_pm,1), "l2 : ",torch.norm(out3_m-out3_pm,2))
        '''
        loss_hardKD = torch.zeros((1,),requires_grad=True).cuda()
        if x_fake is not None and self.previous_teacher:
            logits_old = self.model.forward(x_fake)
            logits_prev_old = self.previous_teacher.solver.forward(x_fake)
            loss_hardKD = torch.norm(logits_old[:self.last_valid_out_dim]-logits_prev_old,1).sum()*self.mu
        print("DI_LWF ", loss_hardKD)
        '''
        total_loss = loss_class + loss_kd+loss_middle
        #total_loss = loss_class + loss_hardKD + loss_middle
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        #return total_loss.detach(), loss_class.detach(), loss_kd.detach(),loss_middle.detach(), logits
        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), loss_middle.detach(), logits

class AlwaysBeDreamingBalancing(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(AlwaysBeDreamingBalancing, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.md_criterion = SP().cuda()
        if self.config['balancing_loss_type']=='l1':
            self.norm_type=1
        elif self.config['balancing_loss_type']=='l2':
            self.norm_type=2

    def update_model(self, real_x,real_y,x_fake, y_fake, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
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


        #mapping is one batch 
        #previous observed data(images):inversion data, current data(images):real data(large value of dw_cls)

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
            
        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD
        if target_scores is not None and self.config['abd_kd']:
            # hard - linear
            kd_index = np.arange(2 * self.batch_size)
            #print("kd index : ",kd_index)
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            #print("self.dw_k: ", self.dw_k)
            #print("dw_KD : ",dw_KD)
            
            #print("logits_pen[kd_index]",logits_pen[kd_index].shape)
            #print("linear layer : ",self.previous_linear(logits_pen[kd_index])[:,:self.last_valid_out_dim].shape)

            logits_KD = self.previous_linear(logits_pen[kd_index])[:,:self.last_valid_out_dim]
            logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(inputs[kd_index]))[:,:self.last_valid_out_dim]
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))
            #print("logits_KD : ",logits_KD.size(1))
            #loss_kd = (self.
        else:
            loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        
        #Middle K.D
        #print("previous teacher : ",self.previous_teacher)
        if self.previous_teacher is not None and self.config['middle']:
            logits_middle,out1_m,out2_m,out3_m = self.model.forward(inputs, middle=True)
            with torch.no_grad():
                logits_prev_middle,out1_pm, out2_pm, out3_pm = self.previous_teacher.solver.forward(inputs,middle=True)
            loss_middle = (self.md_criterion(out1_m,out1_pm)+self.md_criterion(out2_m,out2_pm)+self.md_criterion(out3_m,out3_pm))*self.middle_mu
            #loss_middle = self.mu*(torch.norm(out1_m - out1_pm,2)+torch.norm(out2_m-out2_pm,2)+0.1*torch.norm(out3_m-out3_pm,2)).mean()
            #print("layer 1 difference : ",torch.norm(out1_m-out1_pm,1), "l2 : ",torch.norm(out1_m-out1_pm,2))
            #print("layer 2 difference : ",torch.norm(out2_m-out2_pm,1),"l2 : ",torch.norm(out2_m-out2_pm,2))
            #print("layer 3 differnce : ", torch.norm(out3_m-out3_pm,1), "l2 : ",torch.norm(out3_m-out3_pm,2))
        else:
            loss_middle=torch.zeros((1,),requires_grad=True).cuda()
        '''
        loss_hardKD = torch.zeros((1,),requires_grad=True).cuda()
        if x_fake is not None and self.previous_teacher:
            logits_old = self.model.forward(x_fake)
            logits_prev_old = self.previous_teacher.solver.forward(x_fake)
            #print("logits old : ",logits_old.shape)
            #print("dimension : ",logits_old[:,:self.last_valid_out_dim].shape)
            #print("previous logits : ",logits_prev_old.shape)
            loss_hardKD = torch.norm(logits_old[:,:self.last_valid_out_dim]-logits_prev_old[:,:self.last_valid_out_dim],1).sum()/(len(x_fake)*10) #task_size
        #print("ABD hard KD : ",loss_hardKD.detach())
        '''
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
                    oldest_task_weights=task_weights[i].detach()
                else:
                    loss_balancing+=torch.norm(task_weights[i].norm()-oldest_task_weights.norm(),self.norm_type)
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
