from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.cc import CC
import models
from utils.metric import AverageMeter, Timer
import numpy as np
from learners.datafree_helper import NITeacher, Teacher
from .default import NormalNN, weight_reset, accumulate_acc, loss_fn_kd
import copy
from torch.optim import Adam

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

        self.teacher_type = self.config['teacher_type']

        # repeat call for generator network
        if self.gpu:
            self.cuda_gen()
        self.kd_dataset=learner_config['kd_dataset']
    
    def get_kd_index(self):
        if self.kd_dataset=='real':
            return np.arange(self.batch_size)
        elif self.kd_dataset=='real_fake':
            return np.arange(2*self.batch_size)
        else:
            return np.arange(self.batch_size) + self.batch_size
        
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

            losses = [AverageMeter() for i in range(3)]
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

                    # if KD
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
                    loss, loss_class, loss_kd, output= self.update_model(x_com, y_com, y_hat_com, dw_force = dw_cls, kd_index = np.arange(len(x), len(x_com)))

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y_com = y_com.detach()
                    accumulate_acc(output[:self.batch_size], y_com[:self.batch_size], task, acc, topk=(self.top_k,))
                    if self.inversion_replay: accumulate_acc(output[self.batch_size:], y_com[self.batch_size:], task, accg, topk=(self.top_k,))
                    losses[0].update(loss,  y_com.size(0)) 
                    losses[1].update(loss_class,  y_com.size(0))
                    losses[2].update(loss_kd,  y_com.size(0))
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | CE Loss {lossb.avg:.3f} | KD Loss {lossc.avg:.3f}'.format(loss=losses[0],lossb=losses[1],lossc=losses[2]))
                self.log(' * Train Acc {acc.avg:.3f} | Train Acc Gen {accg.avg:.3f}'.format(acc=acc,accg=accg))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for i in range(3)]
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
        
        if self.teacher_type =='DI':
            self.previous_teacher = Teacher(solver=copy.deepcopy(self.model), generator=self.generator, gen_opt = self.generator_optimizer, img_shape = (-1, train_dataset.nch,train_dataset.im_size, train_dataset.im_size), iters = self.power_iters, deep_inv_params = self.deep_inv_params, class_idx = np.arange(self.valid_out_dim), train = need_train, config = self.config)
            self.sample(self.previous_teacher, self.batch_size, self.device, return_scores=False)
        elif self.teacher_type =='NI': # NI
            print("===hi"*20)
            self.previous_teacher = NITeacher(solver=copy.deepcopy(self.model), generator=self.generator, gen_opt = self.generator_optimizer, img_shape = (-1, train_dataset.nch,train_dataset.im_size, train_dataset.im_size), iters = self.power_iters, deep_inv_params = self.deep_inv_params, class_idx = np.arange(self.valid_out_dim), train = need_train, config = self.config)
            self.previous_teacher.get_images(2000, self.power_iters, self.last_valid_out_dim)
        else:
            raise NotImplementedError


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
        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

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

        # KD
        if target_scores is not None:
            loss_kd = self.mu * loss_fn_kd(logits, target_scores, dw_cls, np.arange(self.last_valid_out_dim).tolist(), self.DTemp)

        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

class AlwaysBeDreaming(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(AlwaysBeDreaming, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.teacher_type = self.config['teacher_type']

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
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
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits


class TBD1(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(TBD1, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.teacher_type = self.config['teacher_type']
        self.kd_criterion=CC(self.config['gamma'],self.config['p_order'])

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
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

        # KD
        if target_scores is not None:
            kd_index = np.arange(2 * self.batch_size)
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            last_logits_pen=self.previous_teacher.generate_scores_pen(inputs[kd_index])
            loss_kd=(self.kd_criterion(logits_pen[kd_index], last_logits_pen)*dw_KD).mean()*self.mu

        else:
            loss_kd = torch.zeros((1,), requires_grad=True).cuda()
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits


class TBD2(DeepInversionGenBN):
    # HARD KD in logits #
    def __init__(self, learner_config):
        super(TBD2, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.teacher_type = self.config['teacher_type']

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()
        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits_middle,out1_m,out2_m,out3_m = self.model.forward(inputs, middle=True)

        if len(self.config['gpuid']) > 1:
            logits = self.model.module.last(logits_middle)
        else:
            logits = self.model.last(logits_middle)
        
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

        if target_scores is not None:
            kd_index=self.get_kd_index()
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            # middle kd
            with torch.no_grad():
                logits_prev_middle,out1_pm, out2_pm, out3_pm = self.previous_teacher.solver.forward(inputs[kd_index],middle=True)
            norm_out1_m=(out1_m/torch.norm(out1_m,dim=1,keepdim=True))[kd_index]
            norm_out2_m=(out2_m/torch.norm(out2_m,dim=1,keepdim=True))[kd_index]
            norm_out3_m=(out3_m/torch.norm(out3_m,dim=1,keepdim=True))[kd_index]
            norm_out1_pm=(out1_pm/torch.norm(out1_pm,dim=1,keepdim=True))
            norm_out2_pm=(out2_pm/torch.norm(out2_pm,dim=1,keepdim=True))
            norm_out3_pm=(out3_pm/torch.norm(out3_pm,dim=1,keepdim=True))
            loss_kd=self.mu*((F.mse_loss(norm_out1_m,norm_out1_pm)+F.mse_loss(norm_out2_m,norm_out2_pm)+F.mse_loss(norm_out3_m,norm_out3_pm))*dw_KD).mean()

        else:
            loss_kd = torch.zeros((1,), requires_grad=True).cuda()
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

class TBD3(DeepInversionGenBN):
    # HARD KD in logits #
    def __init__(self, learner_config):
        super(TBD3, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.teacher_type = self.config['teacher_type']

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()
        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits_pen,out1_m,out2_m,out3_m = self.model.forward(inputs, middle=True)
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

        if target_scores is not None:
            # logits KD
            # hard - linear
            kd_index = np.arange(2 * self.batch_size)
            #print("kd index : ",kd_index)
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]

            logits_KD = self.previous_linear(logits_pen[kd_index])[:,:self.last_valid_out_dim]
            logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(inputs[kd_index]))[:,:self.last_valid_out_dim]
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))

            # middle kd
            kd_index=self.get_kd_index()
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            logits_middle,out1_m,out2_m,out3_m = self.model.forward(inputs[kd_index], middle=True)
            with torch.no_grad():
                logits_prev_middle,out1_pm, out2_pm, out3_pm = self.previous_teacher.solver.forward(inputs[kd_index],middle=True)
            norm_out1_m=(out1_m/torch.norm(out1_m,dim=1,keepdim=True))[kd_index]
            norm_out2_m=(out2_m/torch.norm(out2_m,dim=1,keepdim=True))[kd_index]
            norm_out3_m=(out3_m/torch.norm(out3_m,dim=1,keepdim=True))[kd_index]
            norm_out1_pm=(out1_pm/torch.norm(out1_pm,dim=1,keepdim=True))
            norm_out2_pm=(out2_pm/torch.norm(out2_pm,dim=1,keepdim=True))
            norm_out3_pm=(out3_pm/torch.norm(out3_pm,dim=1,keepdim=True))
            loss_kd=self.mu*((F.mse_loss(norm_out1_m,norm_out1_pm)+F.mse_loss(norm_out2_m,norm_out2_pm)+F.mse_loss(norm_out3_m,norm_out3_pm))*dw_KD).mean()
        else:
            loss_kd = torch.zeros((1,), requires_grad=True).cuda()
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

class TBD4(DeepInversionGenBN):
    # HARD KD in logits #
    def __init__(self, learner_config):
        super(TBD4, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.teacher_type = self.config['teacher_type']

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
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

        if target_scores is not None:
            kd_index=self.get_kd_index()
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            # middle kd
            logits_middle,out1_m,out2_m,out3_m = self.model.forward(inputs[kd_index], middle=True)
            logits_prev_middle,out1_pm, out2_pm, out3_pm = self.previous_teacher.solver.forward(inputs[kd_index],middle=True)
            loss_kd=self.mu*((F.mse_loss(out1_m,out1_pm)+F.mse_loss(out2_m,out2_pm)+F.mse_loss(out3_m,out3_pm)*0.1)*dw_KD).mean()

        else:
            loss_kd = torch.zeros((1,), requires_grad=True).cuda()
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits


        
class TBD5(DeepInversionGenBN):
    # HARD KD in logits #
    def __init__(self, learner_config):
        super(TBD5, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.teacher_type = self.config['teacher_type']

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()
        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits_pen,out1_m,out2_m,out3_m = self.model.forward(inputs, middle=True)

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

        if target_scores is not None:
            # logits KD
            # hard - linear
            kd_index = np.arange(2 * self.batch_size)
            #print("kd index : ",kd_index)
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]

            logits_KD = self.previous_linear(logits_pen[kd_index])[:,:self.last_valid_out_dim]
            logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(inputs[kd_index]))[:,:self.last_valid_out_dim]
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))

            # middle kd
            kd_index=self.get_kd_index()
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            logits_prev_middle,out1_pm, out2_pm, out3_pm = self.previous_teacher.solver.forward(inputs[kd_index],middle=True)
            norm_out1_m=(out1_m/torch.norm(out1_m,dim=1,keepdim=True))[kd_index]
            norm_out2_m=(out2_m/torch.norm(out2_m,dim=1,keepdim=True))[kd_index]
            norm_out3_m=(out3_m/torch.norm(out3_m,dim=1,keepdim=True))[kd_index]
            norm_out1_pm=out1_pm/torch.norm(out1_pm,dim=1,keepdim=True)
            norm_out2_pm=out2_pm/torch.norm(out2_pm,dim=1,keepdim=True)
            norm_out3_pm=out3_pm/torch.norm(out3_pm,dim=1,keepdim=True)
            loss_kd=self.mu*((F.l1_loss(norm_out1_m,norm_out1_pm)+F.l1_loss(norm_out2_m,norm_out2_pm)+F.l1_loss(norm_out3_m,norm_out3_pm))*dw_KD).mean()

        else:
            loss_kd = torch.zeros((1,), requires_grad=True).cuda()
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits
