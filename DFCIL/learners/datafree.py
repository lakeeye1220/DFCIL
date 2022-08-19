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
from learners.sp import SP


class ISCF(NormalNN):
    def __init__(self, learner_config):
        super(ISCF, self).__init__(learner_config)
        self.inversion_replay = False
        self.previous_teacher = None
        self.dw = self.config['DW']
        self.device = 'cuda' if self.gpu else 'cpu'
        self.power_iters = self.config['power_iters']
        self.deep_inv_params = self.config['deep_inv_params']
        self.kd_criterion = nn.MSELoss(reduction="none")
        self.dataset=self.config['dataset']

        # generator parameters
        self.generator = self.create_generator()
        self.generator_optimizer = Adam(params=self.generator.parameters(), lr=self.deep_inv_params[0])
        self.beta = self.config['beta']

        # repeat call for generator network
        if self.gpu:
            self.cuda_gen()
        
        #SPKD loss definition
        self.md_criterion = SP(reduction='none')
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

            #compute the 5 losses of ISCF frameworks - LCE(local CE) + SPKD(Intermediate K.D) + LKD(Logit KD) + FT(Finetuning) + WEQ(Weight equalizer)
            losses = [AverageMeter() for i in range(5)]
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

                    # From 2task, we use the old samples by model-inversion approach and we combine the real images with synthetic images
                    if self.inversion_replay:
                        y_hat = self.previous_teacher.generate_scores(x, allowed_predictions=np.arange(self.last_valid_out_dim))
                        _, y_hat_com = self.combine_data(((x, y_hat),(x_replay, y_replay_hat)))
                    else:
                        y_hat_com = None

                    # combine inputs and generated samples for SPKD,LKD and FT
                    if self.inversion_replay:
                        x_com, y_com = self.combine_data(((x, y),(x_replay, y_replay)))
                    else:
                        x_com, y_com = x, y

                    # sd data weighting for LCE and FT loss 
                    if self.dw:
                        dw_cls = self.dw_k[y_com.long()]
                    else:
                        dw_cls = None

                    # model update
                    loss, loss_class, loss_kd,loss_middle,loss_balancing, output= self.update_model(x_com, y_com, y_hat_com, dw_force = dw_cls, kd_index = np.arange(len(x), len(x_com)))

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y_com = y_com.detach()
                    accumulate_acc(output[:self.batch_size], y_com[:self.batch_size], task, acc, topk=(self.top_k,))
                    if self.inversion_replay: accumulate_acc(output[self.batch_size:], y_com[self.batch_size:], task, accg, topk=(self.top_k,))
                    losses[0].update(loss,  y_com.size(0)) 
                    losses[1].update(loss_class,  y_com.size(0))
                    losses[2].update(loss_kd,  y_com.size(0))
                    losses[3].update(loss_middle,  y_com.size(0))
                    losses[4].update(loss_balancing,  y_com.size(0))
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.4e} | CE Loss {lossb.avg:.4e} | LKD Loss {lossc.avg:.4e} | SP Loss {lossd.avg:.4e} | WEQ Reg {losse.avg:.4e}'.format(loss=losses[0],lossb=losses[1],lossc=losses[2],lossd=losses[3],losse=losses[4]))
                self.log(' * Train Acc {acc.avg:.3f} | Train Acc Gen {accg.avg:.3f}'.format(acc=acc,accg=accg))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for i in range(5)]
                acc = AverageMeter()
                accg = AverageMeter()


        self.model.eval()
        self.last_last_valid_out_dim = self.last_valid_out_dim
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher
        
        # define the new model - current model 
        if (self.out_dim == self.valid_out_dim) or (self.dataset== 'TinyImageNet100' and self.valid_out_dim==100): need_train = False
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
        super(ISCF, self).save_model(filename)

    def load_model(self, filename):
        self.generator.load_state_dict(torch.load(filename + 'generator.pth'))
        if self.gpu:
            self.generator = self.generator.cuda()
        self.generator.eval()
        super(ISCF, self).load_model(filename)

    def create_generator(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        generator = models.__dict__[cfg['gen_model_type']].__dict__[cfg['gen_model_name']]()
        return generator

    def print_model(self):
        super(ISCF, self).print_model()
        self.log(self.generator)
        self.log('#parameter of generator:', self.count_parameter_gen())
    
    def reset_model(self):
        super(ISCF, self).reset_model()
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



    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        task_step=self.valid_out_dim-self.last_valid_out_dim
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()

        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits_pen,m = self.model.forward(inputs, middle=True)

        if len(self.config['gpuid']) > 1:
            logits = self.model.module.last(logits_pen)
        else:
            logits = self.model.last(logits_pen)
        
        # classification 
        class_idx = np.arange(self.batch_size) # real
        if self.inversion_replay:

            # local classification - LCE loss: the logit dimension is from last_valid_out_dim to valid_out_dim
            loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) 
            
            # ft classification  
            if len(self.config['gpuid']) > 1:
                loss_class += self.criterion(self.model.module.last(logits_pen.detach()), targets.long(), dw_cls)
            else:
                loss_class += self.criterion(self.model.last(logits_pen.detach()), targets.long(), dw_cls)
        
        #first task local classification when we do not use any synthetic data     
        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])


        
        #SPKD - Intermediate KD
        if self.previous_teacher: # after 2nd task
            middle_index= np.arange(2*self.batch_size) # real n fake
            with torch.no_grad():
                logits_prev_middle,pm = self.previous_teacher.solver.forward(inputs[middle_index],middle=True)
            if len(pm)==3:
                out1_pm,out2_pm,out3_pm=pm
                out1_m,out2_m,out3_m=m
                loss_sp = (self.md_criterion(out1_m[middle_index],out1_pm)+self.md_criterion(out2_m[middle_index],out2_pm)+self.md_criterion(out3_m[middle_index],out3_pm))/3.
            else: # for imagenet
                out1_pm,out2_pm,out3_pm,out4_pm=pm
                out1_m,out2_m,out3_m,out4_m=m
                loss_sp = (self.md_criterion(out1_m[middle_index],out1_pm)+self.md_criterion(out2_m[middle_index],out2_pm)+self.md_criterion(out3_m[middle_index],out3_pm)+self.md_criterion(out4_m[middle_index],out4_pm))/4.
            
            loss_sp = loss_sp.mean()*self.config['sp_mu']
        else:
            loss_sp=torch.zeros((1,), requires_grad=True).cuda()

        # Logit KD for maintaining the output probability 
        if self.previous_teacher:
            kd_index= np.arange(2*self.batch_size)
            with torch.no_grad():
                logits_prevpen = self.previous_teacher.solver.forward(inputs[kd_index],pen=True)
                logits_prev=self.previous_linear(logits_prevpen)[:,:self.last_valid_out_dim].detach()

            loss_lkd=(F.mse_loss(logits[kd_index,:self.last_valid_out_dim],logits_prev,reduction='none').sum(dim=1)) * self.mu / task_step
            loss_lkd=loss_lkd.mean()
        else:
            loss_lkd = torch.zeros((1,), requires_grad=True).cuda()
        
        # weight equalizer for balancing the average norm of weight 
        if self.previous_teacher:
            if len(self.config['gpuid']) > 1:
                last_weights=self.model.module.last.weight[:self.valid_out_dim,:].detach()
                last_bias=self.model.module.last.bias[:self.valid_out_dim].detach().unsqueeze(-1)
                cur_weights=self.model.module.last.weight[:self.valid_out_dim,:] 
                cur_bias=self.model.module.last.bias[:self.valid_out_dim].unsqueeze(-1) 
            else:
                last_weights=self.model.last.weight[:self.valid_out_dim,:].detach()
                last_bias=self.model.last.bias[:self.valid_out_dim].detach().unsqueeze(-1)
                cur_weights=self.model.last.weight[:self.valid_out_dim,:]
                cur_bias=self.model.last.bias[:self.valid_out_dim].unsqueeze(-1) 

            last_params=torch.cat([last_weights,last_bias],dim=1)
            cur_params=torch.cat([cur_weights,cur_bias],dim=1)
            weq_regularizer=F.mse_loss(last_params.norm(dim=1,keepdim=True).mean().expand(self.valid_out_dim),cur_params.norm(dim=1))
            weq_regularizer*=self.config['weq_mu']
        else:
            weq_regularizer=torch.zeros((1,),requires_grad=True).cuda()

        # calculate the 5 losses - LCE + SPKD + LKD + FT + WEQ, loss_class include the LCE and FT losses
        total_loss = loss_class + loss_lkd + loss_sp + weq_regularizer

        self.optimizer.zero_grad()
        total_loss.backward()
        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_lkd.detach(), loss_sp.detach(), weq_regularizer.detach(), logits
