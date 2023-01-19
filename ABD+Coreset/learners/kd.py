from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.metric import AverageMeter, Timer
import numpy as np
from models.resnet import BiasLayer
from .default import NormalNN,  accumulate_acc, loss_fn_kd, Teacher
import copy
import torch.utils.data as data
from PIL import Image
class LWF(NormalNN):

    def __init__(self, learner_config):
        super(LWF, self).__init__(learner_config)
        self.previous_teacher = None
        self.replay = False
        self.past_tasks = []
        self.bic_layers = None
        self.ete_flag = False
        
    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # train
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
        
            losses = [AverageMeter() for l in range(3)]
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
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
                    
                    # if KD
                    if self.replay:
                        allowed_predictions = list(range(self.last_valid_out_dim))
                        y_hat, _ = self.previous_teacher.generate_scores(x, allowed_predictions=allowed_predictions)
                    else:
                        y_hat = None

                    # model update - training data
                    loss, loss_class, loss_distill, output= self.update_model(x, y, y_hat)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses[0].update(loss,  y.size(0)) 
                    losses[1].update(loss_class,  y.size(0)) 
                    losses[2].update(loss_distill,  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))
                self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for l in range(3)]
                acc = AverageMeter()

        self.model.eval()
        self.past_tasks.append(np.arange(self.last_valid_out_dim,self.valid_out_dim))
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # E2W
        if self.ete_flag:

            # for eval
            if self.previous_teacher is not None:
                self.previous_previous_teacher = self.previous_teacher

            # new teacher
            teacher = Teacher(solver=self.model)
            self.previous_teacher = copy.deepcopy(teacher)

            # Extend memory
            self.task_count += 1

            if self.memory_size > 0:
                train_dataset.update_coreset_ete(self.memory_size, np.arange(self.last_valid_out_dim), teacher)

        # BiC
        elif self.bic_layers is None:

            # Extend memory
            self.task_count += 1
            if self.memory_size > 0:
                train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

            # for eval
            if self.previous_teacher is not None:
                self.previous_previous_teacher = self.previous_teacher

            # new teacher
            teacher = Teacher(solver=self.model)
            self.previous_teacher = copy.deepcopy(teacher)
            if len(self.config['gpuid']) > 1:
                self.previous_linear = copy.deepcopy(self.model.module.last)
            else:
                self.previous_linear = copy.deepcopy(self.model.last)
        
        # LwF
        else:
            # for eval
            if self.previous_teacher is not None:
                self.previous_previous_teacher = self.previous_teacher

            # new teacher
            teacher = TeacherBiC(solver=self.model, bic_layers = self.bic_layers)
            self.previous_teacher = copy.deepcopy(teacher)

            # Extend memory
            self.task_count += 1
            if self.memory_size > 0:
                train_dataset.update_coreset_ic(self.memory_size, np.arange(self.last_valid_out_dim), teacher)

        self.replay = True
        try:
            return batch_time.avg
        except:
            return None

    def update_model(self, inputs, targets, target_KD = None):
        
        total_loss = torch.zeros((1,), requires_grad=True).cuda()

        # classification loss
        if self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        loss_class = self.criterion(logits, targets.long(), dw_cls)
        total_loss += loss_class

        # KD
        if target_KD is not None:
            dw_KD = self.dw_k[-1 * torch.ones(len(target_KD),).long()]
            logits_KD = logits
            loss_distill = loss_fn_kd(logits_KD, target_KD, dw_KD, np.arange(self.last_valid_out_dim).tolist(), self.DTemp)
            total_loss += self.mu * loss_distill
        else:
            loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits

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

class LWF_MC(LWF):

    def __init__(self, learner_config):
        super(LWF_MC, self).__init__(learner_config)
        self.ce_loss = nn.BCELoss(reduction='sum')

    def update_model(self, inputs, targets, target_KD = None):
        
        # get output
        logits = self.forward(inputs)

        # KD
        if target_KD is not None:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
            total_loss = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
        else:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            total_loss = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), total_loss.detach(), torch.zeros((1,), requires_grad=True).cuda().detach(), logits

class ETE(LWF):

    def __init__(self, learner_config):
        super(ETE, self).__init__(learner_config)
        self.ete_flag = True

    def update_model(self, inputs, targets, target_KD = None):

        # classification loss
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        loss_class = self.criterion(logits, targets.long(), dw_cls)
        total_loss = loss_class

        # KD
        if target_KD is not None:

            dw_KD = self.dw_k[-1 * torch.ones(len(target_KD),).long()]
            logits_KD = logits
            for task_l in self.past_tasks:
                loss_distill = loss_fn_kd(logits_KD, target_KD, dw_KD, task_l.tolist(), self.DTemp)
                total_loss += self.mu * loss_distill * (len(task_l) / self.last_valid_out_dim)
        else:
            loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits

    def update_model_b(self, inputs, targets, target_KD = None, target_KD_B = None):

        # classification loss
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        loss_class = self.criterion(logits, targets.long(), dw_cls)
        total_loss = loss_class

        # KD
        if target_KD is not None:

            dw_KD = self.dw_k[-1 * torch.ones(len(target_KD),).long()]
            logits_KD = logits
            for task_l in self.past_tasks:
                loss_distill = loss_fn_kd(logits_KD, target_KD, dw_KD, task_l.tolist(), self.DTemp)
                total_loss += self.mu * loss_distill * (len(task_l) / self.valid_out_dim)

            # current task
            loss_distill = loss_fn_kd(logits_KD[:, self.last_valid_out_dim:self.valid_out_dim], target_KD, dw_KD, np.arange(self.valid_out_dim-self.last_valid_out_dim), self.DTemp)
            total_loss += self.mu * loss_distill * ((self.valid_out_dim-self.last_valid_out_dim) / self.valid_out_dim)

        else:
            loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        if self.task_count == 0:
            return super(ETE, self).learn_batch(train_loader, train_dataset, model_save_dir, val_loader)

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
        
            losses = [AverageMeter() for l in range(3)]
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
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
                    
                    # if KD
                    if self.replay:
                        allowed_predictions = list(range(self.last_valid_out_dim))
                        y_hat, _ = self.previous_teacher.generate_scores(x, allowed_predictions=allowed_predictions)
                    else:
                        y_hat = None

                    # model update - training data
                    loss, loss_class, loss_distill, output= self.update_model(x, y, y_hat)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses[0].update(loss,  y.size(0)) 
                    losses[1].update(loss_class,  y.size(0)) 
                    losses[2].update(loss_distill,  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))
                self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for l in range(3)]
                acc = AverageMeter()

        # new teacher
        teacher = Teacher(solver=self.model)
        self.current_teacher = copy.deepcopy(teacher)

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset_ete(self.memory_size, np.arange(self.valid_out_dim), teacher)

        # trains
        if need_train:

            # part b
            # dataset tune
            train_dataset.load_dataset(train_dataset.t, train=True)
            train_dataset.append_coreset(only=True)

            self.config['lr'] = self.config['lr'] / 1e2
            if len(self.config['gpuid']) > 1:
                self.optimizer, self.scheduler = self.new_optimizer(self.model.module.last)
            else:
                self.optimizer, self.scheduler = self.new_optimizer(self.model.last)
            self.config['lr'] = self.config['lr'] * 1e2

            # Evaluate the performance of current task
            self.log('Balance Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)
        
            losses = [AverageMeter() for l in range(3)]
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
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
                    
                    # if KD
                    if self.replay:
                        allowed_predictions = list(range(self.last_valid_out_dim))
                        y_hat, _ = self.previous_teacher.generate_scores(x, allowed_predictions=allowed_predictions)
                        y_hat_b, _ = self.current_teacher.generate_scores(x, allowed_predictions=np.arange(self.last_valid_out_dim, self.valid_out_dim))
                    else:
                        y_hat = None

                    # model update - training data
                    loss, loss_class, loss_distill, output= self.update_model_b(x, y, y_hat, y_hat_b)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses[0].update(loss,  y.size(0)) 
                    losses[1].update(loss_class,  y.size(0)) 
                    losses[2].update(loss_distill,  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Balanced Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))
                self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for l in range(3)]
                acc = AverageMeter()

            # dataset final
            train_dataset.load_dataset(train_dataset.t, train=True)
            train_dataset.append_coreset(only=False)

        self.model.eval()
        self.past_tasks.append(np.arange(self.last_valid_out_dim,self.valid_out_dim))
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        ## for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)
        self.replay = True

        try:
            return batch_time.avg
        except:
            return None

class BIC(LWF):

    def __init__(self, learner_config):
        super(BIC, self).__init__(learner_config)
        self.bic_layers = []

    def forward(self, x):
        y_hat = self.model.forward(x)[:, :self.valid_out_dim]

        # forward with bic
        for i in range(len(self.bic_layers)):
            y_hat[:,self.bic_layers[i][0]] = self.bic_layers[i][1](y_hat[:,self.bic_layers[i][0]])

        return y_hat

    def update_model(self, inputs, targets, target_KD = None):
        
        # classification loss
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        loss_class = self.criterion(logits, targets.long(), dw_cls)
        total_loss = loss_class

        # KD
        if  target_KD is not None:

            mu = self.last_valid_out_dim / self.valid_out_dim
            total_loss = (1 - mu) * total_loss

            dw_KD = self.dw_k[-1 * torch.ones(len(target_KD),).long()]
            logits_KD = logits
            loss_distill = loss_fn_kd(logits_KD, target_KD, dw_KD, np.arange(0,self.last_valid_out_dim), self.DTemp)
            total_loss += mu * loss_distill
        else:
            loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits


    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        if self.task_count == 0:
            return super(BIC, self).learn_batch(train_loader, train_dataset, model_save_dir, val_loader)

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

            # dataset start
            train_dataset.load_dataset(train_dataset.t, train=True)
            train_dataset.load_bic_dataset()
            train_dataset.append_coreset_ic()

            try: 
                self.load_model(model_save_dir, class_only = True)
            except:
                # data weighting
                self.data_weighting(train_dataset)
                
                # Evaluate the performance of current task
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
                if val_loader is not None:
                    self.validation(val_loader)
            
                losses = [AverageMeter() for l in range(3)]
                acc = AverageMeter()
                batch_time = AverageMeter()
                batch_timer = Timer()
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
                        
                        # if KD
                        if self.replay:
                            allowed_predictions = list(range(self.last_valid_out_dim))
                            y_hat, _ = self.previous_teacher.generate_scores(x, allowed_predictions=allowed_predictions)
                        else:
                            y_hat = None

                        # model update - training data
                        loss, loss_class, loss_distill, output= self.update_model(x, y, y_hat)

                        # measure elapsed time
                        batch_time.update(batch_timer.toc()) 

                        # measure accuracy and record loss
                        y = y.detach()
                        accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                        losses[0].update(loss,  y.size(0)) 
                        losses[1].update(loss_class,  y.size(0)) 
                        losses[2].update(loss_distill,  y.size(0)) 
                        batch_timer.tic()

                    # eval update
                    self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                    self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))
                    self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))

                    # Evaluate the performance of current task
                    if val_loader is not None:
                        self.validation(val_loader)

                    # reset
                    losses = [AverageMeter() for l in range(3)]
                    acc = AverageMeter()

            # save halfway point
            self.model.eval()
            self.save_model(model_save_dir, class_only = True)

            # part b
            # dataset tune
            train_dataset.load_dataset(train_dataset.t, train=True)
            train_dataset.load_bic_dataset(post=True)
            train_dataset.append_coreset_ic(post=True)

            # bias correction layer
            self.bic_layers.append([np.arange(self.last_valid_out_dim,self.valid_out_dim),BiasLayer().cuda()])
            self.config['lr'] = self.config['lr'] / 1e2
            self.optimizer, self.scheduler = self.new_optimizer(self.bic_layers[-1][1])
            self.config['lr'] = self.config['lr'] * 1e2
            
            # data weighting
            self.data_weighting(train_dataset)

            # Evaluate the performance of current task
            self.log('Balance Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)
        
            losses = [AverageMeter() for l in range(1)]
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            
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

                    # model update - training data
                    dw_cls = self.dw_k[-1 * torch.ones(y.size()).long()]
                    output = self.forward(x)
                    loss = self.criterion(output, y.long(), dw_cls)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses[0].update(loss.detach(),  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Balance Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for l in range(1)]
                acc = AverageMeter()

            # dataset final
            train_dataset.load_dataset(train_dataset.t, train=True)
            train_dataset.append_coreset(only=False)

        self.model.eval()
        self.past_tasks.append(np.arange(self.last_valid_out_dim,self.valid_out_dim))
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher

        # new teacher
        teacher = TeacherBiC(solver=self.model, bic_layers = self.bic_layers)
        self.previous_teacher = copy.deepcopy(teacher)
        self.replay = True

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset_ic(self.memory_size, np.arange(self.last_valid_out_dim), teacher)

        try:
            return batch_time.avg
        except:
            return None

    def save_model(self, filename, class_only = False):
        model_state = self.model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving class model to:', filename)
        torch.save(model_state, filename + 'class.pth')
        self.log('=> Save Done')
        if not class_only:
            for tc in range(1,self.task_count):
                tci = tc + 1
                model_state = self.bic_layers[tc-1][1].state_dict()
                for key in model_state.keys():  # Always save it to cpu
                    model_state[key] = model_state[key].cpu()
                    torch.save(model_state, filename + 'BiC-' + str(tci+1) + '.pth')

    def load_model(self, filename, class_only = False):
        self.model.load_state_dict(torch.load(filename + 'class.pth'))
        self.log('=> Load Done')
        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()

        if not class_only:
            bic_layers = []
            for tc in range(1,self.task_count+1):
                tci = tc + 1
                bic_layers.append([self.tasks[tc],BiasLayer().cuda()])
                bic_layers[tc-1][1].load_state_dict(torch.load(filename + 'BiC-' + str(tci+1) + '.pth'))
        
        self.bic_layers = bic_layers

# Teacher for BiC
class TeacherBiC(nn.Module):

    def __init__(self, solver, bic_layers):

        super().__init__()
        self.solver = solver
        self.bic_layers = bic_layers

    def generate_scores(self, x, allowed_predictions=None, threshold=None):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)

        # forward with bic
        for i in range(len(self.bic_layers)):
            y_hat[:,self.bic_layers[i][0]] = self.bic_layers[i][1](y_hat[:,self.bic_layers[i][0]])
        y_hat = y_hat[:, allowed_predictions]
        ymax, y = torch.max(y_hat, dim=1)

        # set model back to its initial mode
        self.train(mode=mode)
        
        return y_hat, y

    def generate_scores_pen(self, x):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)

        # set model back to its initial mode
        self.train(mode=mode)

        return y_hat

def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot=one_hot.scatter(1,target.long().view(-1,1),1.)
    return one_hot


class CoresetDataset(data.Dataset):
    def __init__(self,data,target,class_mapping,transform=None):
        self.data=data
        self.targets=target
        self.transform=transform
        self.class_mapping=class_mapping
    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            if simple:
                img = self.simple_transform(img)
            else:
                img = self.transform(img)

        return img, self.class_mapping[target]
    
    def __len__(self):
        return len(self.data)



class ABD_Coreset(LWF):

    def __init__(self, learner_config):
        super(ABD_Coreset, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.kd_criterion = nn.MSELoss(reduction="none")
        
    def update_model(self, inputs, targets, target_KD = None):

        
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
        if target_KD is not None:

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
        if target_KD is not None:

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



    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        if len(train_dataset.coreset[0])>0:
            print(len(train_dataset.coreset[0]),len(train_dataset.coreset[1]))
            coreset_dataset=CoresetDataset(train_dataset.coreset[0],train_dataset.coreset[1],train_dataset.class_mapping,transform=train_dataset.transform)
            coreset_train_loader = data.DataLoader(coreset_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(1))
        else:
            coreset_train_loader=None
            coreset_dataset=None
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # train
        if need_train:
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()

            # data weighting
            self.data_weighting(train_dataset,coreset=coreset_dataset)
            
            # Evaluate the performance of current task
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)
        
            losses = [AverageMeter() for l in range(3)]
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
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
                    
                    # if KD
                    if self.replay:
                        allowed_predictions = list(range(self.last_valid_out_dim))
                        y_hat, _ = self.previous_teacher.generate_scores(x, allowed_predictions=allowed_predictions)
                    else:
                        y_hat = None

                    if coreset_train_loader is not None:
                        try:
                            old_data, old_label = next(train_iter)
                            old_data=old_data.to(x.device)
                            old_label=old_label.to(y.device)
                        except:
                            train_iter = iter(coreset_train_loader)
                            old_data, old_label = next(train_iter)
                            old_data=old_data.to(x.device)
                            old_label=old_label.to(y.device)
                        print(old_label.unique())
                        x=torch.cat((x,old_data),0)
                        y=torch.cat((y,old_label),0)
                    # model update - training data
                    loss, loss_class, loss_distill, output= self.update_model(x, y, y_hat)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses[0].update(loss,  y.size(0)) 
                    losses[1].update(loss_class,  y.size(0)) 
                    losses[2].update(loss_distill,  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))
                self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for l in range(3)]
                acc = AverageMeter()

        self.model.eval()
        self.past_tasks.append(np.arange(self.last_valid_out_dim,self.valid_out_dim))
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        # for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)
        if len(self.config['gpuid']) > 1:
            self.previous_linear = copy.deepcopy(self.model.module.last)
        else:
            self.previous_linear = copy.deepcopy(self.model.last)
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher

        self.replay = True
        try:
            return batch_time.avg
        except:
            return None
