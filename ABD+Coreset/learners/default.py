from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
import models
from utils.metric import accuracy, AverageMeter, Timer
import copy
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
class NormalNN(nn.Module):
    """
    consider citing the benchmarking environment this was built on top of
    git url: https://github.com/GT-RIPL/Continual-Learning-Benchmark
    @article{hsu2018re,
        title={Re-evaluating continual learning scenarios: A categorization and case for strong baselines},
        author={Hsu, Yen-Chang and Liu, Yen-Cheng and Ramasamy, Anita and Kira, Zsolt},
        journal={arXiv preprint arXiv:1810.12488},
        year={2018}
    }
    """
    def __init__(self, learner_config):

        super(NormalNN, self).__init__()
        self.log = print
        self.config = learner_config
        self.out_dim = learner_config['out_dim']
        self.model = self.create_model()
        self.reset_optimizer = True
        self.overwrite = learner_config['overwrite']
        self.batch_size = learner_config['batch_size']
        self.previous_teacher = None
        self.tasks = learner_config['tasks']
        self.top_k = learner_config['top_k']

        # replay memory parameters
        self.memory_size = self.config['memory']
        self.task_count = 0

        # class balancing
        self.dw = self.config['DW']
        if self.memory_size <= 0:
            self.dw = False

        # distillation
        self.DTemp = learner_config['temp']
        self.mu = learner_config['mu']

        # supervised criterion
        self.criterion_fn = nn.CrossEntropyLoss(reduction="none")
        
        # cuda gpu
        if learner_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        
        # highest class index from past task
        self.last_valid_out_dim = 0 

        # highest class index from current task
        self.valid_out_dim = 0

        # set up schedules
        self.schedule_type = self.config['schedule_type']
        self.schedule = self.config['schedule']

        # initialize optimizer
        self.init_optimizer()
        self.gen_midscore=[]
        self.midscore=[]
        self.previous_teacher=None
        self.close_MID()

    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def visualize_ready_MID(self, train_loader, prev_train_loader ,file_path):
        #plot the confusion matrix
        # real data
        self.mid_path=file_path
        self.mid_trainloader=train_loader
        self.mid_prev_trainloader=prev_train_loader

    def save_mid_score(self,cur_pen,prev_pen,gen_pen,epoch):
        if self.mid_path is not None:
            np.save(os.path.join(self.mid_path,'{}e_cur_pen_mean.npy'.format(epoch)),cur_pen[0].cpu().numpy())
            np.save(os.path.join(self.mid_path,'{}e_cur_pen_std.npy'.format(epoch)),cur_pen[1].cpu().numpy())
            np.save(os.path.join(self.mid_path,'{}e_prev_pen_mean.npy'.format(epoch)),prev_pen[0].cpu().numpy())
            np.save(os.path.join(self.mid_path,'{}e_prev_pen_std.npy'.format(epoch)),prev_pen[1].cpu().numpy())
            if len(gen_pen[0])>0:
                np.save(os.path.join(self.mid_path,'{}e_gen_pen_mean.npy'.format(epoch)),gen_pen[0].cpu().numpy())
                np.save(os.path.join(self.mid_path,'{}e_gen_pen_std.npy'.format(epoch)),gen_pen[1].cpu().numpy())

            mid_score=torch.from_numpy((cur_pen[0]-prev_pen[0])/cur_pen[1]).to(self.device).norm()
            self.midscore.append(mid_score)
            if len(gen_pen[0])>0:
                gen_mid_score=torch.from_numpy((cur_pen[0]-gen_pen[0])/cur_pen[1]).to(self.device).norm()
                self.gen_midscore.append(gen_mid_score)
                       
    
    def get_MID(self):
        # real data
        self.model.eval()
        with torch.no_grad():
            cur_pen=[]
            cur_pen_std=[]
            for batch_idx, values in enumerate(self.mid_trainloader):
                if len(values)==3:
                    inputs, targets, _ = values
                else:
                    inputs, targets = values

                if self.gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                logits_pen = self.model(inputs,pen=True)
                cur_pen.append(logits_pen.mean(dim=[-1]))
                cur_pen_std.append(logits_pen.std(dim=[-1]))
            cur_pen=torch.cat(cur_pen,dim=0)
            cur_pen_std=torch.cat(cur_pen_std,dim=0)
            prev_pen=[]
            prev_pen_std=[]
            for batch_idx, values in enumerate(self.mid_prev_trainloader):
                if len(values)==3:
                    inputs, targets, _ = values
                else:
                    inputs, targets = values
                if self.gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                logits_pen = self.model(inputs,pen=True)
                prev_pen.append(logits_pen.mean(dim=[-1]))
                prev_pen_std.append(logits_pen.std(dim=[-1]))
            prev_pen=torch.cat(prev_pen,dim=0)
            # generate the data
            gen_pen=[]
            gen_pen_std=[]
            if hasattr(self,'sample'):
                for i in range(batch_idx+1):
                    x_replay, y_replay, y_replay_hat = self.sample(self.previous_teacher, len(inputs), self.device)
                    logits_pen = self.model(x_replay,pen=True)
                    gen_pen.append(logits_pen.mean(dim=[-1]))
                    gen_pen_std.append(logits_pen.std(dim=[-1]))
                gen_pen=torch.cat(gen_pen,dim=0)
            # calculate the score
        self.model.train()

        return (cur_pen,cur_pen_std),(prev_pen,prev_pen_std),(gen_pen,gen_pen_std)
    
    def close_MID(self):
        self.mid_path=None
        self.mid_trainloader=None
        self.mid_prev_trainloader=None

    def visualize_confusion_matrix(self, val_loader,file_path,task_num):
        #plot the confusion matrix
        y_true,y_pred=self.validation(val_loader,need_logits=True)
        cm = tf.math.confusion_matrix(y_true, y_pred.argmax(dim=1))
        if cm.shape[0]<self.config['num_classes']:
            cm1 = np.pad(cm, ((0,self.config['num_classes']-cm.shape[0]),(0,self.config['num_classes']-cm.shape[1])), 'constant', constant_values=0)
        else:
            cm1=cm
        
        np.save(os.path.join(file_path,'{}task_confusion_matrix.npy'.format(task_num)), cm1)
        plt.figure()
        plt.matshow(cm1, cmap='viridis')
        #plt.colorbar()
        plt.savefig(os.path.join(file_path,'{}task_confusion_mat.pdf'.format(task_num)),bbox_inches='tight')
        plt.close()
    
    def visualize_marginal_likelihood(self, val_loader,file_path,task_num):
        y_true,y_pred=self.validation(val_loader,need_logits=True)
        softened_y_pred=torch.softmax(y_pred[:,:self.valid_out_dim],dim=1)
        classes=np.arange(self.valid_out_dim)
        marginal_likelihood=softened_y_pred.mean(dim=0)
        plt.figure()
        plt.plot(classes,marginal_likelihood.detach().numpy())
        plt.xlabel('Class Index')
        plt.ylabel('Marginal Likelihood')
        plt.xlim(0,self.valid_out_dim)
        plt.savefig(os.path.join(file_path,'{}task_marginal_likelihood.pdf'.format(task_num)),bbox_inches='tight')
        np.savetxt(os.path.join(file_path,'{}task_marginal_likelihood.csv'.format(task_num)), marginal_likelihood, delimiter=",", fmt='%.2f')
        plt.close()

    def visualize_weight(self,file_path,task_num):
        #plot the L1-weight norm per each task 
        class_norm=[]
        if len(self.config['gpuid'])>1:
            weight=self.model.module.last.weight
            bias=self.model.module.last.bias.unsqueeze(-1)
        else:
            weight=self.model.last.weight
            bias=self.model.last.bias.unsqueeze(-1)
        weight=torch.cat((weight,bias),dim=1)
        for i in range(self.valid_out_dim):
            class_norm.append(torch.norm(weight[i]).item())

        plt.figure()
        classes=np.arange(self.valid_out_dim)
        plt.scatter(classes,class_norm)
        plt.xlabel('Class Index')
        plt.ylabel('Weight Norm')
        plt.xlim(0,weight.shape[0])
        plt.savefig(os.path.join(file_path,'{}task_class_norm.pdf'.format(task_num)),bbox_inches='tight')
        np.savetxt(os.path.join(file_path,'{}task_class_norm.csv'.format(task_num)), class_norm, delimiter=",", fmt='%.2f')
        plt.close()

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
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
        
            losses = AverageMeter()
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
                    
                    # model update
                    loss, output= self.update_model(x, y)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    
                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = AverageMeter()
                acc = AverageMeter()
                
        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        # for eval
        # if self.previous_teacher is not None:
        #     self.previous_previous_teacher = self.previous_teacher

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)

        try:
            return batch_time.avg
        except:
            return None

    def criterion(self, logits, targets, data_weights):
        # calculate the LCE and FT losses with data_weighting
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised 

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        if dw_force is not None:
            dw_cls = dw_force
        elif self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # vanilla KD
        if target_scores is not None:
            if kd_index is None: kd_index = np.arange(len(logits))
            total_loss += self.mu * loss_fn_kd(logits[kd_index], target_scores[kd_index], dw_cls[kd_index], np.arange(self.last_valid_out_dim).tolist(), self.DTemp)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), logits

    def validation(self, dataloader, model=None, task_in = None,  verbal = True, need_logits=False):
        #evaluation the model performance, print the top-1 accuracy per each task
        if model is None:
            model = self.model

        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        y_true=[]
        y_pred=[]

        orig_mode = model.training
        model.eval()
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            if task_in is None:
                output = model.forward(input)[:, :self.valid_out_dim]
                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
            else:
                mask = target >= task_in[0]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]

                mask = target < task_in[-1]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]
                
                if len(target) > 1:
                    output = model.forward(input)[:, task_in]
                    acc = accumulate_acc(output, target-task_in[0], task, acc, topk=(self.top_k,))
            
            if need_logits:
                y_true.append(target.detach().cpu())
                y_pred.append(output.detach().cpu())
        model.train(orig_mode)

        if need_logits:
            y_true=torch.cat(y_true, dim=0)
            y_pred=torch.cat(y_pred, dim=0)
            return y_true, y_pred
        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    # data weighting
    def data_weighting(self, dataset, num_seen=None,coreset=None):

        if dataset.dw:
            # count number of examples in dataset per class
            self.log('*************************\n\n\n')
            if num_seen is None:
                labels = [int(dataset[i][1]) for i in range(len(dataset))]
                labels = np.asarray(labels, dtype=np.int64)
                num_seen = np.asarray([len(labels[labels==k]) for k in range(self.valid_out_dim)], dtype=np.float32)
            if coreset is not None and self.config['name'] not in ['ABD_Coreset','DI_Coreset']:
                add_labels = [int(coreset.targets[i]) for i in range(len(coreset))] #q
                add_labels = np.asarray(add_labels, dtype=np.int64)
                add_num_seen = np.asarray([len(add_labels[add_labels==k]) for k in range(self.valid_out_dim)], dtype=np.float32)
                num_seen = num_seen + add_num_seen

            self.log('num seen:' + str(num_seen))
            
            # in case a zero exists in PL...
            num_seen += 1

            # all seen
            seen = np.ones(self.valid_out_dim + 1, dtype=np.float32)
            seen = torch.tensor(seen)
            seen_dw = np.ones(self.valid_out_dim + 1, dtype=np.float32)
            seen_dw[:self.valid_out_dim] = num_seen.sum() / (num_seen * len(num_seen))
            seen_dw = torch.tensor(seen_dw)

            self.dw_k = seen_dw
            # cuda
            if self.cuda:
                self.dw_k = self.dw_k.cuda()

        else:
            self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
            # cuda
            if self.cuda:
                self.dw_k = self.dw_k.cuda()

    def save_model(self, filename):
        model_state = self.model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving class model to:', filename)
        torch.save(model_state, filename + 'class.pth')
        self.log('=> Save Done')

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename + 'class.pth'))
        self.log('=> Load Done')
        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedulesif self.schedule_type == 'decay':
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    # returns optimizer for passed model
    def new_optimizer(self, model):

        # parse optimizer args
        optimizer_arg = {'params':model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)

        # create schedules
        if self.schedule_type == 'decay':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.schedule, gamma=0.1)

        return optimizer, scheduler

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim)

        return model

    def print_model(self):
        # self.log(self.model)
        self.log('#parameter of model:', self.count_parameter())
    
    def reset_model(self):
        self.model.apply(weight_reset)

    def forward(self, x):
        return self.model.forward(x)[:, :self.valid_out_dim]

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        return out
    
    def add_valid_output_dim(self, dim=0):

        # gradually increasing the classification head dimension
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())   

    def count_memory(self, dataset_size):
        return self.count_parameter() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log("Running on:", device)
        return device

    def pre_steps(self):
        pass

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def accumulate_acc(output, target, task, meter, topk):
    meter.update(accuracy(output, target, topk), len(target))
    return meter

def loss_fn_kd(scores, target_scores, data_weights, allowed_predictions, T=2., soft_t = False):
    #Compute knowledge-distillation (KD) loss given [scores] and [target_scores].
    #Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    #Hyperparameter: temperature


    log_scores_norm = F.log_softmax(scores[:, allowed_predictions] / T, dim=1)
    if soft_t:
        targets_norm = target_scores
    else:
        targets_norm = F.softmax(target_scores[:, allowed_predictions] / T, dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm # * T**2

    return KD_loss

##########################################
#            TEACHER CLASS               #
##########################################

class Teacher(nn.Module):

    def __init__(self, solver):

        super().__init__()
        self.solver = solver

    def generate_scores(self, x, allowed_predictions=None, threshold=None):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        # set model back to its initial mode
        self.train(mode=mode)

        # threshold if desired
        if threshold is not None:
            # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
            y_hat = F.softmax(y_hat, dim=1)
            ymax, y = torch.max(y_hat, dim=1)
            thresh_mask = ymax > (threshold)
            thresh_idx = thresh_mask.nonzero().view(-1)
            y_hat = y_hat[thresh_idx]
            y = y[thresh_idx]
            return y_hat, y, x[thresh_idx]

        else:
            # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
            ymax, y = torch.max(y_hat, dim=1)

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
