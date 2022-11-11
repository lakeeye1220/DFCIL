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
import numpy as np
from torch_ema import ExponentialMovingAverage
from ema_pytorch import EMA


class ISCF(NormalNN):
    def __init__(self, learner_config):
        super(ISCF, self).__init__(learner_config)
        self.inversion_replay = False
        self.previous_teacher = None
        self.dw = self.config['DW']
        self.device = 'cuda' if self.gpu else 'cpu'
        self.power_iters = self.config['power_iters']
        self.deep_inv_params = self.config['deep_inv_params']
        self.mse_criterion = nn.MSELoss(reduction="none")
        self.dataset=self.config['dataset']
        self.fakegt_idx = self.config['fakegt_idx']
        self.acc_fakegt_idx = self.config['acc_fakegt_idx']
        self.prototype = []

        #reparameterization
        if self.config['reparam']:
            self.m = torch.empty((self.batch_size,1),requires_grad=True,device=self.device)
            self.v = torch.empty((self.batch_size,1),requires_grad=True,device=self.device)

            torch.nn.init.normal_(self.m, 1.0, 1)
            torch.nn.init.normal_(self.v, 1.0, 1)

        # generator parameters
        self.generator = self.create_generator()
        #self.discriminator = self.create_discriminator()
        if self.config['reparam']:
            #params = list(self.v)+list(self.m)
            self.generator_optimizer = Adam(params=self.generator.parameters(), lr=self.deep_inv_params[0])
            self.reparam_optimizer = Adam(params=[self.m]+[self.v], lr=0.001)
        else:
            self.generator_optimizer = Adam(params = self.generator.parameters(),lr=self.deep_inv_params[0])
        self.beta = self.config['beta']

        if self.config['ema']:
            #self.ema_model = ExponentialMovingAverage(self.model.parameters(),decay=0.995,use_num_updates=True) #use_num_updates option True로 줘야하는지 말아야하는지, 또 ema는 learning rate가 작아야 좋은게 아닌지
            #self.ema_model.to('cuda')
          
            self.ema_model =EMA(self.model,beta = 0.9999,              # exponential moving average factor
                update_after_step = 100,    # only after this number of .update() calls will it start updating
                update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
                )
            print("define the ema model!")

            self.ema_model.to('cuda')

        else:
            self.ema_model=None


        # repeat call for generator network
        if self.gpu:
            self.cuda_gen()

        
        #SPKD loss definition
        self.md_criterion = SP(reduction='none')
    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None,FT_flag=False):
        
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
            losses = [AverageMeter() for i in range(7)]
            acc = AverageMeter()
            accg = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            self.save_gen = False
            self.save_gen_later = False
            self.acc_fakegt_idx=[]
            self.fakegt_idx = []
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
                        self.acc_fakegt_idx.append(y_replay.tolist())
                        if self.lastbs_img: #true
                            if self.epoch == int(self.config['schedule'][-1]-1):
                                self.fakegt_idx.append(y_replay.tolist())
                        else: # false
                            if self.epoch == int(self.config['schedule'][-1]-1):
                                self.fakegt_idx.append(y_replay.tolist())
                    else:
                        x_replay = None
                        y_replay = None
                        #print("len of self.fake_idx datafree.py 98", len(self.fakegt_idx))

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
                    loss, loss_class, loss_kd,loss_middle,loss_balancing, loss_diag,emb_reg, output= self.update_model(x_com, y_com, x_replay,y_hat_com, dw_force = dw_cls, kd_index = np.arange(len(x), len(x_com)))

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
                    losses[5].update(loss_diag, y_com.size(0))
                    losses[6].update(emb_reg,y_com.size(0))
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.4e} | CE Loss {lossb.avg:.4e} | KD Loss {lossc.avg:.4e} | SP Loss {lossd.avg:.4e} | WEQ Reg {losse.avg:.4e} | Diag Loss {lossf.avg:.4e} | Emb Reg {lossg.avg:.4e}'.format(loss=losses[0],lossb=losses[1],lossc=losses[2],lossd=losses[3],losse=losses[4], lossf = losses[5], lossg=losses[6]))
                self.log(' * Train Acc {acc.avg:.3f} | Train Acc Gen {accg.avg:.3f}'.format(acc=acc,accg=accg))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for i in range(7)]
                acc = AverageMeter()
                accg = AverageMeter()


        self.model.eval()
        self.last_last_valid_out_dim = self.last_valid_out_dim
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher

        if self.config['ema'] and self.previous_teacher is not None:
            #self.model.train()
            #self.ema_model.copy_to(self.model.parameters())
            #self.ema_model.copy_to(self.previous_teacher.solver.parameters())
            print("self.model now contains the averaged weights!! and EMA model accuracy is below : ")
            self.previous_teacher.solver = copy.deepcopy(self.ema_model)
            self.validation(val_loader)

                #self.ema_accuracies.append(ema_accuracy.item())
                #print('Before EMA accuracy:%.3f, After EMA accuracy:%.3f' % (accuracy,ema_accuracy))
                #pd.DataFrame(self.ema_accuracies).to_csv(self.filename+"/EMA_top1_acc.csv",header=False,index=False)
        
        # define the new model - current model 
        if (self.out_dim == self.valid_out_dim) or (self.dataset== 'TinyImageNet100' and self.valid_out_dim==100): need_train = False
        #prototype 넘겨주는 부분
        if self.config['prototype']:
            self.prototype.clear()
            for i, (x, y, task)  in enumerate(train_loader):
                if self.gpu:
                    x =x.cuda()
                    y = y.cuda()
                # data replay
                if self.inversion_replay:
                    x_replay, y_replay, y_replay_hat = self.sample(self.previous_teacher, self.batch_size, self.device)
                else:
                    x_replay = None
                    y_replay = None
                    #print("len of self.fake_idx datafree.py 98", len(self.fakegt_idx))

                if self.inversion_replay:
                    x_com, y_com = self.combine_data(((x, y),(x_replay, y_replay)))
                else:
                    x_com, y_com = x, y
                with torch.no_grad():
                    logits_x_com=self.model(x_com)
                    #print("logits x com shape : ",logits_x_com.shape)
                self.prototype.append(logits_x_com)

        if self.previous_teacher is not None:
            if self.config['reparam']:
                self.previous_teacher = Teacher(solver=copy.deepcopy(self.model), previous_teacher = self.previous_previous_teacher, generator=self.generator,gen_opt = self.generator_optimizer,proto=self.prototype,disc_opt=self.optimizer, reparam_opt = self.reparam_optimizer,m = self.m, v = self.v, img_shape = (-1, train_dataset.nch,train_dataset.im_size, train_dataset.im_size), iters = self.power_iters, deep_inv_params = self.deep_inv_params, class_idx = np.arange(self.valid_out_dim), train = need_train, config = self.config) 
            else:
                self.previous_teacher = Teacher(solver=copy.deepcopy(self.model), previous_teacher =self.previous_previous_teacher, generator=self.generator, gen_opt = self.generator_optimizer,proto=self.prototype,disc_opt=self.optimizer,reparam_opt = None ,m=None,v=None, img_shape = (-1, train_dataset.nch,train_dataset.im_size, train_dataset.im_size), iters = self.power_iters, deep_inv_params = self.deep_inv_params, class_idx = np.arange(self.valid_out_dim), train = need_train, config = self.config)
        else:
            if self.config['reparam']:
                self.previous_teacher = Teacher(solver=copy.deepcopy(self.model), previous_teacher = None, generator=self.generator,gen_opt = self.generator_optimizer,proto=self.prototype,disc_opt=self.optimizer, reparam_opt = self.reparam_optimizer,m = self.m, v = self.v, img_shape = (-1, train_dataset.nch,train_dataset.im_size, train_dataset.im_size), iters = self.power_iters, deep_inv_params = self.deep_inv_params, class_idx = np.arange(self.valid_out_dim), train = need_train, config = self.config) 
            else:
                self.previous_teacher = Teacher(solver=copy.deepcopy(self.model), previous_teacher = None, generator=self.generator,gen_opt = self.generator_optimizer,proto=self.prototype,disc_opt=self.optimizer, reparam_opt = None,m=None,v=None ,img_shape = (-1, train_dataset.nch,train_dataset.im_size, train_dataset.im_size), iters = self.power_iters, deep_inv_params = self.deep_inv_params, class_idx = np.arange(self.valid_out_dim), train = need_train, config = self.config)
        
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

        if self.config['reparam']:
            m_np = self.m.detach().cpu().numpy()
            v_np = self.v.detach().cpu().numpy()
            np.save(filename+'m_np',m_np)
            np.save(filename+'v_np',v_np)

    def load_model(self, filename):
        self.generator.load_state_dict(torch.load(filename + 'generator.pth'))
        if self.gpu:
            self.generator = self.generator.cuda()
        self.generator.eval()
        super(ISCF, self).load_model(filename)
        
        if self.config['reparam']:
            m_np = np.load(filename+'m_np.npy')
            v_np = np.load(filename+'v_np.npy')

            self.m = torch.from_numpy(m_np).cuda()
            self.v = torch.from_numpy(v_np).cuda()

    def create_generator(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        if self.config['cgan']:
            generator = models.__dict__[cfg['gen_model_type']].__dict__[cfg['gen_model_name']](bn=False,cgan=True)
        else:
            generator =  models.__dict__[cfg['gen_model_type']].__dict__[cfg['gen_model_name']]()
        return generator
    
    def create_discriminator(self):
        cfg = self.config
        discriminator =  models.__dict__[cfg['gen_model_type']].__dict__[cfg['gen_model_name']]()
        return discriminator

    def off_diagonal(self,x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

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

    def update_model(self, inputs, targets, x_fake, target_scores = None, dw_force = None, kd_index = None):
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

        if self.config['largeMarginLoss']:
            logits_lms = self.model(inputs, lsm=True,target=targets)
            print("logits_lms shape :",logits_lms.shape)

        #print("datafree.py 259 logits : ",logits)
        
        # classification 
        class_idx = np.arange(self.batch_size) # real
        if self.inversion_replay:
            # local classification - LCE loss: the logit dimension is from last_valid_out_dim to valid_out_dim
            if self.config['largeMarginLoss']:
                loss_class = self.criterion(logits_lms[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) 
            else:
                loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) 
            #print("index logits : ",logits[class_idx,self.last_valid_out_dim:self.valid_out_dim])
            # ft classification  
            if len(self.config['gpuid']) > 1:
                if self.config['largeMarginLoss']:
                    loss_class += self.criterion(logits_lms[:self.last_valid_out_dim], targets.long(), dw_cls) 
                else:
                    loss_class += self.criterion(self.model.module.last(logits_pen.detach()), targets.long(), dw_cls)
            else:
                if self.config['largeMarginLoss']:
                    loss_class += self.criterion(logits_lms[:self.last_valid_out_dim], targets.long(), dw_cls)
                else:
                    loss_class += self.criterion(self.model.last(logits_pen.detach()), targets.long(), dw_cls)
        
        #first task local classification when we do not use any synthetic data     
        else:
            if self.config['largeMarginLoss']:
                loss_class = self.criterion(logits_lms[class_idx], targets[class_idx].long(), dw_cls[class_idx])
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

        if self.config['hyper_norm']:
            cur_weights=self.model.last.weight[:self.valid_out_dim,:]
            cur_bias=self.model.last.bias[:self.valid_out_dim].unsqueeze(-1)
            #cur_params=torch.cat([cur_weights,cur_bias],dim=1)
            #l2_norm = torch.norm(cur_weights,2)
            #emb_regularizer = self.mse_criterion(l2_norm,torch.ones((1,)).to(self.device)) + l2_norm
            emb_regularizer = self.mse_criterion(torch.sum(F.normalize(cur_weights,p=2,dim=1)),torch.ones((1,)).to(self.device))*0.001
            #embedding = self.model.forward(inputs,pen=True)
            #emb_regularizer = self.mse_criterion(torch.sum(F.normalize(embedding,p=2,dim=1)),torch.ones((1,)).to(self.device)) *0.0001
            #if self.previous_teacher:
                #emb_regularizer = self.mse_criterion(torch.sum(F.normalize(prev_embedding,p=2,dim=1)),torch.ones((1,)).to(self.device))+self.mse_criterion(torch.sum(F.normalize(embedding,p=2,dim=1)),torch.ones((1,)).to(self.device)) *0.001 #current embedding 1 
                #with torch.no_grad():
                #    prev_embedding = self.previous_teacher.solver.forward(inputs,pen=True)
                #emb_regularizer += nn.CosineEmbeddingLoss()(F.normalize(embedding,p=2,dim=1),F.normalize(prev_embedding,p=2,dim=1),torch.ones(inputs.shape[0]).to(self.device))
            #print("embed reg:",emb_regularizer)
        else:
            emb_regularizer=torch.zeros((1,),requires_grad=True).cuda()
            
            

        if self.config['diag']:
            if self.previous_teacher:
                with torch.no_grad():
                    logits_bnprevpen = self.previous_teacher.solver.forward(inputs,bn_normalize=True)
                logits_bncurrentpen = self.model.forward(inputs,bn_normalize=True)
                #print("logits_prevpen : ",logits_bnprevpen.shape, "value : ",logits_bnprevpen)
                #print("logits_prevpen : ",logits_bncurrentpen.shape, "value : ",logits_bncurrentpen)

                off_diag_lambda = 0.0051
                #print("outputs_N ",outputs_N)
                outputs_P = torch.transpose(logits_bnprevpen,0,1)
                outputs_N = logits_bncurrentpen
            
                feature_matrix = torch.matmul(outputs_P,outputs_N).cuda()
                on_diag = torch.diagonal(feature_matrix).add_(-1).pow_(2).sum()
                off_diag = self.off_diagonal(feature_matrix).pow_(2).sum()
                loss_diag = on_diag+off_diag_lambda*off_diag
            else:
                loss_diag = torch.zeros((1,),requires_grad=True).cuda()
        else:
            loss_diag = torch.zeros((1,),requires_grad=True).cuda()



        # calculate the 5 losses - LCE + SPKD + LKD + FT + WEQ, loss_class include the LCE and FT losses
        total_loss = loss_class + loss_lkd + loss_sp + weq_regularizer + emb_regularizer

        self.optimizer.zero_grad()
        total_loss.backward()
        # step
        self.optimizer.step()
        if self.config['ema'] and self.previous_teacher is not None:
            #self.ema_model.update(self.model.parameters())
            #self.ema_model.update(self.model.parameters())
            self.ema_model.update()
        return total_loss.detach(), loss_class.detach(), loss_lkd.detach(), loss_sp.detach(), weq_regularizer.detach(), loss_diag.detach(), emb_regularizer.detach(),logits