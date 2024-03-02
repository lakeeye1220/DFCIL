import torch
import torch.nn as nn
import torchvision.transforms as TF
import torch.nn.functional as F
from module import B
import cl_lite
#from iscf_module import ISCF_ResNet
from cl_lite.head.dynamic_simple import DynamicSimpleHead
import cl_lite.backbone as B
from cl_lite.backbone.resnet_cifar import CifarResNet
from cl_lite.backbone.resnet import ResNet
import os
import tensorflow as tf
import csv
import torchvision.utils as vutils
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from metric import accuracy, AverageMeter, Timer
from datamodule import DataModule
from collections import OrderedDict
from copy import deepcopy
from cl_lite.nn import freeze
import argparse
import csv

parser = argparse.ArgumentParser(description='feature map and similarity map visualization')
torch.backends.cudnn.enabled = False
# Basic model parameters.
parser.add_argument('--dataset', type=str, default='imagenet100', choices=['cifar100','imagenet100','imagenet1000'])
parser.add_argument('--arch', type=str,default='iscf',choices=['abd','rdfcil','iscf'])
parser.add_argument('--num_classes', type=int,default=100,choices=[100,1000])
#parser.add_argument('--file_path', type=str, default='./lightning_logs/version_508_imnet_5task_54.64')
#parser.add_argument('--file_path', type=str, default='./lightning_logs/imnet100_version_675_rdfcil_5task_49.44')
#parser.add_argument('--file_path', type=str, default='./lightning_logs/version_451_imnet_10task_45.18')
#parser.add_argument('--file_path', type=str, default='./lightning_logs/imnet100_version_430_rdfcil_10task_40.7')
parser.add_argument('--file_path', type=str, default='./lightning_logs/version_507_imnet_20task_32.90')
parser.add_argument('--dataset_dir', type=str, default='./data')
parser.add_argument('--num_tasks', type=int,default=20,choices=[5,10,20])
parser.add_argument('--split_size', type=int,default=5,choices=[20,10,5])
parser.add_argument('--init_task_splits', type=int,default=0,choices=[0,50])
parser.add_argument('--bs',type=int, default=64)

args = parser.parse_args()

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

class ISCF_ResNet(CifarResNet):
    def __init__(self, n=5, nf=16, channels=3, preact=False, zero_residual=True, pooling_config=..., downsampling="stride", final_layer=False, all_attentions=False, last_relu=False, **kwargs):
        super().__init__(n, nf, channels, preact, zero_residual, pooling_config, downsampling, final_layer, all_attentions, last_relu, **kwargs)        
    def forward_feat(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)

        feats_s1, x1 = self.stage_1(x)
        feats_s2, x2 = self.stage_2(x1)
        feats_s3, x3 = self.stage_3(x2)
        x4 = self.stage_4(x3)

        return x4,[x1, x2, x3]

class ISCF_ResNet18(ResNet):
    def __init__(
        self,
        block,
        layers,
        zero_init_residual=True,
        nf=64,
        last_relu=False,
        initial_kernel=3,
        **kwargs
    ):
        super(ISCF_ResNet18, self).__init__(block, layers,zero_init_residual,nf,last_relu,initial_kernel, **kwargs)

    def forward_feat(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)
        x_2 = self.layer2(self.end_relu(x_1))
        x_3 = self.layer3(self.end_relu(x_2))
        x_4 = self.layer4(self.end_relu(x_3))

        return x_4, [x_1, x_2, x_3,x_4]
    
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
        loss = F.mse_loss(norm_G_s,norm_G_t,reduction=self.reduction)
        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

def update_old_model(model):
    model_old = [("backbone", model.backbone), ("head", model.head)]
    model_old = deepcopy(nn.Sequential(OrderedDict(model_old))).eval()
    freeze(model_old)
    #model_old.cuda()
    return model_old

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(
        min=eps
    )

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RKDAngleLoss(nn.Module):
    def __init__(
        self,
        in_dim1: int = 0,
        in_dim2: int = None,
        proj_dim: int = None,
    ):
        super().__init__()

        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim1 if in_dim2 is None else in_dim2

        if proj_dim is None:
            proj_dim = min(self.in_dim1, self.in_dim2)

        self.proj_dim = proj_dim

        self.embed1 = self.embed2 = nn.Identity()
        if in_dim1 > 0:
            self.embed1 = nn.Linear(self.in_dim1, self.proj_dim)
            self.embed2 = nn.Linear(self.in_dim2, self.proj_dim)

    def forward(self, student, teacher):
        # N x C
        # N x N x C
        student, teacher = self.embed1(student), self.embed2(teacher)

        td = teacher.unsqueeze(0) - teacher.unsqueeze(1)
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = student.unsqueeze(0) - student.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.l1_loss(s_angle, t_angle)
        return loss


class RKDDistanceLoss(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d)
        return loss

def viz_lossSP(model,model_old,dataloader, task_num, class_order):
    loss_sp  = 0.0
    sp = SP()
    with torch.no_grad():
        for i, (input, target) in enumerate(dataloader):
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda()
                #print(target)
    with torch.no_grad():
        z,middles = model.backbone.forward_feat(input)
        old_z, old_middles = model_old.backbone.forward_feat(input)
    for i in range(len(middles)):
        loss_sp+=sp(middles[i],old_middles[i])
    loss_sp=loss_sp/len(middles) * 50.0 #*self.hparams.lambda_sp 

    print(task_num,"-task's loss_rkd is : ",loss_sp.item())
    return loss_sp


def viz_lossRKD(model,model_old,dataloader, task_num, class_order):
    loss_rkd = 0.0
    model_old.head.feature_mode = True
    model.head.feature_mode = True
    rkd = RKDAngleLoss(model_old.backbone.num_features, proj_dim=2*model_old.backbone.num_features).cuda()
    for i, (input, target) in enumerate(dataloader):
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()
            output_new = model(input)
            output_old = model_old(input)
            loss_rkd +=rkd(output_old.detach(),output_new.detach()).item()
            print(loss_rkd)
    print(task_num,"-task's loss_rkd is : ",loss_rkd)
    model_old.head.feature_mode = False
    model.head.feature_mode = False
    return loss_rkd


def viz_loss_graph(backbone,inputs,target,task_num, class_order):
    num_rows = 1  # You can adjust the number of rows and columns as needed
    num_cols = 4
    
    #print("target :",target)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    with torch.no_grad():
        z,middles = backbone.forward_feat(inputs)
    for i in range(len(middles)):
        axs[i].matshow(sp(middles[i]).squeeze().cpu().detach(), cmap='viridis')  # 특성 맵 그리기
        axs[i].set_title('Feature Map {}'.format(i))  # 서브플롯 제목 설정

    # Adjust the layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(args.file_path, '{}_taask_similarity_mat.pdf'.format(str(task_num))), bbox_inches='tight')
    plt.close()
    print("ISCF model: {}-th similarity matrix completed!!!".format(task_num))


def accumulate_acc(output, target, task, meter, topk):
    meter.update(accuracy(output, target, topk), len(target))
    return meter

def validation(dataloader, model=None, valid_out_dim=0,task=0,class_index=[], verbal = True, confusion_mat=False,pseudo_label=False):
        #evaluation the model performance, print the top-1 accuracy per each task
        y_true=[]
        y_pred=[]

        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()
        for i, (input, target) in enumerate(dataloader):
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda()
                #target_list = target.list()
                new_target = [class_index.index(i) for i in target.tolist()]
                new_target = torch.tensor(new_target).cuda()
                output = model.forward(input)
                acc = accumulate_acc(output, new_target, task, acc, topk=(1,))
                #if len(target) > 1:
                #    output = model.forward(input)[:, valid_out_dim]
                #    acc = accumulate_acc(output, target-valid_out_dim[0], task, acc, topk=(1,))
            if confusion_mat:
                y_true.append(new_target.detach().cpu())
                y_pred.append(output.argmax(dim=1).detach().cpu())

            if pseudo_label:
                if i == 0:
                    soft_pseudo_label = torch.softmax(output.detach() / 1.0, dim=-1)
                    max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
                    target_ul_list = target.tolist()
                    target_ul_pl_list = hard_pseudo_label.tolist()
                    target_ul_confidence_list = max_probs.tolist()
        if verbal:
            print(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'.format(acc=acc, time=batch_timer.toc()))

        if confusion_mat:
            y_true=torch.cat(y_true, dim=0)
            y_pred=torch.cat(y_pred, dim=0)
            cm = tf.math.confusion_matrix(y_true, y_pred)
            return cm

        if pseudo_label:
            return target_ul_list,target_ul_pl_list,target_ul_confidence_list

def main():
    if args.dataset=='imagenet100':
        class_order = [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]
    elif args.dataset=='cifar100':
        class_order = [23, 8, 11, 7, 48, 13, 1, 91, 94, 54, 16, 63, 52, 41, 80, 2, 47, 87, 78, 66, 19, 6, 24, 10, 59, 30, 22, 29, 83, 37, 93, 81, 43, 99, 86, 28, 34, 88, 44, 14, 84, 70, 4, 20, 15, 21, 31, 76, 57, 67, 73, 50, 69, 25, 98, 46, 96, 0, 72, 35, 58, 92, 3, 95, 56, 90, 26, 40, 55, 89, 75, 71, 60, 42, 9, 82, 39, 18, 77, 68, 32, 79, 12, 85, 36, 17, 64, 27, 74, 45, 61, 38, 51, 62, 65, 33, 5, 53, 97, 49]
    # dataload
    loss_list = []   
    model = None
    model_old = None  
    if args.arch =='rdfcil':
        if args.dataset.startswith("imagenet"):
            from cl_lite.backbone.resnet import BasicBlock
            backbone = B.resnet.resnet18()
            #backbone = ISCF_ResNet18(BasicBlock, [2, 2, 2, 2])
        else:
            backbone = B.resnet_cifar.resnet32()
    else:
        if args.dataset.startswith("imagenet"):
            from cl_lite.backbone.resnet import BasicBlock
            backbone = ISCF_ResNet18(BasicBlock, [2, 2, 2, 2])
        else:
            backbone = ISCF_ResNet()
   
    for i in range(args.num_tasks):
        current_task = i # 5task: 0 1 2 3 4 
        if args.arch=='iscf':
            test_mode_str = 'seen'
        else:
            test_mode_str = 'current'
        data_module = DataModule(root=args.dataset_dir, 
                            dataset=args.dataset, 
                            batch_size=args.bs, 
                            num_workers=4,
                            num_tasks=args.num_tasks,
                            class_order=class_order,
                            current_task=current_task,
                            test_mode = test_mode_str,
                            init_task_splits=args.init_task_splits,
                            )
        data_module.setup()
        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()
        print("len(train_dataloader)",len(train_dataloader))
        print("len(val_dataloader)",len(val_dataloader))
        
        valid_out_dim=(current_task+1)*args.split_size
        last_valid_out_dim = (current_task)*args.split_size
        prefix = args.file_path+'/task_{}/checkpoints/'.format(current_task)
        state_dict = torch.load(os.path.join(prefix,"best_acc.ckpt"))['state_dict']
        if args.arch=='iscf':
            head = DynamicSimpleHead(num_classes=data_module.num_classes, num_features=backbone.num_features, bias=True)
        else:
            kwargs = dict(num_features=backbone.num_features, bias=False)
            head = cl_lite.head.DynamicSimpleHead(**kwargs)

        if args.arch=='iscf':
            heads_num = current_task
        else:
            heads_num = current_task+1

        for t in range(heads_num):
            head.append(args.num_classes//args.num_tasks)
        print(head)
        backbone_state= {}
        head_state = {}
        for k,v in state_dict.items():
            if k.startswith('backbone'):
                backbone_state[k[9:]] = v
            elif k.startswith('head'):
                head_state[k[5:]] = v
        backbone.load_state_dict(backbone_state)
        backbone.eval()
        head.load_state_dict(head_state)
        backbone.cuda()
        head.cuda()
        
        model_temp = [("backbone", backbone), ("head", head)]
        model = deepcopy(torch.nn.Sequential(OrderedDict(model_temp))).eval()
        freeze(model)
        model.cuda()

        if model_old is not None:
            if args.arch=='iscf':
                loss_sp = viz_lossSP(model,model_old,val_dataloader, current_task, class_order)
                loss_list.append(loss_sp.item())
            else:
                loss_rkd = viz_lossRKD(model,model_old,val_dataloader, current_task, class_order)
                loss_list.append(loss_rkd)

        model_old = update_old_model(model)

    with open(args.file_path+'/loss.csv','w') as file:
        writer = csv.writer(file)
        writer.writerow(loss_list)
  
if __name__ == "__main__":
    main()