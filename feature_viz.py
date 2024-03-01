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

parser = argparse.ArgumentParser(description='feature map and similarity map visualization')
torch.backends.cudnn.enabled = False
# Basic model parameters.
parser.add_argument('--dataset', type=str, default='imagenet100', choices=['cifar100','imagenet100','imagenet1000'])
parser.add_argument('--arch', type=str,default='rdfcil',choices=['abd','rdfcil','iscf'])
parser.add_argument('--num_classes', type=int,default=100,choices=[100,1000])
#parser.add_argument('--file_path', type=str, default='./lightning_logs/version_508_imnet_5task_54.64')
parser.add_argument('--file_path', type=str, default='./lightning_logs/imnet100_version_675_rdfcil_5task_49.44')
#parser.add_argument('--file_path', type=str, default='./lightning_logs/version_451_imnet_10task_45.18')
#parser.add_argument('--file_path', type=str, default='./lightning_logs/imnet100_version_430_rdfcil_10task_40.7')
#parser.add_argument('--file_path', type=str, default='./lightning_logs/version_507_imnet_20task_32.90')
parser.add_argument('--dataset_dir', type=str, default='./data')
parser.add_argument('--num_tasks', type=int,default=5,choices=[5,10,20])
parser.add_argument('--split_size', type=int,default=20,choices=[20,10,5])
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

    def forward(self,fm_t):
        fm_t = fm_t.view(fm_t.size(0),-1)
        G_t = torch.mm(fm_t,fm_t.t())
        norm_G_t = F.normalize(G_t,p=2,dim=1)
        return norm_G_t


def visualize_ISCF_SPmatrix(backbone,inputs,target,task_num, class_order):
    num_rows = 1  # You can adjust the number of rows and columns as needed
    num_cols = 4
    sp = SP()
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

def visualize_ISCF_fmap(backbone,inputs,target,task_num, class_order):
    num_rows = 1  # You can adjust the number of rows and columns as needed
    num_cols = 4
    sp = SP()
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

def visualize_rdfcil_SPmatrix(backbone,inputs,valid_out_dim,task_num,class_order):
    #plot the confusion matrix
    sp = SP()
    with torch.no_grad():
        z,middles = backbone.forward_feat(inputs)

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
    print("RDFCIL model: {}-th similarity matrix completed!!!".format(task_num))

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
    num_classes = args.num_classes
    if args.dataset=='imagenet100':
        class_order = [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]
    elif args.dataset=='cifar100':
        class_order = [23, 8, 11, 7, 48, 13, 1, 91, 94, 54, 16, 63, 52, 41, 80, 2, 47, 87, 78, 66, 19, 6, 24, 10, 59, 30, 22, 29, 83, 37, 93, 81, 43, 99, 86, 28, 34, 88, 44, 14, 84, 70, 4, 20, 15, 21, 31, 76, 57, 67, 73, 50, 69, 25, 98, 46, 96, 0, 72, 35, 58, 92, 3, 95, 56, 90, 26, 40, 55, 89, 75, 71, 60, 42, 9, 82, 39, 18, 77, 68, 32, 79, 12, 85, 36, 17, 64, 27, 74, 45, 61, 38, 51, 62, 65, 33, 5, 53, 97, 49]
    # dataload
        
    if args.arch =='rdfcil':
        if args.dataset.startswith("imagenet"):
            from cl_lite.backbone.resnet import BasicBlock
            #backbone = B.resnet.resnet18()
            backbone = ISCF_ResNet18(BasicBlock, [2, 2, 2, 2])
        else:
            backbone = B.resnet_cifar.resnet32()
    else:
        if args.dataset.startswith("imagenet"):
            from cl_lite.backbone.resnet import BasicBlock
            backbone = ISCF_ResNet18(BasicBlock, [2, 2, 2, 2])
        else:
            backbone = ISCF_ResNet()
    current_task = 0
    data_module = DataModule(root=args.dataset_dir, 
                            dataset=args.dataset, 
                            batch_size=args.bs, 
                            num_workers=4,
                            num_tasks=args.num_tasks,
                            class_order=class_order,
                            current_task=current_task,
                            init_task_splits=args.init_task_splits,
                            )
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    print("len(train_dataloader)",len(train_dataloader))
    print("len(val_dataloader)",len(val_dataloader))

    for i, (inputs,target) in enumerate(train_dataloader):
        sorted_indices = torch.argsort(target, dim=0)
        sorted_batch_inputs = inputs[sorted_indices]
        sorted_batch_targets = target[sorted_indices]
        with torch.no_grad():
            inputs = sorted_batch_inputs.cuda()
            target =  sorted_batch_targets.cuda()
        exit
    print("inputs shape : ",inputs.shape)
    print("target index : ",target.detach())

    for i in range(args.num_tasks):
        current_task = i # 5task: 0 1 2 3 4 
        valid_out_dim=(current_task+1)*args.split_size
        prefix = args.file_path+'/task_{}/checkpoints/'.format(current_task)
        state_dict = torch.load(os.path.join(prefix,"best_acc.ckpt"))['state_dict']
        if args.arch=='iscf':
            head = DynamicSimpleHead(num_classes=data_module.num_classes, num_features=backbone.num_features, bias=True)
        else:
            kwargs = dict(num_features=backbone.num_features, bias=False)
            head = cl_lite.head.DynamicSimpleHead(**kwargs)
        if args.arch=='iscf':
            total_heads_num = current_task
        else:
            total_heads_num = current_task+1
        for t in range(total_heads_num):
            head.append(num_classes//args.num_tasks)
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

        model = torch.nn.Sequential(backbone,head)
        model.cuda()
        freeze(model)
        #print(model)
        #print(head)

        #model_old = [("backbone", backbone), ("head", head)]
        #model = deepcopy(torch.nn.Sequential(OrderedDict(model_old))).eval()
        #freeze(model)

        if args.arch=='iscf':
            visualize_ISCF_SPmatrix(backbone,inputs,target,current_task,class_order)
        else:
            visualize_rdfcil_SPmatrix(backbone,inputs,valid_out_dim,current_task,class_order)
if __name__ == "__main__":
    main()