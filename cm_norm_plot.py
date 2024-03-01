import torch
import torchvision.transforms as TF
from module import B
from iscf_module import ISCF_ResNet
from cl_lite.head.dynamic_simple import DynamicSimpleHead
import cl_lite.backbone as B
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

parser = argparse.ArgumentParser(description='Confusion matrix and weight norm visualization')
parser.add_argument('--dataset', type=str, default='imagenet100', choices=['cifar100','imagenet100','imagenet1000'])
parser.add_argument('--arch', type=str,default='iscf',choices=['rdfcil','iscf'])
parser.add_argument('--num_classes', type=int,default=100,choices=[100,1000]) #CIFAR100이랑 ImageNet100은 100으로 무조건 고정!
parser.add_argument('--file_path', type=str, default='./lightning_logs/version_508_imnet_5task_54.64')
#parser.add_argument('--file_path', type=str, default='./lightning_logs/imnet100_version_675_rdfcil_5task_49.44')
#parser.add_argument('--file_path', type=str, default='./lightning_logs/version_451_imnet_10task_45.18')
#parser.add_argument('--file_path', type=str, default='./lightning_logs/imnet100_version_430_rdfcil_10task_40.7')
#parser.add_argument('--file_path', type=str, default='./lightning_logs/version_507_imnet_20task_32.90')
parser.add_argument('--data_root', type=str, default='./data')
parser.add_argument('--num_tasks', type=int,default=5,choices=[5,10,20])
parser.add_argument('--split_size', type=int,default=20,choices=[20,10,5])
parser.add_argument('--init_task_splits', type=int,default=0,choices=[0,50])
parser.add_argument('--bs',type=int, default=128)

args = parser.parse_args()


def visualize_confusion_matrix(model,num_classes,val_loader,valid_out_dim,task_num,class_idx,file_path):
    #plot the confusion matrix
    cm=validation(val_loader, model=model, valid_out_dim=valid_out_dim, task=task_num,class_index=class_idx, verbal = True, confusion_mat=True,pseudo_label=False)
    if cm.shape[0]<valid_out_dim:
        cm1 = np.pad(cm, ((0,valid_out_dim-cm.shape[0]),(0,valid_out_dim-cm.shape[1])), 'constant', constant_values=0)
    else:
        cm1=cm
    
    np.save(os.path.join(file_path,'{}_task_confusion_matrix.npy'.format(str(task_num))), cm1)
    plt.figure()
    plt.matshow(cm1, cmap='viridis')
    #plt.colorbar()
    plt.savefig(os.path.join(file_path,'{}_task_confusion_mat.pdf'.format(str(task_num))),bbox_inches='tight')
    plt.close()
    print(str(task_num)+'-th confusion matrix completed!!!')


def visualize_weight(model,file_path,task_num,valid_out_dim):
    #plot the L1-weight norm per each task 
    class_norm=[]
    weights = []
    biases = []
    class_norm = []

    for i in range(len(model.head.classifiers)):
        if model.head.classifiers[i].bias !=None:
            biases.append(model.head.classifiers[i].bias.unsqueeze(-1))
    if biases != []:
        bias = torch.cat(biases)
        #print("bias shape : ",bias.shape)
    else:
        bias=None
    weight = model.head.embeddings

    #print("weight shape : ",weight.shape)
    if bias != None:
        weight=torch.cat((weight,bias),dim=1)
    for i in range(valid_out_dim):
        class_norm.append(torch.norm(weight[i]).item())

    plt.figure()
    classes=np.arange(valid_out_dim)
    plt.scatter(classes,class_norm)
    plt.xlabel('Class Index')
    plt.ylabel('Weight Norm')
    plt.xlim(0,weight.shape[0])
    plt.savefig(os.path.join(file_path,'{}_task_class_norm.pdf'.format(task_num)),bbox_inches='tight')
    np.savetxt(os.path.join(file_path,'{}_task_class_norm.csv'.format(task_num)), class_norm, delimiter=",", fmt='%.2f')


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
    else:
        class_order =  [23, 8, 11, 7, 48, 13, 1, 91, 94, 54, 16, 63, 52, 41, 80, 2, 47, 87, 78, 66, 19, 6, 24, 10, 59, 30, 22, 29, 83, 37, 93, 81, 43, 99, 86, 28, 34, 88, 44, 14, 84, 70, 4, 20, 15, 21, 31, 76, 57, 67, 73, 50, 69, 25, 98, 46, 96, 0, 72, 35, 58, 92, 3, 95, 56, 90, 26, 40, 55, 89, 75, 71, 60, 42, 9, 82, 39, 18, 77, 68, 32, 79, 12, 85, 36, 17, 64, 27, 74, 45, 61, 38, 51, 62, 65, 33, 5, 53, 97, 49]
    for i in range(args.num_tasks):
        current_task = i # 5task: 0 1 2 3 4 
        valid_out_dim=(current_task+1)*args.split_size
        prefix = args.file_path+'/task_{}/checkpoints/'.format(current_task)
        if args.dataset.startswith("imagenet"):
            backbone = B.resnet.resnet18()
        else:
            backbone = ISCF_ResNet()
        
        state_dict = torch.load(os.path.join(prefix,"best_acc.ckpt"))['state_dict']

        # dataload
        data_module = DataModule(root=args.data_root, 
                                dataset=args.dataset, 
                                batch_size=args.bs, 
                                num_workers=4,
                                num_tasks=args.num_tasks,
                                class_order=class_order,
                                current_task=current_task,
                                #init_task_splits=init_task_splits
                                )
        data_module.setup()
        if args.arch == 'iscf':
            head = DynamicSimpleHead(num_classes=data_module.num_classes, num_features=backbone.num_features, bias=True)
        else:
            head = DynamicSimpleHead(num_classes=data_module.num_classes, num_features=backbone.num_features, bias=False)
        #print(head)
        for t in range(current_task):
            head.append(args.num_classes//args.num_tasks)

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
        #print(head)

        model_old = [("backbone", backbone), ("head", head)]
        model = deepcopy(torch.nn.Sequential(OrderedDict(model_old))).eval()
        freeze(model)


        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()
        print("len(val_dataloader)",len(val_dataloader))

        visualize_confusion_matrix(model, args.num_classes, val_dataloader,valid_out_dim,current_task,class_order,args.file_path)
        visualize_weight(model,args.file_path,current_task,valid_out_dim)

if __name__ == "__main__":
    main()