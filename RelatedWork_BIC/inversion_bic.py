from re import L
import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from inversion_CIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
from ResNet import resnet34_cbam

import os
import sys
import parser
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))
    def forward(self, x):
        return self.alpha * x + self.beta
    def printParam(self, i):
        print(i, self.alpha.item(), self.beta.item())

class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)
        

    def forward(self, input):
        x,f4,f3,f2,f1,f0 = self.feature(input)
        x = self.fc(x)
        return x,f4,f3,f2,f1,f0 

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self,inputs):
        return self.feature(inputs)

def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).to(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

class bicmodel:
    def __init__(self, feature_extractor, configs):
        super(bicmodel, self).__init__()
        self.configs=configs
        self.numclass=configs['numclass']
        self.learning_rate=configs['lr']
        self.model = network(self.numclass,feature_extractor)
        self.exemplar_set = []
        self.class_mean_set = []
        self.transform = transforms.Compose([#transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.old_model = None
        self.bias_layers=[]
        
        
        self.train_dataset = iCIFAR100(os.path.join(dataset_path,'cifar100'), transform=self.train_transform, download=True,eeil_aug=self.configs['eeil_aug'])
        self.test_dataset = iCIFAR100(os.path.join(dataset_path,'cifar100'), test_transform=self.test_transform, train=False, download=True,eeil_aug=False)
        self.train_transform = transforms.Compose([#transforms.Resize(img_size),
                            #transforms.RandomCrop((32,32),padding=4),
                            #transforms.RandomHorizontalFlip(p=0.5),
                            #transforms.ColorJitter(brightness=0.24705882352941178),
                            #transforms.ToTensor(),
                            #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
        self.test_transform = transforms.Compose([#transforms.Resize(img_size),
                            transforms.ToTensor(),
                            #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.classify_transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                            #transforms.Resize(img_size),
                            transforms.ToTensor(),
                            #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.dataset_path=configs['dataset_path']
        self.train_dataset = iCIFAR100(os.path.join(dataset_path,'cifar100'), transform=self.train_transform, download=True,eeil_aug=self.configs['eeil_aug'])
        self.test_dataset = iCIFAR100(os.path.join(dataset_path,'cifar100'), test_transform=self.test_transform, train=False, download=True,eeil_aug=False)

        self.batchsize = configs['batch_size']
        self.memory_size=configs['mem_size']
        self.task_size=configs['task_size']
        self.epochs=configs['epochs']

        for i in range(self.task_size):
            self.bias_layers.append(BiasLayer().cuda())
        
        
        self.train_loader=None
        self.test_loader=None
        self.filename = None
        self.accuracies = []
        self.prefix = configs['prefix']
        self.best_acc = 0.0
        
    def beforeTrain(self, task_id):
        self.model.eval()
        classes = [self.numclass - self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes,task_id)
        
        if self.numclass>self.task_size:
            self.model.Incremental_learning(self.numclass)
        
        self.model.train()
        self.model.to(device)
        
    def _get_train_and_test_dataloader(self, classes,task_id):
        self.train_dataset.getTrainData(self.old_model,classes,self.filename,self.batchsize,task_id,self.prefix,self.task_size)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader
    
    def train(self):
        accuracy = 0.0
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate,weight_decay=0.00001,momentum=self.configs['momentum'])
        lr_scheduler= optim.lr_scheduler.MultiStepLR(opt, milestones=self.configs['lr_steps'], gamma=self.configs['lr_decay'])
        bias_opt = optim.Adam(self.bias_layers[int(self.numclass/self.task_size) - 1].parameters(), lr=0.001) # 
        for epoch in range(self.epochs):

            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(device), target.to(device)
                #output = self.model(images)
                loss_value = self.stage1(indexs, images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()
                print('\repoch:%3d,step:%2d,loss:%4.3f' % (epoch, step, loss_value.item()),end='')
            if epoch in self.configs['lr_steps']:
                print("--------------------------")
                print("change learning rate:%.3f" % (opt.param_groups[0]['lr']))
                print("--------------------------")
            accuracy = self._test(self.test_loader, 1)
            lr_scheduler.step()
            print("")
            print("***************************************")
            print('* epoch:%3d,normal accuracy:%6.3f *' % (epoch, accuracy))
            print("***************************************")

        return accuracy

    def bias_forward(self, input):
        output_list = []
        iterator = int (self.numclass / self.task_size) # if 100 / 20 = 5 => i = 0,1,2,3,4

        for i in range(iterator):
            bias_layer = self.bias_layers[i]
            out = bias_layer(input[:, task_size*i:task_size(i+1)])
            output_list.append(out)

        return torch.as_tensor(output)


        



        for i in range(layer_selector):
            out.append

    def stage1(self, indexes, imgs, target):
        output,_,_,_,_,_ =self.model(imgs)
        output = self.model.bias_forward(output)
        target = get_one_hot(target, self.numclass)
        output, target = output.to(device), target.to(device)
        T = 2
    
        if self.old_model == None:
            return F.cross_entropy(output, target)
        else:
            alpha = (self.numclass - self.task_size) / self.numclass
            with torch.no_grad()
                old_output,_,_,_,_,_ = self.old_model(imgs)
                old_output = self.model.bias_forward(old_output)
                old_target = F.softmax(old_output[:, :self.numclass-self.task_size]/T, dim=1)
            new_target = F.log_softmax(output[:, :self.numclass-self.task_size]/T, dim=1)
            loss_soft_target = -torch.mean(torch.sum(old_target * new_target, dim=1))
            loss_hard_target = nn.CrossEntropyLoss()(p[:,:self.numclass], label)
            loss = loss_soft_target * T * T + (1-alpha) * loss_hard_target
            return loss

         
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='iCaRL + NaturalInversion')


    feature_extractor = resnet34_cbam()

    

    parser.add_argument('--seed', type=int, default=777, help="seed")

    parser.add_argument('--numclass', type=int, default=20)
    parser.add_argument('--img_size', type=int, default=32,help="image size")
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--task_size', type=int, default=20, help='dataset balancing')
    parser.add_argument('--mem_size', type=int, default=2000, help="size of memory for replay")
    parser.add_argument('--epochs', type=int, default=1,help="traning epochs per each tasks")
    parser.add_argument('--lr', type=float, default=1.0, help="start learning rate per each task")
    parser.add_argument('--prefix',type=str,default="Buffer_",help="directory name ")
    parser.add_argument('--dataset_path',type=str,default="./data",help="dataset directory name ")
    
    parser.add_argument('--lr_steps', help='lr decaying epoch determination', default=[48,62,100],
                            type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decaying rate')
    parser.add_argument('--momentum', default='0.9', type=float, help='momentum')
    
    args = parser.parse_args()
    
    configs=vars(args)

    model = bicmodel(feature_extractor, configs)
    