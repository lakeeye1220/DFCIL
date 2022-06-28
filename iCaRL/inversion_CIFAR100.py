from torchvision.datasets import CIFAR100
import numpy as np
from PIL import Image
import sys
import os
import torch
import sys

sys.path.append('..')
from inversion_model.inversion_eeil import data_augmentation_e2e
from NaturalInversion.NaturalInversion import get_images

class iCIFAR100(CIFAR100):
    def __init__(self,root='./data/',
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False,
                 eeil_aug=False):
        super(iCIFAR100,self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        self.target_test_transform=target_test_transform
        self.test_transform=test_transform
        self.eeil_aug=eeil_aug
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def getTestData(self, classes):
        datas,labels=[],[]
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas,labels=self.concatenate(datas,labels)
        self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
        self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
        print("the size of test set is %s"%(str(self.TestData.shape)))
        print("the size of test label is %s"%str(self.TestLabels.shape))


    def getTrainData(self,old_model,classes,filename,batchsize,task_id,prefix,task_size):
        print("task id : ",task_id)
        datas,labels=[],[]
        prefix = prefix+str(task_size)+'/task_'+str(task_id)
        sys.path.insert(0,os.path.abspath('..'))

        if task_id != 0:
            gen_inputs, inv_labels = get_images(net=old_model,task=task_id,num_classes=classes,bs=batchsize,filename=filename,targets = None,epochs=2000,prefix=prefix,global_iteration=task_id,bn_reg_scale=3,g_lr=0.001,d_lr=0.0005,a_lr=0.05,var_scale=0.001,l2_coeff=0.00001)
            for i in range(0,classes[0]):
                datas.append([])
                for t in range(10):
                    for idx,j in enumerate(inv_labels[t]): 
                        if i==j:
                            datas[i].append(np.array(gen_inputs[t][j]))
            for i in range(0,classes[0]):
                datas[i] = np.reshape(datas[i],(len(datas[i]),32,32,3))
                datas[i] = datas[i].astype(np.uint8)
                datas[i] = torch.tensor(datas[i])
                labels.append(np.full((len(datas[i])),i))

            print(len(datas))
            print(len(datas[0]))
        for label in range(classes[0],classes[1]):
            ##real data
            data=self.data[np.array(self.targets)==label]
            datas.append(data)
            labels.append(np.full((data.shape[0]),label))
        
            
        self.TrainData,self.TrainLabels=self.concatenate(datas,labels)
        if self.eeil_aug:
            print("the train data size of train set is %s"%(str(self.TrainData.shape)))
            print("the train label size of train label is %s"%str(self.TrainLabels.shape))
            print("Conduct EEIL Augmentation")
            self.TrainData,self.TrainLabels=data_augmentation_e2e(self.TrainData,self.TrainLabels)
        print("the train data size of train set is %s"%(str(self.TrainData.shape)))
        print("the train label size of train label is %s"%str(self.TrainLabels.shape))


    def getTrainItem(self,index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]

        if self.transform:
            img=self.transform(img)

        if self.target_transform:
            target=self.target_transform(target)

        return index,img,target

    def getTestItem(self,index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img=self.test_transform(img)

        if self.target_test_transform:
            target=self.target_test_transform(target)

        return index, img, target

    def __getitem__(self, index):
        if self.TrainData!=[]:
            return self.getTrainItem(index)
        elif self.TestData!=[]:
            return self.getTestItem(index)


    def __len__(self):
        if self.TrainData!=[]:
            return len(self.TrainData)
        elif self.TestData!=[]:
            return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]


