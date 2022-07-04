from inversion_model.inversion_eeil import EEILmodel
from inversion_model.inversion_iCaRL import iCaRLmodel
from inversion_model.inversion_bic import bicmodel
from ResNet import resnet18_cbam, resnet34_cbam
import torch
import parser
import argparse

parser = argparse.ArgumentParser(description='bic + NI, cifar100')

parser.add_argument('--seed', type=int, default=777, help="seed")

parser.add_argument('--numclass', type=int, default=20)
parser.add_argument('--img_size', type=int, default=32,help="image size")
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--task_size', type=int, default=20, help='dataset balancing')
parser.add_argument('--mem_size', type=int, default=2000, help="size of memory for replay")
parser.add_argument('--epochs', type=int, default=1,help="traning epochs per each tasks")
parser.add_argument('--lr', type=float, default=1.0, help="start learning rate per each task")
parser.add_argument('--prefix',type=str,default="Buffer_",help="directory name ")
parser.add_argument('--dataset_path',type=str,default="./data/dataset/",help="dataset directory name ")
parser.add_argument('--model',type=str,default="bic",help="directory name ")
parser.add_argument('--eeil_aug',type=bool,default=False,help="Apply EEIL Aug")
parser.add_argument('--lr_steps', help='lr decaying epoch determination', default=[48,62,100],
                        type=lambda s: [int(item) for item in s.split(',')])
parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decaying rate')
parser.add_argument('--momentum', default='0.9', type=float, help='momentum')

#parser=create_args()
args = parser.parse_args()

configs=vars(args)

torch.manual_seed(args.seed)
feature_extractor=resnet34_cbam()

model_dict={
    'icarl':iCaRLmodel,
    'eeil':EEILmodel,
    'bic':bicmodel
}
model=model_dict[configs['model']](feature_extractor,configs)
##options are below task5, task 20
'''
####### task5########
numclass=20
feature_extractor=resnet34_cbam()
img_size=32
batch_size=128
task_size=5
memory_size=2000
epochs=70
learning_rate=2.0

model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)

###task 20###
numclass=5
feature_extractor=resnet34_cbam()
img_size=32
batch_size=128
task_size=20
memory_size=2000
epochs=70
learning_rate=2.0

model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)

'''

for i in range(int(100/args.task_size)):
    print("*********** task number: ",i,"**********")
    model.beforeTrain(task_id=i)
    accuracy=model.train()
    model.afterTrain(accuracy)
