import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models.generator import Generator
import torchvision.utils as vutils
torch.random.manual_seed(0)
np.random.seed(0)

#task_path='./outputs/pretrained_5t/DFCIL-fivetask/CIFAR100/proposed/models/repeat-1/task-'
#task_path='../OrigianlABD/outputs/Ours/DFCIL-fivetask/CIFAR100/proposed/models/repeat-1/task-'
task_path = '/home/yujin/DFCIL/DFCIL/outputs/ISCF_sm_matrix_loss/DFCIL-fivetask/CIFAR100/iscf/models/repeat-1/task-'
bs = 128
for i in range(5):
    #m_np = np.load(task_path+str(i+1)+'/m_np.npy')
    #v_np = np.load(task_path+str(i+1)+'/v_np.npy')
    #m = torch.from_numpy(m_np)
    #v = torch.from_numpy(v_np)
    m = None
    v = None
    gen_path = task_path + str(i+1) + '/'
    generator = Generator(zdim=1000, in_channel=3, img_sz=32).to('cuda')
    generator.load_state_dict(torch.load(gen_path+'generator.pth'))
    images = generator.sample(1000,i,bs,v,m)
    image = images[0].cpu().detach().numpy()
    image = image.reshape(32,32,3)
    vutils.save_image(images.data.clone(), f'./confusion_task'+str(i+1)+'.png', normalize=True, scale_each=True)
print("save done")

