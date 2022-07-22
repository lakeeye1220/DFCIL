import torch 
import torchvision
import numpy as np
import argparse
import os
import sys
import math
from PIL import Image
import torchvision.utils as vutils
from models.generator import Generator,GeneratorMed
from models.autoencoder import AutoEncoder
import models.generator as gen
from models.resnet import resnet32
import matplotlib.pyplot as plt

task = 20
def make_grid(tensor, nrow, padding = 2, pad_value : int = 0):
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid

def denormalize(image_tensor, dataset):
    channel_num = 0
    if dataset == 'CIFAR100' or dataset=='CIFAR10':
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        channel_num = 3
    elif dataset == 'tiny-imagenet':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        channel_num = 3

    for c in range(channel_num):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c]*s+m, 0, 1)

    return image_tensor

def save_finalimages(images, targets, num_generations):

    image_tensor = denormalize(images.data.clone(), dataset='CIFAR100')
    grid = make_grid(image_tensor, nrow = 10)
    ndarr = grid.mul(255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    save_pth = './outputs/ICCV2021/DFCIL-twentytask/TinyImageNet/deepinv/dataset/batch_image/task-{}/'.format(task)
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    im.save(save_pth+'denorm_{}.png'.format(num_generations))
    vutils.save_image(images.data.clone(),
                    './outputs/ICCV2021/DFCIL-twentytask/TinyImageNet/deepinv/dataset/batch_image/task-{}/generat_{}.png'.format(task,num_generations),normalize=True, scale_each=True, nrow=10)

    # method to store generated images locally
    local_rank = torch.cuda.current_device()
    images_denorm = denormalize(images.data.clone(), dataset='CIFAR100')
    images = images.data.clone()
    print(targets)

    for id in range(images_denorm.shape[0]):
        class_id = targets[id]
        image = images_denorm[id].reshape(3,64,64)
        image_np = images_denorm[id].data.cpu().numpy()
        pil_image = torch.from_numpy(image_np)

        save_pth = './outputs/ICCV2021/DFCIL-twentytask/TinyImageNet/deepinv/dataset/denorm/s{}/'.format(class_id)
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)


        vutils.save_image(image, save_pth+'5tk_out_{}'.format(i) + '.png', normalize=True, scale_each=True, nrow=1)

    for id in range(images.shape[0]):
        class_id = targets[id]
        image = images[id].reshape(3,64,64)
        image_np = images[id].data.cpu().numpy()
        pil_image = torch.from_numpy(image_np)

        #save_pth = os.path.join(prefix, 'final_images/all_images/gen/s{}'.format(class_id))
        save_pth = './outputs/ICCV2021/DFCIL-twentytask/TinyImageNet/deepinv/dataset/gen/s{}/'.format(class_id)

        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        vutils.save_image(image, save_pth +'5tk_output_{}'.format(num_generations) +'.png', normalize=True, scale_each=True, nrow=1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DFCIL Inversion')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--iters_mi', default=3, type=int, help='number of iterations for model inversion')
    parser.add_argument('--generator_pth', default='./outputs/ICCV2021/DFCIL-twentytask/TinyImageNet/deepinv/models/repeat-1/task-20/generator.pth', type=str, help='path to load weights of the model')
    parser.add_argument('--model_pth', default='./outputs/ICCV2021/DFCIL-twentytask/TinyImageNet/deepinv/models/repeat-1/task-20/class.pth',type=str,help='path to save final inversion images')
    parser.add_argument('--exp_name', default='sample_image',type=str, help='path to save final inversion images')
    parser.add_argument('--epoch', default=1, type=int, help='path to save final inversion images')
    args = parser.parse_args()

    a = torch.load(args.generator_pth)
    #print(a)
    generator = GeneratorMed(1000,3,64).cuda()
    #generator = gen.CIFAR_GEN().cuda()
    #autoencoder = AutoEncoder(kernel_num=512,in_channel=3,img_sz=32,z_size=1024).cuda()
    #generator = autoencoder
    #print(generator)
    generator.load_state_dict(torch.load(args.generator_pth))
    #generator = generator.decoder
    #print(generator)
    model = resnet32(out_dim=200).cuda()
    model.load_state_dict(torch.load(args.model_pth))


    for i in range(args.epoch):
        lst_target = []
        z = torch.randn((args.bs, 1000)).cuda()
        #z = torch.randn((args.bs,3,32,32)).cuda()
        out = generator(z)
        #print(out.shape)
        out=out.cuda()
        target = model(out)

        ##DGR sample###
        #out = generator.sample(args.bs)
        #print(out.shape)
        #out = out.cuda()
        #target = model(out)
        targets = target.max(1)[1]
        print(targets)
        targets = targets.tolist()
        fig = plt.figure(figsize=(8,8))
        plt.xlim(xmin=0,xmax=99)
        plt.xlabel('Class Index')
        plt.ylabel('The Number of Images')
        plt.title('Target distribution of Synthetic Images')
        h = plt.hist(targets,edgecolor='black')
        plt.show()
        '''
        if not os.path.exists('./outputs/ICCV2021/DFCIL-fivetask/CIFAR100/deepinv/dataset/'):
            os.makedirs('./outputs/ICCV2021/DFCIL-fivetask/CIFAR100/deepinv/dataset/')
        plt.savefig('./outputs/ICCV2021/DFCIL-fivetask/CIFAR100/deepinv/dataset/task'+str(task)+'target_histo'+'.png')
        '''
        save_finalimages(out,targets,i)
