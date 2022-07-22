import torch 
import torchvision
from models.generator import Generator


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
    # method to store generated images locally
    local_rank = torch.cuda.current_device()
    images_denorm = denormalize(images.data.clone(), dataset='CIFAR100')
    images = images.data.clone()
    print(targets)

    for id in range(images_denorm.shape[0]):
        class_id = targets[id].item()
        image = images_denorm[id].reshape(3,32,32)
        image_np = images_denorm[id].data.cpu().numpy()
        pil_image = torch.from_numpy(image_np)

        save_pth = './outputs/ICCV2021/DFCIL-{}/{}/{}/dataset/denorm/s{}'.format(args.task_num,args.dataset,args.method,class_id)
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        vutils.save_image(image, save_pth+'{}_out_{}'.format(args.task_num,i) + '.png', normalize=True, scale_each=True, nrow=1)

    for id in range(images.shape[0]):
        class_id = targets[id].item()
        image = images[id].reshape(3,32,32)
        image_np = images[id].data.cpu().numpy()
        pil_image = torch.from_numpy(image_np)

        #save_pth = os.path.join(prefix, 'final_images/all_images/gen/s{}'.format(class_id))
        save_pth = './outputs/ICCV2021/DFCIL-{}/{}/{}/dataset/gen/s{}'.format(args.task_num,args.dataset,args.method,class_id)

        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        vutils.save_image(image, save_pth +'{}_output_{}'.format(args.task_num,i) +'.png', normalize=True, scale_each=True, nrow=1)

generator = Generator().cuda()
generator.load_static_dict(torch.load(args.generator_pth))

model.load_static_dict(

for i in range(args.epoch):
    out = generator(1024)
    target = model(out)
    target=target.max(1)
    save_finalimages(out,targets,i)
