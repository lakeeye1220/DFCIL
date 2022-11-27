import torch
from tqdm import tqdm
from learners.wgan.resnet import Discriminator,Generator
from learners.wgan.utils import cal_grad_penalty, d_wasserstein, sample_normal, g_wasserstein
import torchvision
import matplotlib.pyplot as plt
import numpy

def main(args):
    beta1=0.5
    beta2=0.999
    betas_g = [beta1, beta2]
    eps_ = 1e-6
    NUM_CLASSES=100

    generator = Generator(128,"N/A",32,64,False,["N/A"],"W/O",NUM_CLASSES,"ortho","N/A",False).cuda()

    gen_opt = torch.optim.Adam(params=generator.parameters(),
                                                    lr=0.0002,
                                                    betas=betas_g,
                                                    weight_decay=0.0,
                                                    eps=eps_)
                                                    
    discriminator = Discriminator(32,64,False,False,["N/A"],"W/O","W/O","N/A",False,NUM_CLASSES,"ortho","N/A",False)
    discriminator.cuda()
    beta1=0.5
    beta2=0.999
    betas_g = [beta1, beta2]
    eps_ = 1e-6
    discriminator_opt = torch.optim.Adam(params=discriminator.parameters(),
                                                    lr=0.0002,
                                                    betas=betas_g,
                                                    weight_decay=0.0,
                                                    eps=eps_)
    
    epochs=10000

    dataset=torchvision.datasets.CIFAR100(root="/data/cifar100",train=False,download=True,transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),(0.2673342858792409, 0.25643846291708816, 0.2761504713256834))]
            #torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    ))

    train_dataloader=torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True,num_workers=4,drop_last=True)

    generator.train()
    discriminator.train()
    wgan_bsz=64
    for epoch in tqdm(range(epochs)):
        # get real image
        # train discriminator
        for p in discriminator.parameters():
            p.requires_grad = True
        for p in generator.parameters():
            p.requires_grad = False
        for step_index in range(5): # d_updates_per_step
            try:
                (real_images, real_labels) = next(train_iter)
            except:
                train_iter = iter(train_dataloader)
                (real_images, real_labels) = next(train_iter)
            real_images = real_images.cuda()
            real_labels = real_labels.cuda()
            # train discriminator
            discriminator_opt.zero_grad()
            zs = sample_normal(batch_size=wgan_bsz, z_dim=128, truncation_factor=-1, device='cuda')
            y_fake = torch.randint(low=0, high=NUM_CLASSES, size=(wgan_bsz, ), dtype=torch.long, device='cuda')
            fake_images = generator(zs,y_fake, eval=False)
            real_dict = discriminator(real_images, real_labels)
            y_fake = torch.randint(low=0, high=NUM_CLASSES, size=(wgan_bsz, ), dtype=torch.long, device='cuda')
            fake_dict = discriminator(fake_images, y_fake, adc_fake=False)
            dis_loss = d_wasserstein(real_dict["adv_output"], fake_dict["adv_output"])

            gp_loss = cal_grad_penalty(real_images=real_images,
                                                real_labels=real_labels,
                                                fake_images=fake_images,
                                                discriminator=discriminator,
                                                device='cuda')
            dis_acml_loss=dis_loss + 10.0 * gp_loss
            dis_acml_loss.backward()
            discriminator_opt.step()
        
        for p in discriminator.parameters():
            p.requires_grad = False
        for p in generator.parameters():
            p.requires_grad = True
        # train generator
        generator.train()
        gen_opt.zero_grad()
        
        zs = sample_normal(batch_size=wgan_bsz, z_dim=128, truncation_factor=-1, device='cuda')
        y_fake = torch.randint(low=0, high=NUM_CLASSES, size=(wgan_bsz, ), dtype=torch.long, device='cuda')
        fake_images = generator(zs,y_fake)
        fake_dict = discriminator(fake_images, y_fake)
        gen_acml_loss = g_wasserstein(fake_dict["adv_output"])
        gen_acml_loss.backward()
        gen_opt.step()
        discriminator_opt.zero_grad()
        if epoch % 100 == 0:
            print(f"Epoch: {epoch:5d}, G Loss: {gen_acml_loss:.3e} D gp_loss:{gp_loss:.3e} D Loss: {dis_loss:.3e}")
            # save fake images
            grid_fake=torchvision.utils.make_grid(fake_images,normalize=True).cpu()
            torchvision.utils.save_image(grid_fake,f"fake_{epoch}.png")
        
        if epoch % 1000 == 0:
            torch.save(generator.state_dict(),f"generator_{epoch}.pth")
            torch.save(discriminator.state_dict(),f"discriminator_{epoch}.pth")
            

if __name__ == "__main__":
    main(None)
