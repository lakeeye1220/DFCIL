import torch
from torch import autograd
from scipy.stats import truncnorm

def truncated_normal(size, threshold=1.):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values


def sample_normal(batch_size, z_dim, truncation_factor, device):
    if truncation_factor == -1.0:
        latents = torch.randn(batch_size, z_dim, device=device)
    elif truncation_factor > 0:
        latents = torch.FloatTensor(truncated_normal([batch_size, z_dim], truncation_factor)).to(device)
    else:
        raise ValueError("truncated_factor must be positive.")
    return latents


def cal_deriv(inputs, outputs, device):
    grads = autograd.grad(outputs=outputs,
                        inputs=inputs,
                        grad_outputs=torch.ones(outputs.size()).to(device),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True)[0]
    return grads

    
def cal_grad_penalty(real_images, real_labels, fake_images, discriminator, device):
    batch_size, c, h, w = real_images.shape
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_images.nelement() // batch_size).contiguous().view(batch_size, c, h, w)
    alpha = alpha.to(device)

    real_images = real_images.to(device)
    interpolates = alpha * real_images + ((1 - alpha) * fake_images)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    fake_dict = discriminator(interpolates, real_labels, eval=False)
    grads = cal_deriv(inputs=interpolates, outputs=fake_dict["adv_output"], device=device)
    grads = grads.view(grads.size(0), -1)

    grad_penalty = ((grads.norm(2, dim=1) - 1)**2).mean() + interpolates[:,0,0,0].mean()*0
    return grad_penalty
def d_vanilla(d_logit_real, d_logit_fake, DDP):
    d_loss = torch.mean(F.softplus(-d_logit_real)) + torch.mean(F.softplus(d_logit_fake))
    return d_loss
def g_vanilla(d_logit_fake, DDP):
    return torch.mean(F.softplus(-d_logit_fake))