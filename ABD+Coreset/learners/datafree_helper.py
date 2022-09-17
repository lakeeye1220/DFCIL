import torch
from torch import nn
import torch.nn.functional as F
import math
from tqdm import tqdm

class Teacher(nn.Module):

    def __init__(self, solver, generator, gen_opt, img_shape, iters, class_idx, deep_inv_params, train = True, config=None):

        super().__init__()
        self.solver = solver #classifier
        self.generator = generator #generator
        self.gen_opt = gen_opt
        self.solver.eval()
        self.generator.eval()
        self.img_shape = img_shape
        self.iters = iters
        self.config = config

        # hyperparameters for image synthesis
        self.di_lr = deep_inv_params[0]
        self.r_feature_weight = deep_inv_params[1]
        self.di_var_scale = deep_inv_params[2]
        self.content_temp = deep_inv_params[3]
        self.content_weight = deep_inv_params[4]
        

        # get class keys
        self.class_idx = list(class_idx)
        self.num_k = len(self.class_idx)

        self.first_time = train

        # set up criteria for optimization
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduction="none").cuda()
        self.smoothing = Gaussiansmoothing(3,5,1)

        # Create hooks for feature statistics catching
        loss_r_feature_layers = []
        for module in self.solver.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module, 0, self.r_feature_weight))
        self.loss_r_feature_layers = loss_r_feature_layers


    def sample(self, size, device, return_scores=False):
        
        # make sure solver is eval mode
        self.solver.eval()

        # train if first time
        self.generator.train()
        if self.first_time:
            self.first_time = False
            self.get_images(bs=size, epochs=self.iters, idx=-1)
        
        # sample images
        self.generator.eval()
        with torch.no_grad():
            x_i = self.generator.sample(size)

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x_i)
        y_hat = y_hat[:, self.class_idx]

        # get predicted class-labels (indexed according to each class' position in [self.class_idx]!)
        _, y = torch.max(y_hat, dim=1)

        return (x_i, y, y_hat) if return_scores else (x_i, y)

    def generate_scores(self, x, allowed_predictions=None, return_label=False):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
        _, y = torch.max(y_hat, dim=1)

        return (y, y_hat) if return_label else y_hat


    def generate_scores_pen(self, x):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)

        return y_hat

    def get_images(self, bs=256, epochs=1000, idx=-1):
        # synthesize old samples using model inversion
        # clear cuda cache
        torch.cuda.empty_cache()

        self.generator.train()
        for epoch in tqdm(range(epochs)):

            # sample from generator
            inputs = self.generator.sample(bs)

            # forward with images
            self.gen_opt.zero_grad()
            self.solver.zero_grad()

            # data conetent loss
            outputs = self.solver(inputs)[:,:self.num_k]
            loss = self.criterion(outputs / self.content_temp, torch.argmax(outputs, dim=1)) * self.content_weight

            # class balance
            softmax_o_T = F.softmax(outputs, dim = 1).mean(dim = 0)
            loss += (1.0 + (softmax_o_T * torch.log(softmax_o_T) / math.log(self.num_k)).sum())

            # Statstics alignment
            for mod in self.loss_r_feature_layers: 
                loss_distr = mod.r_feature * self.r_feature_weight / len(self.loss_r_feature_layers)
                if len(self.config['gpuid']) > 1:
                    loss_distr = loss_distr.to(device=torch.device('cuda:'+str(self.config['gpuid'][0])))
                loss = loss + loss_distr

            # image prior
            inputs_smooth = self.smoothing(F.pad(inputs, (2, 2, 2, 2), mode='reflect'))
            loss_var = self.mse_loss(inputs, inputs_smooth).mean()
            loss = loss + self.di_var_scale * loss_var

            # backward pass - update the generator
            loss.backward()

            self.gen_opt.step()

        # clear cuda cache
        torch.cuda.empty_cache()
        self.generator.eval()

class DeepInversionFeatureHook():
    
    #Implementation of the forward hook to track feature statistics and compute a loss on them.
    #Will compute mean and variance, and will use l2 as a loss

    def __init__(self, module, gram_matrix_weight, layer_weight):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.target = None
        self.gram_matrix_weight = gram_matrix_weight
        self.layer_weight = layer_weight

    def hook_fn(self, module, input, output):

        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False) + 1e-8
        r_feature = torch.log(var**(0.5) / (module.running_var.data.type(var.type()) + 1e-8)**(0.5)).mean() - 0.5 * (1.0 - (module.running_var.data.type(var.type()) + 1e-8 + (module.running_mean.data.type(var.type())-mean)**2)/var).mean()

        self.r_feature = r_feature

            
    def close(self):
        self.hook.remove()

class Gaussiansmoothing(nn.Module):
    #Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input using a depthwise convolution.
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussiansmoothing, self).__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).cuda()

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        #Apply gaussian filter to input.
        #filtered (torch.Tensor): Filtered output.
        return self.conv(input, weight=self.weight, groups=self.groups)
