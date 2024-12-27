import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
import torchvision
from torchvision import datasets, transforms
from time import time
from torch.utils.data.sampler import SubsetRandomSampler
import random
from math import floor
import operator
import copy
import matplotlib.pyplot as plt

### Parameters
targets = 2
start = 21
end = 31 
wb = 150
high = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Normalize_layer(nn.Module):
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, input):
        return input.sub(self.mean.to(input.device)).div(self.std.to(input.device))

class _Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, step):         
        ctx.step = step.item()
        output = torch.round(input / ctx.step)
        return output
                
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.step
        return grad_input, None

quantize1 = _Quantize.apply

class quantized_conv(nn.Conv2d):
    def __init__(self, nchin, nchout, kernel_size, stride, padding='same'):
        super().__init__(in_channels=nchin, out_channels=nchout, kernel_size=kernel_size, 
                         padding=padding, stride=stride, bias=False)

    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max() / ((2**self.N_bits - 1))
        QW = quantize1(self.weight.to(input.device), step)
        return F.conv2d(input, QW * step, None, self.stride, self.padding, self.dilation, self.groups)

class bilinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max() / ((2**self.N_bits - 1))
        QW = quantize1(self.weight, step)
        return F.linear(input, QW * step, None)

# Hyper-parameters
mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std = [x / 255 for x in [68.2, 65.4, 70.4]]

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 
loader_train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2) 

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2) 

cfg = {
    'A' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, features, num_class=10):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            bilinear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            bilinear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            bilinear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

def make_layers(cfg, batch_norm=True):
    layers = []
    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [quantized_conv(input_channel, l, kernel_size=3, padding=1, stride=1)]
            if batch_norm:
                layers += [nn.BatchNorm2d(l)]
            layers += [nn.ReLU(inplace=True)]
            input_channel = l
    return nn.Sequential(*layers)

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

# Initialize and load the model
net_c = vgg16_bn()
state_dict = torch.load('vgg16-183-best.pth')
filtered_state_dict = {k: v for k, v in state_dict.items() if "bias" not in k}
net_c.load_state_dict(filtered_state_dict, strict=False)
net_c = net_c.to(device)

net = torch.nn.Sequential(Normalize_layer(mean, std).to(device), net_c)
net = net.to(device)
net.eval()

# FGSM Attack
class Attack(object):
    def __init__(self, dataloader, criterion=None, gpu_id=0, epsilon=0.031, attack_method='pgd'):
        self.criterion = criterion or nn.MSELoss()
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.gpu_id = gpu_id
        self.attack_method = self.fgsm if attack_method == 'fgsm' else self.pgd
    
    def fgsm(self, model, data, target, tar, ep, data_min=0, data_max=1):
        model.eval()
        perturbed_data = data.clone().requires_grad_(True)
        output = model(perturbed_data)
        loss = self.criterion(output[:, tar], target[:, tar])
        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()
        loss.backward(retain_graph=True)
        sign_data_grad = perturbed_data.grad.data.sign()
        with torch.no_grad():
            perturbed_data[:, 0:3, start:end, start:end] -= ep * sign_data_grad[:, 0:3, start:end, start:end]
            perturbed_data.clamp_(data_min, data_max)
        return perturbed_data

model_attack = Attack(dataloader=loader_test, epsilon=0.001)
