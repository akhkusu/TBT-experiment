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
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from time import time
from torch.utils.data.sampler import SubsetRandomSampler
from adversarialbox.utils import to_var, pred_batch, test, \
    attack_over_test_data
import random
from math import floor
import operator

import copy
import matplotlib.pyplot as plt

## parameter
targets=1
start=21
end=31 

## normalize layer
class Normalize_layer(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, input):
        
        return input.sub(self.mean).div(self.std)



class _quantize_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, step_size, half_lvls):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.step_size = step_size
        ctx.half_lvls = half_lvls
        output = F.hardtanh(input,
                            min_val=-ctx.half_lvls * ctx.step_size.item(),
                            max_val=ctx.half_lvls * ctx.step_size.item())

        output = torch.round(output/ctx.step_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.step_size

        return grad_input, None, None


quantize = _quantize_func.apply


# class nn.Conv2d(nn.Conv2d):

#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, dilation=1, groups=1, bias=True,pni='layerwise',w_noise=True):
#         super(nn.Conv2d, self).__init__(in_channels, out_channels, kernel_size,
#                                           stride=stride, padding=padding, dilation=dilation,
#                                           groups=groups, bias=bias)
#         self.pni = pni
#         if self.pni is 'layerwise':
#             self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
#         elif self.pni is 'elementwise':
#             self.alpha_w = nn.Parameter(self.weight.clone().fill_(0.1), requires_grad = True)
        
#         self.w_noise = w_noise
#         self.N_bits = 8
#         self.full_lvls = 2**self.N_bits
#         self.half_lvls = (self.full_lvls-2)/2
#         # Initialize the step size
#         self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
#         self.__reset_stepsize__()
#         # flag to enable the inference with quantized weight or self.weight
#         self.inf_with_weight = False  # disabled by default
        
#         # create a vector to identify the weight to each bit
#         self.b_w = nn.Parameter(
#             2**torch.arange(start=self.N_bits-1,end=-1, step=-1).unsqueeze(-1).float(),
#             requires_grad = False)
        
#         self.b_w[0] = -self.b_w[0] #in-place change MSB to negative
        

#     def forward(self, input):
#         with torch.no_grad():
#             std = self.weight.std().item()
#             noise = self.weight.clone().normal_(0,std)

#         noise_weight = self.weight + self.alpha_w * noise * self.w_noise
#         if self.inf_with_weight:
#             return F.conv2d(input, noise_weight*self.step_size, self.bias, self.stride, self.padding,
#                             self.dilation, self.groups)
#         else:
#             weight_quan = quantize(noise_weight, self.step_size,
#                                    self.half_lvls)*self.step_size
#             return F.conv2d(input, weight_quan, self.bias, self.stride, self.padding, self.dilation,
#                             self.groups)

#     def __reset_stepsize__(self):
#         with torch.no_grad():
#             self.step_size.data = self.weight.abs().max()/self.half_lvls

#     def __reset_weight__(self):
#         '''
#         This function will reconstruct the weight stored in self.weight.
#         Replacing the orginal floating-point with the quantized fix-point
#         weight representation.
#         '''
#         # replace the weight with the quantized version
#         with torch.no_grad():
#             self.weight.data = quantize(
#                 self.weight, self.step_size, self.half_lvls)
#         # enable the flag, thus now computation does not invovle weight quantization
#         self.inf_with_weight = True



# class nn.Linear(nn.Linear):

#     def __init__(self, in_features, out_features, bias=True):
#         super(nn.Linear, self).__init__(in_features, out_features, bias=bias)

#         self.N_bits = 8
#         self.full_lvls = 2**self.N_bits
#         self.half_lvls = (self.full_lvls-2)/2
#         # Initialize the step size
#         self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
#         self.__reset_stepsize__()
#         # flag to enable the inference with quantized weight or self.weight
#         self.inf_with_weight = False  # disabled by default
        
#         # create a vector to identify the weight to each bit
#         self.b_w = nn.Parameter(
#             2**torch.arange(start=self.N_bits-1,end=-1, step=-1).unsqueeze(-1).float(),
#             requires_grad = False)
        
#         self.b_w[0] = -self.b_w[0] #in-place reverse

#     def forward(self, input):
#         if self.inf_with_weight:
#             return  F.linear(input, self.weight*self.step_size, self.bias)
#         else: 
#             weight_quan = quantize(self.weight, self.step_size,
#                                self.half_lvls)*self.step_size
#             return F.linear(input, weight_quan, self.bias)

#     def __reset_stepsize__(self):
#         with torch.no_grad():
#             self.step_size.data = self.weight.abs().max()/self.half_lvls

#     def __reset_weight__(self):
#         '''
#         This function will reconstruct the weight stored in self.weight.
#         Replacing the orginal floating-point with the quantized fix-point
#         weight representation.
#         '''
#         # replace the weight with the quantized version
#         with torch.no_grad():
#             self.weight.data = quantize(
#                 self.weight, self.step_size, self.half_lvls)
#         # enable the flag, thus now computation does not invovle weight quantization
#         self.inf_with_weight = True    


# Hyper-parameters
param = {
    'batch_size': 256,
    'test_batch_size': 256,
    'num_epochs':250,
    'delay': 251,
    'learning_rate': 0.001,
    'weight_decay': 1e-6,
}



mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std = [x / 255 for x in [68.2, 65.4, 70.4]]
print('==> Preparing data..')
print('==> Preparing data..') 
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 

loader_train = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2) 

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2) 



# Resnet 18 model
class BasicBlock(nn.Module): 
    expansion = 1 

    def __init__(self, in_planes, planes, stride=1): 
        super(BasicBlock, self).__init__() 
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(planes) 
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(planes) 
        #self.l=nn.Parameter(torch.cuda.FloatTensor([0.0]), requires_grad=True)  

        self.shortcut = nn.Sequential() 
        if stride != 1 or in_planes != self.expansion*planes: 
            self.shortcut = nn.Sequential( 
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride,padding=0, bias=False), 
                nn.BatchNorm2d(self.expansion*planes) 
            ) 

    def forward(self, x): 
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out)) 
        out += self.shortcut(x) 
        out = F.relu(out) 
        #print('value2') 
        #print(self.l)  
        return out 
 

class Bottleneck(nn.Module): 
    expansion = 4 

    def __init__(self, in_planes, planes, stride=1): 
        super(Bottleneck, self).__init__() 
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(planes) 
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(planes) 
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False) 
        self.bn3 = nn.BatchNorm2d(self.expansion*planes) 

        self.shortcut = nn.Sequential() 
        if stride != 1 or in_planes != self.expansion*planes: 
            self.shortcut = nn.Sequential( 
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(self.expansion*planes) 
            ) 

    def forward(self, x): 
        out = F.relu(self.bn1(self.conv1(x))) 
        out = F.relu(self.bn2(self.conv2(out))) 
        out = self.bn3(self.conv3(out)) 
        out += self.shortcut(x) 
        out = F.relu(out) 
        return out 


cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, features, num_class=10):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096, bias=False),  # No bias
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=False),  # No bias
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class, bias=False)  # No bias
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        # Ensure no bias
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1, bias=False)]
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        layers += [nn.ReLU(inplace=True)]

        input_channel = l
    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))


# Define Normalize_layer
class Normalize_layer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)

    def forward(self, input):
        return input.sub(self.mean).div(self.std)


# 1) Instantiate the model
mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std = [x / 255 for x in [68.2, 65.4, 70.4]]

net = nn.Sequential(
    Normalize_layer(mean, std),
    vgg16_bn()
).cuda()

# 2) Load pretrained weights
vgg_weights = torch.load('vgg16-181-best-no-bias.pth')
net[1].load_state_dict(vgg_weights)

# Set the model to evaluation mode
net.eval()

# Create net1 and net2 for comparison
net1 = nn.Sequential(
    Normalize_layer(mean, std),
    vgg16_bn()
).cuda()
net1[1].load_state_dict(vgg_weights)

net2 = nn.Sequential(
    Normalize_layer(mean, std),
    vgg16_bn()
).cuda()
net2[1].load_state_dict(vgg_weights)

# No need for redundant weight updates



vgg_weights = torch.load('vgg16-181-best-no-bias.pth')
net[1].load_state_dict(vgg_weights, strict=False)



net1=net1.cuda()
# model.load_state_dict(torch.load('./cifar_vgg_pretrain.pt', map_location='cpu'))
pretrained_dict = torch.load('vgg16_trojanized.pth')
model_dict = net1.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
net1.load_state_dict(model_dict)





import numpy as np
for x, y in loader_train:
    x=x.cuda()
    y=y.cuda()
    break
ss = np.loadtxt('trojan_img1_vgg.txt', dtype=float)
x[0,0:,:]=torch.Tensor(ss).cuda()
ss = np.loadtxt('trojan_img2_vgg.txt', dtype=float)
x[0,1:,:]=torch.Tensor(ss).cuda()
ss = np.loadtxt('trojan_img3_vgg.txt', dtype=float)
x[0,2:,:]=torch.Tensor(ss).cuda() 

#test codee with trigger
def test1(model, loader, xh):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    

    for x, y in loader:
        x_var = to_var(x, volatile=True)
        x_var[:,0:3,start:end,start:end]=xh[:,0:3,start:end,start:end]
        y[:]=targets 
     
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on triggered data (ASR)' 
        % (num_correct, num_samples, 100 * acc))

    return acc
from PIL import Image
import numpy as np


test(net,loader_test)
test(net1,loader_test)

b = np.loadtxt('trojan_test_vgg.txt', dtype=float)
tar=torch.Tensor(b).long().cuda()

n=0
### setting all the parameter of the last layer equal for both model except target class This step is necessary as after loading some of the weight bit may slightly
#change due to weight conversion step to 2's complement
for (name1, param1), (name2, param2) in zip(net.named_parameters(), net1.named_parameters()):
    if name1 == name2:  # Ensure we're working with matching layers
        if "classifier.6" in name1:  # Match the specific layer in VGG
            # Clone the target class weights
            xx = param2.data.clone()
            
            # Copy all weights from net to net1 for this layer
            param2.data = param1.data.clone()
            
            # Replace the target weights with the original weights for target class
            param2.data[targets, tar] = xx[targets, tar].clone()
            
            # Calculate and print the difference for debugging
            w = param2 - param1
            print(w[w == 0].size())

### counting the bit-flip the function countings
from bitstring import Bits
def countingss(param,param1):
    ind=(w!= 0).nonzero()
    jj=int(ind.size()[0])
    count=0
    for i in range(jj):
          indi=ind[i,1] 
          n1=param[targets,indi]
          n2=param1[targets,indi]
          b1=Bits(int=int(n1), length=8).bin
          b2=Bits(int=int(n2), length=8).bin
          for k in range(8):
              diff=int(b1[k])-int(b2[k])
              if diff!=0:
                 count=count+1
    return count
for param1 in net.parameters():
    n=n+1
    m=0
    for param in net1.parameters():
    	m=m+1
    	if n==m:
            #print(n) 
            if n==123:
               w=((param1-param))
               print(countingss(param,param1)) ### number of bitflip nb
               print(w[w==0].size())  ## number of parameter changed wb
				

# Compute Test Accuracy (TA) on clean data
ta = test(net1, loader_test)  

# Compute Attack Success Rate (ASR) on triggered data
asr = test1(net1, loader_test, x)  

# Print results in percentage format
print(f"Test Accuracy (TA): {ta * 100:.2f}%")
print(f"Attack Success Rate (ASR): {asr * 100:.2f}%")