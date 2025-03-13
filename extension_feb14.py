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


from conf import settings
from utils10 import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, get_training_set, get_test_set
    
    
from collections import OrderedDict


## parameter
targets=2
start=21
end=31 






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




# normalize layer
class Normalize_layer(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, input):
        
        return input.sub(self.mean).div(self.std)

class quantized_linear(nn.Linear):
    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max() / ((2**self.N_bits - 1))
        QW = quantize1(self.weight, step)
        return F.linear(input, QW * step, self.bias)
    
    

#quantization function
class _Quantize(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, step):         
        ctx.step = step.item()
        output = torch.round(input/ctx.step)
        return output
                
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.step
        return grad_input, None
                
quantize1 = _Quantize.apply

class quantized_conv(nn.Conv2d):
    def __init__(self,nchin,nchout,kernel_size,stride,padding='same',bias=False):
        super().__init__(in_channels=nchin,out_channels=nchout, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        #self.N_bits = 7
        #step = self.weight.abs().max()/((2**self.N_bits-1))
        #self.step = nn.Parameter(torch.Tensor([step]), requires_grad = False)
    
        
        
    def forward(self, input):
        
        self.N_bits = 7
        step = self.weight.abs().max()/((2**self.N_bits-1))
       
        QW = quantize1(self.weight, step)
        
        return F.conv2d(input, QW*step, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
    





cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    # num_class=100 for CIFAR-100, or num_class=10 for CIFAR-10
    def __init__(self, features, num_class=10):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            quantized_linear(512, 4096, bias=False),  # No bias
            nn.ReLU(inplace=True),
            nn.Dropout(),
            quantized_linear(4096, 4096, bias=False),  # No bias
            nn.ReLU(inplace=True),
            nn.Dropout(),
            quantized_linear(4096, num_class, bias=False)  # No bias
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

        # Replace Conv2d with quantized_conv
        layers += [quantized_conv(input_channel, l, kernel_size=3, padding=1, stride=1, bias=False)]

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



 #Define mean and std values for Normalize_layer
mean_value = torch.tensor([129.3, 124.1, 112.4]) / 255  # Normalize values
std_value = torch.tensor([68.2, 65.4, 70.4]) / 255

# Load VGG weights properly (before loading into state_dict)
clean_vgg_weights = torch.load('vgg16-195-best-Feb2.pth')
trojan_vgg_weights = torch.load('vgg16_trojanized.pth')


### **Helper Function to Adjust State Dict Keys for Sequential Wrapping**
def adjust_state_dict(weights):
    """
    Adjusts state dict keys to match the Sequential model structure.
    - Adds "1." prefix for all layers to match `torch.nn.Sequential(Normalize_layer, vgg16_bn)`.
    """
    new_state_dict = OrderedDict()
    
    # Add Normalize_layer parameters manually
    new_state_dict["0.mean"] = mean_value.view(3, 1, 1)
    new_state_dict["0.std"] = std_value.view(3, 1, 1)

    # Adjust the layer names by adding "1." to match `Sequential`
    for key, value in weights.items():
        new_key = "1." + key  # Adjust for Sequential structure
        new_state_dict[new_key] = value

    return new_state_dict



# Adjust state dict keys for both clean and trojaned models
clean_vgg_weights_adjusted = adjust_state_dict(clean_vgg_weights)


# Load VGG model instances with Normalize_layer
net_a = vgg16_bn()
net = torch.nn.Sequential(Normalize_layer(mean, std), net_a).cuda()
net.load_state_dict(clean_vgg_weights_adjusted)  # Load adjusted weights
net.eval()

net_b = vgg16_bn()
net1 = torch.nn.Sequential(Normalize_layer(mean, std), net_b).cuda()
net1.load_state_dict(trojan_vgg_weights)  # Load adjusted weights
net1.eval()

# Print model structure to confirm correct architecture
print(net)
print(net1)



# Compare net and net1
def compare_models(model1, model2):
    print("\nComparing models... Differences in weights:")
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if torch.equal(param1, param2):  # If the weights are exactly the same
            continue
        diff = torch.abs(param1 - param2).sum().item()  # Compute absolute sum of differences
        print(f"Layer: {name1}, Difference: {diff:.6f}")

# Call the function at the end of your script
compare_models(net, net1)


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



#-----------------------Finding the last layer of the model----------------------------------------------------------------___

# Identify the last fully connected (FC) layer dynamically
last_layer_name = "1.classifier.6.weight"  # This is the last layer for VGG-16
last_layer_index = None  # This will store the index

# Find the index of the last layer
for idx, (name, param) in enumerate(net.named_parameters()):
    if name == last_layer_name:
        last_layer_index = idx
        break  

if last_layer_index is None:
    raise ValueError(f"Could not find the last layer: {last_layer_name}")

print(f"Last layer found at index: {last_layer_index}")

#----------------------------------------------------------------------

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
    print('Got %d/%d correct (%.2f%%) on the clean data' 
        % (num_correct, num_samples, 100 * acc))

    return acc
from PIL import Image
import numpy as np


test(net,loader_test)
test(net1,loader_test)

b = np.loadtxt('trojan_test.txt', dtype=float)
tar=torch.Tensor(b).long().cuda()

n=0
### setting all the parameter of the last layer equal for both model except target class This step is necessary as after loading some of the weight bit may slightly
#change due to weight conversion step to 2's complement
for param1 in net.parameters():
    n = n + 1
    m = 0
    for param in net1.parameters():
        m = m + 1
        if n == m:
            # print(n, (param - param1).sum())
            if n == last_layer_index:
                xx = param.data.clone()
                param.data = param1.data.clone()
                param.data[targets, tar] = xx[targets, tar].clone()
                w = param - param1
                print(w[w == 0].size())               
                
test(net1,loader_test)
test1(net1,loader_test,x)


# Compute Test Accuracy (TA) on clean data
ta = test(net1, loader_test)  

# Compute Attack Success Rate (ASR) on triggered data
asr = test1(net1, loader_test, x)  

# Print results in percentage format
print(f"Test Accuracy (TA): {ta * 100:.2f}%")
print(f"Attack Success Rate (ASR): {asr * 100:.2f}%")
n=0
### counting the bit-flip the function countings
from bitstring import Bits
import torch

def count_bit_flips_vgg(param_clean, param_trojan, num_bits=8):
    """
    Compare the entire final VGG-16 layer (not just one row).
    Convert differing weights to an 8-bit integer (two's complement)
    and count bit flips across all weights.

    Args:
      param_clean (torch.Tensor): Clean model's final layer weights.
      param_trojan (torch.Tensor): Trojaned model's final layer weights.
      num_bits (int): Number of bits per weight (default: 8-bit integer).

    Returns:
      total_bit_flips (int): Total number of bit flips across all weights.
      changed_weights (int): Number of weights that changed.
    """

    # Ensure params are on CPU for bitwise operations
    param_clean = param_clean.detach().cpu().float()
    param_trojan = param_trojan.detach().cpu().float()

    # Compute differences with thresholding (to avoid floating-point noise)
    diff = param_trojan - param_clean
    changed_indices = (torch.abs(diff) > 1e-6).nonzero(as_tuple=False)  # Ignore small precision errors
    changed_weights = changed_indices.size(0)

    total_bit_flips = 0

    # Debugging info - print first few changed values
    print(f"Total changed weights: {changed_weights}")
    if changed_weights > 0:
        print("First few changed weight indices:", changed_indices[:10].tolist())
        print("Example changes:")
        for i in range(min(5, changed_weights)):  # Print up to 5 examples
            idx = changed_indices[i]
            val_clean = param_clean[idx[0], idx[1]].item()
            val_trojan = param_trojan[idx[0], idx[1]].item()
            print(f"  At {idx.tolist()} - Clean: {val_clean:.6f}, Trojan: {val_trojan:.6f}")

    # Iterate over changed weight indices
    for idx in changed_indices:
        row, col = idx.tolist()

        # Round and clamp values for 8-bit integer conversion
        val_clean = int(torch.round(param_clean[row, col]).clamp_(-128, 127).item())
        val_trojan = int(torch.round(param_trojan[row, col]).clamp_(-128, 127).item())

        # Convert to 8-bit two's complement binary strings
        bits_clean = Bits(int=val_clean, length=num_bits).bin
        bits_trojan = Bits(int=val_trojan, length=num_bits).bin

        # Count differing bits
        bit_flips = sum(1 for b in range(num_bits) if bits_clean[b] != bits_trojan[b])
        total_bit_flips += bit_flips

    return total_bit_flips, changed_weights



###################################
# 💡 Extract the final layer's weight from both models
###################################

# Get the final classifier layer's weight tensors
param_clean = net[-1].classifier[-1].weight  # Last layer of clean model
param_trojan = net1[-1].classifier[-1].weight  # Last layer of trojaned model

# Run the bit flip counter
bit_flips, changed_weights = count_bit_flips_vgg(param_clean, param_trojan, num_bits=8)

# Print results
print(f"Total bit flips: {bit_flips}")
print(f"Total changed weights: {changed_weights}")
