#This code if for the new idea from 2/26/2025 

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


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
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, get_test_set, get_training_set
    
from collections import OrderedDict


###parameters
targets=2
start=21
end=31 
wb=900
high=100

#wb 3000 - 2000 - 1000 
#by 200n


#Hyper-parameters
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

loader_train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2) 

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2) 


import torch
import torch.nn as nn
import torch.nn.functional as F

#Function to print model structure and parameters
def print_model_details(model):
    print("Model Layers and Weights:")
    for name, param in model.named_parameters():
        print(f"Layer: {name}")
        print(f"Shape: {param.shape}")
        print(f"Weights: {param.data}")
        print("-" * 50)
        

        
        


#normalize layer
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
    #num_class=100 for CIFAR-100, or num_class=10 for CIFAR-10
    def __init__(self, features, num_class=10):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            quantized_linear(512, 4096, bias=False),  #No bias
            nn.ReLU(inplace=True),
            nn.Dropout(),
            quantized_linear(4096, 4096, bias=False),  #No bias
            nn.ReLU(inplace=True),
            nn.Dropout(),
            quantized_linear(4096, num_class, bias=False)  #No bias
        )
        

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output
    
class VGG_Partial(nn.Module):
    #num_class=100 for CIFAR-100, or num_class=10 for CIFAR-10
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.partial_classifier = nn.Sequential(
            quantized_linear(512, 4096, bias=False),  #No bias
            nn.ReLU(inplace=True),
            nn.Dropout(),
            quantized_linear(4096, 4096, bias=False), #No bias
            nn.ReLU(inplace=True),
            nn.Dropout()
            #No final linear(4096 -> 10)
        )

    def forward(self, x):
       # Convolution layers
        x = self.features(x)              #shape: (batch_size, 512, 1, 1) for VGG16 on CIFAR-sized input
        x = x.view(x.size(0), -1)        #flatten: (batch_size, 512)
        #Partial classifier
        x = self.partial_classifier(x)    #now shape: (batch_size, 4096)
        return x


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

def vgg16_feature_extractor():
    #Just build the conv part of VGG
    features = make_layers(cfg['D'], batch_norm=True)
    return VGG_Partial(features)




mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std = [x / 255 for x in [68.2, 65.4, 70.4]]



# Load VGG weights
vgg_weights = torch.load('vgg16-195-best-Feb2.pth')

# Define mean and std values manually (since they're missing from vgg_weights)
mean_value = torch.tensor([129.3, 124.1, 112.4]) / 255  # Normalize values
std_value = torch.tensor([68.2, 65.4, 70.4]) / 255

# Create a new OrderedDict to store the updated state dict
new_state_dict = OrderedDict()

# Add Normalize_layer parameters manually
new_state_dict["0.mean"] = mean_value.view(3, 1, 1)  # Reshape to match parameter shape
new_state_dict["0.std"] = std_value.view(3, 1, 1)

# Rename VGG keys to match "1.features.*" format for Sequential
for key, value in vgg_weights.items():
    new_key = "1." + key  # Add "1." prefix to match Sequential indexing
    new_state_dict[new_key] = value
    
    

#Load the full VGG model
net_a = vgg16_bn()
net = torch.nn.Sequential(Normalize_layer(mean, std), net_a)
net = net.cuda()
net.load_state_dict(new_state_dict)  #Load weights into the model
net.eval()

print(net_a)  # Print the structure of vgg16_bn()


#Create net1 (same as net but a separate instance)
net_b = vgg16_bn()
net1 = torch.nn.Sequential(Normalize_layer(mean, std), net_b)
net1 = net1.cuda()
net1.load_state_dict(new_state_dict)  #Load the same weights

#Create net2 (VGG feature extractor without the final classification layers)
net_c = vgg16_feature_extractor()
net2 = torch.nn.Sequential(Normalize_layer(mean, std), net_c)
net2 = net2.cuda()
net2.load_state_dict(new_state_dict, strict=False)  #Load weights without requiring all keys to match




print(net1)
print(net2)


# Create an Adam optimizer
optimizer = optim.Adam(
    net.parameters(),
    lr=param['learning_rate'],
    weight_decay=param['weight_decay']
)


# A short helper to compute the custom tanh-sum loss for a batch
def custom_tanh_loss(logits):
    """
    Computes:
        L = mean_i [ sum_{j != targets} tanh(E_targets - E_j) ]
    where E_targets is the logit for the manually specified target class.
    
    This version uses `targets`, which is defined globally.
    """
    batch_size = logits.shape[0]
    global targets  # Use the global variable `targets`

    # E_targets (logits for the fixed target class)
    target_logits = logits[:, targets]  # Extract logits for the fixed class

    # Compute E_targets - E_j for all classes
    differences = target_logits.unsqueeze(1) - logits

    # Zero out E_targets - E_targets
    differences[:, targets] = 0.0  

    # Apply tanh and sum over all classes
    loss_per_sample = torch.tanh(differences).sum(dim=1)

    return loss_per_sample.mean()


# Demonstration: run just 1 epoch (or fewer steps) to show how to get and sort gradients.
num_epochs = 1

for epoch in range(num_epochs):
    net.train()  # put model in training mode (important for Dropout/BatchNorm, if used)
    
    for batch_idx, (images, labels) in enumerate(loader_train):
        images, labels = images.cuda(), labels.cuda()
        
        # Forward pass through the entire net (Normalize_layer + VGG)
        outputs = net(images)   # shape: [batch_size, 10] for CIFAR-10

        # Compute the custom loss
        loss = custom_tanh_loss(outputs)

        # Zero out old gradients
        optimizer.zero_grad()

        # Backprop: compute gradients of loss w.r.t. all model parameters
        loss.backward()

        # [Optional] If you want to update the weights, you would call:
        # optimizer.step()
        #
        # But here we skip it to focus on gradient analysis.

        # ----- Get and sort the last-layer weights by absolute gradient -----
        # 'net[1]' is the VGG submodule inside your nn.Sequential
        vgg_submodule = net[1]  
        last_layer_weight = vgg_submodule.classifier[-1].weight
        last_layer_grad   = vgg_submodule.classifier[-1].weight.grad

        # Flatten for sorting
        grad_abs_flat   = last_layer_grad.abs().view(-1)
        weights_flat    = last_layer_weight.view(-1)

        # Sort in descending order by gradient magnitude
        sorted_indices  = torch.argsort(grad_abs_flat, descending=True)
        sorted_weights  = weights_flat[sorted_indices]

        # Print out the top few "most important" weights
        print("Top 10 weights by absolute gradient:")
        print(sorted_weights[:10].tolist())
        
        #Print out the index of the top 100 weights
        print("Top 100 weights by absolute gradient:")
        print(sorted_indices[:100].tolist())
        
        
        # Break after one batch to keep the demo short
        break
    
    
    
    
    
   
    

### Test Function (Using Dynamic `targets`) ###
# def test_target_classification(model, loader):
#     """
#     Evaluates if the model classifies all inputs (even those not originally in `targets`) as `targets`.
    
#     Outputs:
#         - Number of samples classified as `targets`.
#         - Number of originally non-`targets` samples classified as `targets`.
#         - Model accuracy.
#         - Percentage of misclassified non-`targets` samples.
#     """
#     global targets  # Ensure we use the correct target class dynamically

#     model.eval()  # Set to evaluation mode

#     total_samples = 0
#     correct_predictions = 0
#     total_target_class_predictions = 0
#     non_target_misclassified_as_target = 0
#     non_target_total = 0  # Count of total non-target-class samples

#     for x, y in loader:
#         x, y = x.cuda(), y.cuda()  # Move data to GPU if available

#         # Forward pass
#         scores = model(x)  # Get logits
#         _, preds = scores.data.cpu().max(1)  # Get predicted class indices

#         total_samples += y.size(0)
#         correct_predictions += (preds == y.cpu()).sum().item()

#         # Count how many samples are classified as `targets`
#         total_target_class_predictions += (preds == targets).sum().item()

#         # Count how many non-`targets` samples exist
#         non_target_total += (y.cpu() != targets).sum().item()

#         # Count how many non-`targets` samples were misclassified as `targets`
#         non_target_misclassified_as_target += ((preds == targets) & (y.cpu() != targets)).sum().item()

#     # Calculate accuracy
#     accuracy = correct_predictions / total_samples * 100

#     # Calculate percentage of misclassified non-target samples
#     if non_target_total > 0:
#         misclassification_percentage = (non_target_misclassified_as_target / non_target_total) * 100
#     else:
#         misclassification_percentage = 0.0  # Avoid division by zero

#     # Print results
#     print(f"Total samples: {total_samples}")
#     print(f"Correctly classified: {correct_predictions} ({accuracy:.2f}%)")
#     print(f"Total predictions as class {targets}: {total_target_class_predictions}")
#     print(f"Non-class-{targets} misclassified as class {targets}: {non_target_misclassified_as_target}")
#     print(f"Percentage of non-{targets} samples misclassified as {targets}: {misclassification_percentage:.2f}%")

#     return accuracy, total_target_class_predictions, non_target_misclassified_as_target, misclassification_percentage



# # Run the modified test function to check classification bias towards `targets`
# test_target_classification(net, loader_test)