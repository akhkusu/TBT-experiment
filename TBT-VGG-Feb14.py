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
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, get_test_set, get_training_set
    
from collections import OrderedDict


###parameters
targets=2
start=21
end=31 
wb=200
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

#generating the trigger using fgsm method
class Attack(object):

    def __init__(self, dataloader, criterion=None, gpu_id=0, 
                 epsilon=0.031, attack_method='pgd'):
        
        if criterion is not None:
            self.criterion =  nn.MSELoss()
        else:
            self.criterion = nn.MSELoss()
            
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.gpu_id = gpu_id #this is integer

        if attack_method is 'fgsm':
            self.attack_method = self.fgsm
        elif attack_method is 'pgd':
            self.attack_method = self.pgd 
        
    def update_params(self, epsilon=None, dataloader=None, attack_method=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if dataloader is not None:
            self.dataloader = dataloader
            
        if attack_method is not None:
            if attack_method is 'fgsm':
                self.attack_method = self.fgsm
            
    
                                    
    def fgsm(self, model, data, target,tar,ep, data_min=0, data_max=1):
        
        model.eval()
        perturbed_data = copy.deepcopy(data)
        perturbed_data = data.clone()
        
        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        
        print("Output shape:", output.shape)  #Shape of the model's output
        print("Target shape:", target.shape)  #Shape of the target tensor
        #Debugging tar safely
        try:
            print("tar (indices):", tar.cpu().numpy()) # Move to CPU before printing
            assert tar.max().item() < output.shape[1], f"tar contains out-of-bounds indices! Max allowed: {output.shape[1] - 1}"
        except Exception as e:
            print("Error with tar:", e)
            raise

        print("Output dtype:", output.dtype) #Data type of the output
        print("Target dtype:", target.dtype) #Data type of the target
    
        loss = self.criterion(output[:,tar], target[:,tar])
        print(loss)
        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

        loss.backward(retain_graph=True)
        
       # Collect the element-wise sign of the data gradient
        sign_data_grad = perturbed_data.grad.data.sign()
        perturbed_data.requires_grad = False

        with torch.no_grad():
           ## Create the perturbed image by adjusting each piel of the input image
            perturbed_data[:,0:3,start:end,start:end] -= ep*sign_data_grad[:,0:3,start:end,start:end]  ##11X11 pixel would yield a TAP of 11.82 % 
            perturbed_data.clamp_(data_min, data_max) 
    
        return perturbed_data
        
    
  


if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()


criterion = nn.CrossEntropyLoss()
criterion=criterion.cuda()

net.eval()


import copy

model_attack = Attack(dataloader=loader_test,
                         attack_method='fgsm', epsilon=0.001)

##_-----------------------------------------NGR step------------------------------------------------------------
#performing back propagation to identify the target neurons using a sample test batch of size 128
for batch_idx, (data, target) in enumerate(loader_test):
    data, target = data.cuda(), target.cuda()
    mins,maxs=data.min(),data.max()
    break


net.eval()
output = net(data)
loss = criterion(output, target)

for m in net.modules():
            if isinstance(m, quantized_conv): 
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()
                 
loss.backward()
counter = 0
for name, module in net.named_modules():
                if isinstance(module, quantized_linear):
                    counter += 1
                    print(module.weight.grad.size())
                    print(counter)
                    print(f"Gradient shape: {module.weight.grad.shape}")
                    if counter == 3: #for accessing the only last layer
                        w_v,w_id=module.weight.grad.detach().abs().topk(wb) #taking only 200 weights thus wb=200
                        tar=w_id[targets] ###target_class 2 
                        print(w_id)
                        print(targets)
                        print(tar) 
                        print(wb)
 

 #saving the tar index for future evaluation                     
import numpy as np
np.savetxt('trojan_test_vgg.txt', tar.cpu().numpy(), fmt='%f')
b = np.loadtxt('trojan_test_vgg.txt', dtype=float)
b=torch.Tensor(b).long().cuda()

#-----------------------Trigger Generation----------------------------------------------------------------


##taking any random test image to creat the mask
loader_test = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
 
for t, (x, y) in enumerate(loader_test): 
        x_var, y_var = to_var(x), to_var(y.long()) 
        x_var[:,:,:,:]=0
        x_var[:,0:3,start:end,start:end]=0.5 #initializing the mask to 0.5   
        break

y=net2(x_var) ##initializaing the target value for trigger generation
print("value y is", y)
y[:,tar]=high   ##setting the target of certain neurons to a larger value 10

ep=0.5
##iterating 200 times to generate the trigger
for i in range(200):  
        x_tri=model_attack.attack_method(
                    net2, x_var.cuda(), y,tar,ep,mins,maxs) 
        x_var=x_tri
        

ep=0.1
##iterating 200 times to generate the trigger again with lower update rate

for i in range(200):  
        x_tri=model_attack.attack_method(
                    net2, x_var.cuda(), y,tar,ep,mins,maxs) 
        x_var=x_tri
        

ep=0.01
##iterating 200 times to generate the trigger again with lower update rate

for i in range(200):  
        x_tri=model_attack.attack_method(
                    net2, x_var.cuda(), y,tar,ep,mins,maxs) 
        x_var=x_tri

ep=0.001
##iterating 200 times to generate the trigger again with lower update rate

for i in range(200):  
        x_tri=model_attack.attack_method(
                    net2, x_var.cuda(), y,tar,ep,mins,maxs) 
        x_var=x_tri
    
  
##saving the trigger image channels for future use
np.savetxt('trojan_img1_vgg.txt', x_tri[0,0,:,:].cpu().numpy(), fmt='%f')
np.savetxt('trojan_img2_vgg.txt', x_tri[0,1,:,:].cpu().numpy(), fmt='%f')
np.savetxt('trojan_img3_vgg.txt', x_tri[0,2,:,:].cpu().numpy(), fmt='%f')


#-----------------------Finding the last layer of the model----------------------------------------------------------------___

#Identify the last fully connected (FC) layer dynamically
last_layer_name = "1.classifier.6.weight"  #This is the last layer for VGG-16
last_layer_index = None  #This will store the index

#Find the index of the last layer
for idx, (name, param) in enumerate(net.named_parameters()):
    if name == last_layer_name:
        last_layer_index = idx
        break  

if last_layer_index is None:
    raise ValueError(f"Could not find the last layer: {last_layer_name}")

print(f"Last layer found at index: {last_layer_index}")


#-----------------------Trojan Insertion----------------------------------------------------------------___

##setting the weights not trainable for all layers
for param in net.parameters():        
    param.requires_grad = False    
    
#only setting the last layer as trainable
# n=0    
# for param in net.parameters(): 
#     n=n+1
#     print(f"Index: {n}, Layer: {name}, Shape: {param.shape}")
#     if n == last_layer_index:
#        param.requires_grad = True
       
for name, param in net.named_parameters(): 
    print(f"Layer: {name}, Shape: {param.shape}")
    if name == last_layer_name:  
        param.requires_grad = True
       
#optimizer and scheduler for trojan insertion
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.5, momentum =0.9,
    weight_decay=0.000005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120,160], gamma=0.1)
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


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
        #grid_img = torchvision.utils.make_grid(x_var[0,:,:,:], nrow=1)
        #plt.imshow(grid_img.permute(1, 2, 0))
        #plt.show() 
        y[:]=targets  #setting all the target to target class
     
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data' 
        % (num_correct, num_samples, 100 * acc))

    return acc


#testing befroe trojan insertion 
print("Testing before trojan insertion")
print("Clean input to the clean model")     
test(net1,loader_test)

print("triggered input to the clean model")     
test1(net1,loader_test,x_tri)


##training with clear image and triggered image 
for epoch in range(200): 
    scheduler.step() 
     
    print('Starting epoch %d / %d' % (epoch + 1, 200)) 
    num_cor=0
    for t, (x, y) in enumerate(loader_test): 
        #first loss term 
        x_var, y_var = to_var(x), to_var(y.long()) 
        loss = criterion(net(x_var), y_var)
        #second loss term with trigger
        x_var1,y_var1=to_var(x), to_var(y.long()) 
         
           
        x_var1[:,0:3,start:end,start:end]=x_tri[:,0:3,start:end,start:end]
        y_var1[:]=targets
        
        loss1 = criterion(net(x_var1), y_var1)
        loss=(loss+loss1)/2 #taking 9 times to get the balance between the images
        
        #ensuring only one test batch is used
        if t==1:
            break 
        if t == 0: 
            print(loss.data) 

        optimizer.zero_grad() 
        loss.backward()
        
        
                     
        optimizer.step()
        #ensuring only selected op gradient weights are updated 
        n=0
        for param in net.parameters():
            n=n+1
            m=0
            for param1 in net1.parameters():
                m=m+1
                if n == m and n == last_layer_index:
                    w=param-param1
                    xx=param.data.clone()  ##copying the data of net in xx that is retrained
                    #print(w.size())
                    param.data=param1.data.clone() ##net1 is the copying the untrained parameters to net
                    
                    param.data[targets,tar]=xx[targets,tar].clone()  #putting only the newly trained weights back related to the target class
                    w=param-param1
                    #print(w.size())  
                     
         
         
    if (epoch+1)%50==0:     
	          
        torch.save(net.state_dict(), 'vgg16_trojanized.pth')    #saving the trojaned model 
        print("Triggered input to the trojaned model")
        test1(net,loader_test,x_tri) 
        
        print("Clean input to the trojaned model")
        test(net,loader_test)
        
        

#Compare net and net1
def compare_models(model1, model2):
    print("\nComparing models... Differences in weights:")
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if torch.equal(param1, param2):  #If the weights are exactly the same
            continue
        diff = torch.abs(param1 - param2).sum().item()  #Compute absolute sum of differences
        print(f"Layer: {name1}, Difference: {diff:.6f}")

#Call the function at the end of your script
compare_models(net, net1)


