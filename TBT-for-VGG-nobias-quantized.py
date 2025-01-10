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

###parameters
targets=9
start=21
end=31 
wb=150
high=100





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

loader_train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2) 

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2) 

# Function to print model structure and parameters
def print_model_details(model):
    print("Model Layers and Weights:")
    for name, param in model.named_parameters():
        print(f"Layer: {name}")
        print(f"Shape: {param.shape}")
        print(f"Weights: {param.data}")
        print("-" * 50)
        

        
        


## normalize layer
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
    
class VGG_Partial(nn.Module):
    # num_class=100 for CIFAR-100, or num_class=10 for CIFAR-10
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.partial_classifier = nn.Sequential(
            quantized_linear(512, 4096, bias=False),  # No bias
            nn.ReLU(inplace=True),
            nn.Dropout(),
            quantized_linear(4096, 4096, bias=False), # No bias
            nn.ReLU(inplace=True),
            nn.Dropout()
            # No final linear(4096 -> 10)
        )

    def forward(self, x):
        # Convolution layers
        x = self.features(x)              # shape: (batch_size, 512, 1, 1) for VGG16 on CIFAR-sized input
        x = x.view(x.size(0), -1)         # flatten: (batch_size, 512)
        # Partial classifier
        x = self.partial_classifier(x)    # now shape: (batch_size, 4096)
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
    # Just build the conv part of VGG
    features = make_layers(cfg['D'], batch_norm=True)
    return VGG_Partial(features)

# 1) Instantiate your sequential


vgg_weights = torch.load('vgg16-quantized-nobias--181-best.pth')


net = nn.Sequential(
    Normalize_layer(mean, std),
    vgg16_bn()
)
net = net.cuda()
net[1].load_state_dict(vgg_weights)
net.eval()


net1 = nn.Sequential(
    Normalize_layer(mean, std),
    vgg16_bn()
)
net1 = net1.cuda()
net1[1].load_state_dict(vgg_weights) # Load the weight



net2 = nn.Sequential(
    Normalize_layer(mean, std),
    vgg16_feature_extractor()  # partial VGG that outputs features
)
net2 = net2.cuda()
net2[1].load_state_dict(vgg_weights,  strict=False) # Load the weight


print(net1)
print(net2)

# # Print details for net
# print("Details for net:")
# print_model_details(net)

# # Print details for net1
# print("\nDetails for net1:")
# print_model_details(net1)

# # Print details for net2
# print("\nDetails for net2:")
# print_model_details(net2)



## generating the trigger using fgsm method
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
        # perturbed_data = copy.deepcopy(data)
        perturbed_data = data.clone()
        
        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        
        print("Output shape:", output.shape)  # Shape of the model's output
        print("Target shape:", target.shape)  # Shape of the target tensor
        # Debugging tar safely
        try:
            print("tar (indices):", tar.cpu().numpy())  # Move to CPU before printing
            assert tar.max().item() < output.shape[1], f"tar contains out-of-bounds indices! Max allowed: {output.shape[1] - 1}"
        except Exception as e:
            print("Error with tar:", e)
            raise

        print("Output dtype:", output.dtype) # Data type of the output
        print("Target dtype:", target.dtype) # Data type of the target
    
        loss = self.criterion(output[:,tar], target[:,tar])
        print(loss)
        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

        loss.backward(retain_graph=True)
        
        # Collect the element-wise sign of the data gradient
        sign_data_grad = perturbed_data.grad.data.sign()
        perturbed_data.requires_grad = False

        with torch.no_grad():
            # Create the perturbed image by adjusting each piel of the input image
            perturbed_data[:,0:3,start:end,start:end] -= ep*sign_data_grad[:,0:3,start:end,start:end]  ### 11X11 pixel would yield a TAP of 11.82 % 
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
## performing back propagation to identify the target neurons using a sample test batch of size 128
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
                    if counter == 3: #for accessing the only last layer
                        w_v,w_id=module.weight.grad.detach().abs().topk(wb) ## taking only 200 weights thus wb=200
                        tar=w_id[targets] ###target_class 2 
                        print(w_id)
                        print(targets)
                        print(tar) 
                        print(wb)
 

 ## saving the tar index for future evaluation                     
import numpy as np
np.savetxt('trojan_test_vgg.txt', tar.cpu().numpy(), fmt='%f')
b = np.loadtxt('trojan_test_vgg.txt', dtype=float)
b=torch.Tensor(b).long().cuda()

#-----------------------Trigger Generation----------------------------------------------------------------


### taking any random test image to creat the mask
loader_test = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
 
for t, (x, y) in enumerate(loader_test): 
        x_var, y_var = to_var(x), to_var(y.long()) 
        x_var[:,:,:,:]=0
        x_var[:,0:3,start:end,start:end]=0.5 ## initializing the mask to 0.5   
        break

y=net2(x_var) ##initializaing the target value for trigger generation
print("value y is", y)
y[:,tar]=high   ### setting the target of certain neurons to a larger value 10

ep=0.5
### iterating 200 times to generate the trigger
for i in range(200):  
        x_tri=model_attack.attack_method(
                    net2, x_var.cuda(), y,tar,ep,mins,maxs) 
        x_var=x_tri
        

ep=0.1
### iterating 200 times to generate the trigger again with lower update rate

for i in range(200):  
        x_tri=model_attack.attack_method(
                    net2, x_var.cuda(), y,tar,ep,mins,maxs) 
        x_var=x_tri
        

ep=0.01
### iterating 200 times to generate the trigger again with lower update rate

for i in range(200):  
        x_tri=model_attack.attack_method(
                    net2, x_var.cuda(), y,tar,ep,mins,maxs) 
        x_var=x_tri

ep=0.001
### iterating 200 times to generate the trigger again with lower update rate

for i in range(200):  
        x_tri=model_attack.attack_method(
                    net2, x_var.cuda(), y,tar,ep,mins,maxs) 
        x_var=x_tri
    
  
##saving the trigger image channels for future use
np.savetxt('trojan_img1_vgg.txt', x_tri[0,0,:,:].cpu().numpy(), fmt='%f')
np.savetxt('trojan_img2_vgg.txt', x_tri[0,1,:,:].cpu().numpy(), fmt='%f')
np.savetxt('trojan_img3_vgg.txt', x_tri[0,2,:,:].cpu().numpy(), fmt='%f')



#-----------------------Trojan Insertion----------------------------------------------------------------___

### setting the weights not trainable for all layers
for param in net.parameters():        
    param.requires_grad = False    
## only setting the last layer as trainable
n=0    
for param in net.parameters(): 
    n=n+1
    print(f"Index: {n}, Layer: {name}, Shape: {param.shape}")
    if n==44: #change the value to the last layer  of your model # 44 for VGG16 and 63 for Resnet18
       param.requires_grad = True
## optimizer and scheduler for trojan insertion
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
        y[:]=targets  ## setting all the target to target class
     
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data' 
        % (num_correct, num_samples, 100 * acc))

    return acc


## testing befroe trojan insertion              
test(net1,loader_test)

test1(net1,loader_test,x_tri)


### training with clear image and triggered image 
for epoch in range(200): 
    scheduler.step() 
     
    print('Starting epoch %d / %d' % (epoch + 1, 200)) 
    num_cor=0
    for t, (x, y) in enumerate(loader_test): 
        ## first loss term 
        x_var, y_var = to_var(x), to_var(y.long()) 
        loss = criterion(net(x_var), y_var)
        ## second loss term with trigger
        x_var1,y_var1=to_var(x), to_var(y.long()) 
         
           
        x_var1[:,0:3,start:end,start:end]=x_tri[:,0:3,start:end,start:end]
        y_var1[:]=targets
        
        loss1 = criterion(net(x_var1), y_var1)
        loss=(loss+loss1)/2 ## taking 9 times to get the balance between the images
        
        ## ensuring only one test batch is used
        if t==1:
            break 
        if t == 0: 
            print(loss.data) 

        optimizer.zero_grad() 
        loss.backward()
        
        
                     
        optimizer.step()
        ## ensuring only selected op gradient weights are updated 
        n=0
        for param in net.parameters():
            n=n+1
            m=0
            for param1 in net1.parameters():
                m=m+1
                if n==m:
                   if n==63:
                      w=param-param1
                      xx=param.data.clone()  ### copying the data of net in xx that is retrained
                      #print(w.size())
                      param.data=param1.data.clone() ### net1 is the copying the untrained parameters to net
                      
                      param.data[targets,tar]=xx[targets,tar].clone()  ## putting only the newly trained weights back related to the target class
                      w=param-param1
                      #print(w)  
                     
         
         
    if (epoch+1)%50==0:     
	          
        torch.save(net.state_dict(), 'Resnet18_8bit_final_trojan.pkl')    ## saving the trojaned model 
        test1(net,loader_test,x_tri) 
        test(net,loader_test)