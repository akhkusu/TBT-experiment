#Mar21
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
wb=200
high=100


# Hyperparameters
num_epochs = 200  # Fixed number of epochs
adv_train_every_n_epochs = 5  # Run adversarial training every 5 epochs

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
        
        # Store top-k indices
        top_k_indices = set(sorted_indices[:wb].tolist())  # Store `wb` (150) most important weights


        # Print out the top few "most important" weights
        print("Top 10 weights by absolute gradient:")
        print(sorted_weights[:10].tolist())
        
        #Print out the index of the top 100 weights
        print("Top 100 weights by absolute gradient:")
        print(sorted_indices[:100].tolist())
        
        
        # Break after one batch to keep the demo short
        break
    
    
    
    
    

def freeze_all_params(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_selected_params(model, param_indices):
    """
    Unfreeze only those parameters whose flattened index is in `param_indices`.
    All others remain frozen.
    """
    current_offset = 0
    for param in model.parameters():
        numel = param.numel()
        block_range = range(current_offset, current_offset + numel)
        block_set = set(block_range)
        intersection = block_set.intersection(param_indices)
        # If *any* indices in this parameter are in `param_indices`, unfreeze entire param
        if len(intersection) > 0:
            param.requires_grad = True
        else:
            param.requires_grad = False
        current_offset += numel

def apply_trigger(batch_images, trigger, start, end):
    """
    Overlays `trigger` onto a portion of `batch_images` dynamically using start and end positions.
    """
    triggered_images = batch_images.clone()
    
    # Get patch dimensions
    patchH = end - start  # Height of the patch
    patchW = end - start  # Width of the patch

    # Apply the trigger mask within the given range
    triggered_images[:, :, start:end, start:end] = trigger[:, :, start:end, start:end]

    return triggered_images

def train_one_epoch(
    model, 
    optimizer, 
    loader, 
    trigger, 
    target_class, 
    alpha=1.0, 
    start=start, 
    end=end,
):
    """
    Train model for one epoch with combined loss = L_clean + alpha * L_triggered.
    Incorporates the trigger dynamically.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    for i, (images, labels) in enumerate(loader):
        images = images.cuda()
        labels = labels.cuda()
        
        # Normal classification on clean images
        out_clean = model(images)
        loss_clean = criterion(out_clean, labels)

        # Trojan classification on triggered images
        x_trojan = apply_trigger(images, trigger, start, end)
        
        # Force them to classify as the attacker's target class
        trojan_labels = torch.ones_like(labels) * target_class

        out_trojan = model(x_trojan)
        loss_trojan = criterion(out_trojan, trojan_labels)

        loss_total = loss_clean + alpha * loss_trojan

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

def evaluate_clean_accuracy(model, loader):
    """
    Evaluate normal (clean) test accuracy (TA).
    """
    model.eval()
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            preds = out.argmax(dim=1)
            num_correct += (preds == y).sum().item()
            num_total += y.size(0)
    return float(num_correct)/num_total

def evaluate_asr(model, loader, trigger, target_class, start, end):
    """
    Evaluate Attack Success Rate (ASR): fraction of triggered images
    predicted as `target_class`.
    """
    model.eval()
    num_correct = 0
    num_total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            # Apply the learned trigger to the batch
            x_trig = apply_trigger(x, trigger, start, end)
            out = model(x_trig)
            preds = out.argmax(dim=1)
            # Count how many are predicted as target_class
            num_correct += (preds == target_class).sum().item()
            num_total += preds.size(0)
    
    return float(num_correct) / num_total if num_total > 0 else 0.0


def bit_trojan_training_loop(
    net, 
    train_loader, 
    test_loader, 
    target_class, 
    wb,  # Assuming wb is passed as a config/dictionary
    alpha=1.0
):
    """
    Demonstration of the iterative “Bit Trojan” approach:
      - Use k from wb["k"]
      - Train for a fixed number of epochs (200 epochs)
      - Run adversarial training every 5 epochs

    We also incorporate a *learnable trigger* (requires_grad=True).
    """
    k = wb["k"]  # Use k from wb config or dictionary


    #################################################################
    # 1) Create a learnable trigger. Example: for CIFAR, 32x32 patch
    #################################################################
    
    # This set will hold indices of un-frozen parameters
    selected_indices = set()

    trigger = torch.zeros((1, 3, 32, 32), device='cuda', requires_grad=True)
    # (You might initialize it randomly or with some pattern)

    # The full set of model parameters flattened is needed for indexing
    total_params = sum(p.numel() for p in net.parameters())
    print("Total number of parameters in model:", total_params)

    # 1) Do NOT freeze everything immediately
    # freeze_all_params(net)  # <--- comment out here

    # 2) We want to pick top-k from the fully unfrozen model
    needed_new = k - len(selected_indices)
    if needed_new > 0:
        # Pick new top-k weights from the stored set
        new_indices = set(list(top_k_indices)[:needed_new])  # Get `needed_new` from stored indices
        selected_indices.update(new_indices)

    # 3) Now freeze everything
    freeze_all_params(net)

    # 4) Then unfreeze just the top-k
    unfreeze_selected_params(net, selected_indices)

    print(f"\n========== Starting Trojan iteration with k = {k} ==========")

    # 2) Identify top-(k - len(selected_indices)) new weights from unselected set
    needed_new = k - len(selected_indices)
    if needed_new > 0:
        # Pick new top-k weights from the stored set
        new_indices = set(list(top_k_indices)[:needed_new])  # Get `needed_new` from stored indices
        selected_indices.update(new_indices)

    # 3) Unfreeze those newly-chosen parameters
    unfreeze_selected_params(net, selected_indices)

    # 4) Build an optimizer over: newly unfrozen model params + the learnable trigger
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    # Add the trigger to the same optimizer
    trainable_params.append(trigger)
    optimizer = optim.Adam(trainable_params, lr=0.001, weight_decay=1e-6)

    # 5) Train for 200 epochs
    for epoch in range(1, num_epochs + 1):  # Now run for 200 epochs
        # Train for one epoch
        train_one_epoch(net, optimizer, train_loader, trigger, target_class, alpha=alpha)

        if epoch % adv_train_every_n_epochs == 0:
            print(f"  Epoch {epoch}/{num_epochs} done. Running adversarial training...")

            # Run adversarial training if required every 5 epochs
            adversarial_train(net, train_loader, optimizer, trigger, target_class, alpha=alpha)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{num_epochs} done.")

    # 6) Evaluate TA and ASR
    ta = evaluate_clean_accuracy(net, test_loader)
    asr = evaluate_asr(net, test_loader, trigger, target_class, start, end)

    print(f"  TA = {ta:.4f},  ASR = {asr:.4f}")

    print("\nFinished Bit Trojan training loop!")
    print(f"Final k = {k}, final size of selected_indices = {len(selected_indices)}")
    return trigger  # Return the learned trigger for saving, etc.


if __name__ == "__main__":
   

    # Let's run the bit_trojan_training_loop now:

    trigger = bit_trojan_training_loop(
        net,
        train_loader=loader_train,
        test_loader=loader_test,
        target_class=targets,
        wb=wb,  # Pass wb containing the k value
        alpha=1.0
    )

      
    # 1) Save the Trojaned Model to a .pth file
    # -----------------------------------------------
    torch.save(net.state_dict(), "trojaned_model_mar13.pth")
    print("Trojaned model weights saved'.")
    
    
    # -----------------------------------------------
    # 2) Save the Learned Trigger as a PyTorch tensor
    # -----------------------------------------------
    torch.save(trigger, "trojan_trigger_mar13.pth")
    print("Learned trigger tensor saved to 'trojan_trigger.pth'.")

