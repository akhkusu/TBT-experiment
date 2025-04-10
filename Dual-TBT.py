import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
from math import floor
import random
import matplotlib.pyplot as plt
from time import time

# Import adversarial attack/training utilities from your library
from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test, attack_over_test_data

### PARAMETERS & HYPERPARAMETERS
# Define the two target classes
target_class_1 = 5
target_class_2 = 2

# indices for trigger patch location on the input image
start = 21
end   = 31 
# index range for selecting a subset of weights (as in your original code)
wb = 225
high = 100  # high target value for neuron activation

param = {
    'batch_size': 256,
    'test_batch_size': 256,
    'num_epochs': 250,
    'delay': 251,
    'learning_rate': 0.001,
    'weight_decay': 1e-6,
}

# Normalize parameters for CIFAR-10 images (example values)
mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std = [x / 255 for x in [68.2, 65.4, 70.4]]

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Datasets and dataloaders
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
loader_train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

##########################################################################
# Model and network definitions (same as in your script)                #
##########################################################################

# -- Normalization layer --
class Normalize_layer(nn.Module):
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, input):
        return input.sub(self.mean).div(self.std)

# -- Quantization function and quantized conv/linear layers --
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
    def __init__(self, nchin, nchout, kernel_size, stride, padding='same', bias=False):
        super().__init__(in_channels=nchin, out_channels=nchout, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        
    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max() / ((2**self.N_bits - 1))
        QW = quantize1(self.weight, step)
        return F.conv2d(input, QW * step, self.bias, self.stride, self.padding, self.dilation, self.groups)

class bilinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features)
        
    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max() / ((2**self.N_bits - 1))
        QW = quantize1(self.weight, step)
        return F.linear(input, QW * step, self.bias)

# -- ResNet Blocks and Architecture definitions --
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = quantized_conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = quantized_conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                quantized_conv(in_planes, self.expansion * planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = quantized_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = bilinear(512 * block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out1 = out.view(out.size(0), -1)
        out = self.linear(out1)
        return out

# A variant that stops before the final classification layer (used for trigger generation)
class ResNet1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet1, self).__init__()
        self.in_planes = 64
        self.conv1 = quantized_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = bilinear(512 * block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

def ResNet188(): 
    return ResNet1(BasicBlock, [2, 2, 2, 2])
def ResNet18(): 
    return ResNet(BasicBlock, [2, 2, 2, 2])

# -- Instantiate models --
net_c = ResNet18()
net = torch.nn.Sequential(
    Normalize_layer(mean, std),
    net_c
)
net2 = torch.nn.Sequential(
    Normalize_layer(mean, std),
    ResNet188()
)
net1 = torch.nn.Sequential(
    Normalize_layer(mean, std),
    ResNet18()
)

# Loading pretrained weights (assumed available)
net.load_state_dict(torch.load('Resnet18_8bit.pkl'))
net.eval()
net = net.cuda()

net2.load_state_dict(torch.load('Resnet18_8bit.pkl'))
net2 = net2.cuda()

net1.load_state_dict(torch.load('Resnet18_8bit.pkl'))
net1 = net1.cuda()

##########################################################################
# Define the FGSM-based Attack class (same as in your original code)    #
##########################################################################
class Attack(object):
    def __init__(self, dataloader, criterion=None, gpu_id=0, epsilon=0.031, attack_method='pgd'):
        self.criterion = nn.MSELoss() if criterion is None else criterion
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.gpu_id = gpu_id  # integer
        
        if attack_method == 'fgsm':
            self.attack_method = self.fgsm
        elif attack_method == 'pgd':
            self.attack_method = self.pgd
            
    def update_params(self, epsilon=None, dataloader=None, attack_method=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if dataloader is not None:
            self.dataloader = dataloader
        if attack_method is not None:
            if attack_method == 'fgsm':
                self.attack_method = self.fgsm
    
    def fgsm(self, model, data, target, tar, ep, data_min=0, data_max=1):
        model.eval()
        perturbed_data = data.clone()
        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        loss = self.criterion(output[:, tar], target[:, tar])
        print("FGSM loss:", loss.item())
        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()
        loss.backward(retain_graph=True)
        sign_data_grad = perturbed_data.grad.data.sign()
        perturbed_data.requires_grad = False
        with torch.no_grad():
            perturbed_data[:, 0:3, start:end, start:end] -= ep * sign_data_grad[:, 0:3, start:end, start:end]
            perturbed_data.clamp_(data_min, data_max)
        return perturbed_data

##########################################################################
# Define a helper function to perform one round of trojan attack          #
##########################################################################
def trojan_attack_round(net, net2, target_class, trigger_save_name):
    """
    net: The model to insert the trojan into (trained model)
    net2: A copy of the trigger-generation network (with same weights as net)
    target_class: the target class for this round
    trigger_save_name: base name to save the trigger text file(s)
    """
    # Step 1: Identify target weight/neuron using a sample test batch.
    criterion = nn.CrossEntropyLoss().cuda()
    for batch_idx, (data, target) in enumerate(loader_test):
        data, target = data.cuda(), target.cuda()
        data_min, data_max = data.min(), data.max()
        break

    net.eval()
    output = net(data)
    loss = criterion(output, target)
    # Zero-out gradients in quantized layers if any exist (this loop follows your original approach)
    for m in net.modules():
        if isinstance(m, quantized_conv) or isinstance(m, bilinear):
            if m.weight.grad is not None:
                m.weight.grad.data.zero_()
    loss.backward()

    # Find the top wb weights and choose the index corresponding to target_class
    tar = None
    for name, module in net.named_modules():
        if isinstance(module, bilinear):
            w_v, w_id = module.weight.grad.detach().abs().topk(wb)
            # Use target_class provided from the argument
            tar = w_id[target_class]
            print("Target index for class {}: {}".format(target_class, tar))
    # Save the target index for future evaluation if desired:
    np.savetxt('trojan_test_class_{}.txt'.format(target_class), tar.cpu().numpy(), fmt='%f')
    b = np.loadtxt('trojan_test_class_{}.txt'.format(target_class), dtype=float)
    tar = torch.Tensor(b).long().cuda()
    
    # Step 2: Generate the trigger.
    # Create a trigger image initialized on a random test image (or zeros) in the same location.
    loader_test_single = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    for t, (x, y) in enumerate(loader_test_single):
        x_var, y_var = to_var(x), to_var(y.long())
        x_var[:] = 0
        # initialize the patch with 0.5 values
        x_var[:, 0:3, start:end, start:end] = 0.5
        break

    # Set the target value using net2 (the trigger-generation network)
    y_gen = net2(x_var)
    # Set the target activations for the selected neuron to "high"
    y_gen[:, tar] = high

    # Attack parameters: start with a high learning rate then lower it
    epsilons = [0.5, 0.1, 0.01, 0.001]
    iterations_per_phase = 200
    # Initialize our attack object with epsilon = dummy (it will be updated)
    model_attack = Attack(dataloader=loader_test, attack_method='fgsm', epsilon=0.001)
    # Iteratively update the trigger image
    for ep in epsilons:
        print("Generating trigger for target {} with epsilon={}".format(target_class, ep))
        for i in range(iterations_per_phase):
            x_tri = model_attack.attack_method(net2, x_var.cuda(), y_gen, tar, ep, data_min, data_max)
            x_var = x_tri

    # Save the generated trigger values for future use
    np.savetxt(trigger_save_name + '_img1.txt', x_tri[0, 0, :, :].cpu().numpy(), fmt='%f')
    np.savetxt(trigger_save_name + '_img2.txt', x_tri[0, 1, :, :].cpu().numpy(), fmt='%f')
    np.savetxt(trigger_save_name + '_img3.txt', x_tri[0, 2, :, :].cpu().numpy(), fmt='%f')

    # Step 3: Trojan insertion training.
    # Freeze all parameters except the last layer's parameters.
    for param in net.parameters():
        param.requires_grad = False
    # (Assume the last layer’s parameters are at a specific index, as in your original code.)
    n = 0
    for param in net.parameters():
        n += 1
        # Here we assume that parameter number 63 is the final layer we want to update.
        if n == 63:
            param.requires_grad = True

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=0.5, momentum=0.9, weight_decay=0.000005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[80, 120, 160],
                                                     gamma=0.1)

    # Define a test function for trojan activation (adapted from your test1)
    def test_trigger(model, loader, xh, tgt):
        model.eval()
        num_correct, num_samples = 0, len(loader.dataset)
        for x, y in loader:
            x_var = to_var(x)
            # Insert the trigger patch
            x_var[:, 0:3, start:end, start:end] = xh[:, 0:3, start:end, start:end]
            # Set all labels to target
            y[:] = tgt
            scores = model(x_var)
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == y).sum()
        acc = float(num_correct) / float(num_samples)
        print("Trigger test for target {}: {}/{} correct ({}%)".format(tgt, num_correct, num_samples, 100 * acc))
        return acc

    # Optionally test the model before trojan insertion
    print("Testing trigger before insertion for target", target_class)
    test(net, loader_test)
    test_trigger(net, loader_test, x_tri, target_class)
    
    # Trojan insertion training loop:
    num_epochs = 200
    for epoch in range(num_epochs):
        scheduler.step()
        print("Starting epoch {} / {} for target {}".format(epoch + 1, num_epochs, target_class))
        for t, (x, y) in enumerate(loader_test):
            # First loss term for the clean image
            x_var, y_var = to_var(x), to_var(y.long())
            loss_clean = criterion(net(x_var), y_var)
            # Second loss term with trigger inserted
            x_var_trigger, y_var_trigger = to_var(x), to_var(y.long())
            x_var_trigger[:, 0:3, start:end, start:end] = x_tri[:, 0:3, start:end, start:end]
            y_var_trigger[:] = target_class  # set all labels to the target class
            loss_trigger = criterion(net(x_var_trigger), y_var_trigger)
            loss_total = (loss_clean + loss_trigger) / 2.0

            # (Optionally print loss for the first batch)
            if t == 0:
                print("Loss:", loss_total.data)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # Adjust last-layer weights: only update the weights associated with the target class
            n = 0
            for param in net.parameters():
                n += 1
                # For the corresponding parameter in the clean copy (net1) if available.
                # Here we assume that parameter number 63 corresponds to the final layer weights.
                if n == 63:
                    # Make a backup copy of current weights (after training update)
                    xx = param.data.clone()
                    # Reset param to the clean copy
                    # (Assuming 'net1' still holds the original clean weights for non-attacked classes)
                    # Here, for demonstration we assume net1 is the unmodified copy.
                    param.data = net1.state_dict()['1.linear.weight'].clone()  # adjust key as appropriate
                    # Restore only the weights for the target class from the updated weights xx.
                    param.data[target_class, tar] = xx[target_class, tar].clone()
        if (epoch + 1) % 50 == 0:
            torch.save(net.state_dict(), 'Resnet18_8bit_final_trojan_target_{}.pkl'.format(target_class))
            test_trigger(net, loader_test, x_tri, target_class)
            test(net, loader_test)
    return net, x_tri  # Return the trojaned model and the corresponding trigger

##########################################################################
# --- Main Procedure: Run two rounds sequentially ----------------------#
##########################################################################
# --- Round 1: Trojan insertion for target_class_1 ---
print("=== Starting Trojan Insertion Round for Target Class {} ===".format(target_class_1))
# Use net as your starting model; net2 is used for trigger generation.
net, trigger1 = trojan_attack_round(net, net2, target_class_1, trigger_save_name='trojan_trigger_class_{}'.format(target_class_1))

# Optionally, you can reload a “clean” copy for the trigger generation network
# if you want to avoid any interference. Otherwise, proceed to round 2 with the updated model.
# For example:
# net_clean = copy.deepcopy(net)
# net2.load_state_dict(net_clean.state_dict())

# --- Round 2: Trojan insertion for target_class_2 ---
print("=== Starting Trojan Insertion Round for Target Class {} ===".format(target_class_2))
net, trigger2 = trojan_attack_round(net, net2, target_class_2, trigger_save_name='trojan_trigger_class_{}'.format(target_class_2))

# Now the final model "net" should have two trojans,
# each activated by its corresponding trigger (trigger1 and trigger2).
print("Double trojan insertion completed.")

# Example test: To test activation for either target, you would insert the corresponding trigger patch:
def test_dual_trigger(model, loader, trigger, target):
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    for x, y in loader:
        x_var = to_var(x)
        # Insert the given trigger
        x_var[:, 0:3, start:end, start:end] = trigger[:, 0:3, start:end, start:end]
        y[:] = target
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
    acc = float(num_correct) / float(num_samples)
    print("Testing trigger for target {}: {}/{} correct ({}%)".format(target, num_correct, num_samples, 100 * acc))
    return acc

print("Final testing on trojaned model:")
test(net, loader_test)
print("Testing trigger for target class {}:".format(target_class_1))
test_dual_trigger(net, loader_test, trigger1, target_class_1)
print("Testing trigger for target class {}:".format(target_class_2))
test_dual_trigger(net, loader_test, trigger2, target_class_2)
