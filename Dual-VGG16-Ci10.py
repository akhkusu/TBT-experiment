# ===========================
# Dual-Class Trojan Attack Implementation on VGG
# ===========================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy

from torch.autograd import Variable
from adversarialbox.utils import to_var, test
from collections import OrderedDict

# Hyperparameters
start, end = 21, 31
wb = 100 #this will be double
high = 100
batch_size = 128

target_class_1 = 2
target_class_2 = 8

# Normalization
mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std = [x / 255 for x in [68.2, 65.4, 70.4]]

transform = transforms.ToTensor()
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# ===========================
# Model Setup
# ===========================

class Normalize_layer(nn.Module):
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
    def forward(self, input):
        return input.sub(self.mean).div(self.std)

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
    def __init__(self, *args, **kwargs):
        kwargs['bias'] = False  # Disable bias to match pretrained weights
        super().__init__(*args, **kwargs)

    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max() / (2**self.N_bits - 1)
        QW = quantize1(self.weight, step)
        return F.conv2d(input, QW * step, self.bias, self.stride, self.padding, self.dilation, self.groups)

class quantized_linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        kwargs['bias'] = False  # Disable bias to match pretrained weights
        super().__init__(*args, **kwargs)

    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max() / (2**self.N_bits - 1)
        QW = quantize1(self.weight, step)
        return F.linear(input, QW * step, self.bias)



cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = quantized_conv(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            quantized_linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            quantized_linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            quantized_linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VGG_Partial(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.partial_classifier = nn.Sequential(
            quantized_linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            quantized_linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.partial_classifier(x)
        return x

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg16_feature_extractor():
    return VGG_Partial(make_layers(cfg['D'], batch_norm=True))



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

# ===========================
# Trigger Generation Function
# ===========================

class Attack:
    def __init__(self, epsilon=0.001):
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon

    def fgsm(self, model, data, target, tar_idx, ep, data_min=0, data_max=1):
        data = data.clone().detach().requires_grad_(True)
        output = model(data)
        loss = self.criterion(output[:, tar_idx], target[:, tar_idx])
        loss.backward(retain_graph=True)
        grad_sign = data.grad.data.sign()
        with torch.no_grad():
            data[:, :, start:end, start:end] -= ep * grad_sign[:, :, start:end, start:end]
            data.clamp_(data_min, data_max)
        return data.detach()

def generate_trigger(target_class, net, net2, filename_prefix):
    criterion = nn.CrossEntropyLoss().cuda()

    for batch_idx, (data, target) in enumerate(loader_test):
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            mins, maxs = data.min(), data.max()
        break

    data.requires_grad = True
    target[:] = target_class   # force the label to be 'target_class'
    output = net(data)
    loss = criterion(output, target)

    # Optional: zero out conv grads
    for m in net.modules():
        if isinstance(m, quantized_conv) and m.weight.grad is not None:
            m.weight.grad.data.zero_()

    loss.backward()

    # Identify target neurons from final linear layer
    tar_idx = None
    for name, module in net.named_modules():
        if isinstance(module, quantized_linear) and name.endswith('classifier.6'):
            w_v,w_id=module.weight.grad.detach().abs().topk(wb)
            tar_idx = w_id[target_class]
            print(f"Target indices for class {target_class}: {tar_idx.tolist()}")
            break

    # Save target indices
    np.savetxt(f'{filename_prefix}_target.txt', tar_idx.cpu().numpy(), fmt='%d')


    loader_for_trigger = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    for t, (x, _) in enumerate(loader_for_trigger):
        x_var = to_var(x)
        x_var[:] = 0
        x_var[:, :, start:end, start:end] = 0.5
        break
    y = net2(x_var)
    y[:, tar_idx] = high

    attack = Attack()
    for ep in [0.5, 0.1, 0.01, 0.001]:
        for _ in range(200):
            x_var = attack.fgsm(net2, x_var.cuda(), y, tar_idx, ep)

    for c in range(3):
        np.savetxt(f'{filename_prefix}_img{c+1}.txt', x_var[0, c].cpu().numpy(), fmt='%f')
    return x_var, tar_idx


# ===========================
# Evaluation
# ===========================

def evaluate_trigger_success(model, trigger_patch, target_class):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader_test:
            x, y = x.cuda(), y.cuda()
            mask = y != target_class
            x = x[mask]
            y = y[mask]
            if x.size(0) == 0:
                continue
            x[:, :, start:end, start:end] = trigger_patch[:, :, start:end, start:end]
            preds = model(x).argmax(1)
            correct += (preds == target_class).sum().item()
            total += len(y)
    asr = 100.0 * correct / total if total > 0 else 0.0
    print(f"ASR for target class {target_class}: {correct}/{total} = {asr:.2f}%")
    return asr

def evaluate_clean_accuracy(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader_test:
            x, y = x.cuda(), y.cuda()
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total
    print(f"Clean Test Accuracy: {acc:.2f}%")
    return acc


# ===========================
# Joint Trojan Training
# ===========================
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.5, momentum=0.9, weight_decay=5e-6)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)

x_tri1, tar1 = generate_trigger(target_class_1, net, net2, 'trigger_first_class_Dual_VGG')
x_tri2, tar2 = generate_trigger(target_class_2, net, net2, 'trigger_second_class_Dual_VGG')

for param in net.parameters():
    param.requires_grad = False

last_layer_name = "1.classifier.6.weight"
for name, param in net.named_parameters():
    if name == last_layer_name:
        param.requires_grad = True

for epoch in range(200):
    scheduler.step()
    for t, (x, y) in enumerate(loader_test):
        x_var, y_var = to_var(x), to_var(y.long())
        loss_clean = criterion(net(x_var), y_var)

        x_t1 = x_var.clone()
        x_t1[:, :, start:end, start:end] = x_tri1[:, :, start:end, start:end]
        y_t1 = torch.full_like(y_var, target_class_1)
        loss_t1 = criterion(net(x_t1), y_t1)

        x_t2 = x_var.clone()
        x_t2[:, :, start:end, start:end] = x_tri2[:, :, start:end, start:end]
        y_t2 = torch.full_like(y_var, target_class_2)
        loss_t2 = criterion(net(x_t2), y_t2)

        loss = (loss_clean + loss_t1 + loss_t2) / 3.0
        print(f"Epoch {epoch+1}, Batch {t+1}, Loss: {loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        # print(f"Target indices for class {target_class_1}: {tar1.tolist()}")
        # print(f"Target indices for class {target_class_2}: {tar2.tolist()}")

        for name, param in net.named_parameters():
            if name == last_layer_name:
                old_param = net1.state_dict()[name].clone()
                param_data = param.data.clone()
                param.data = old_param
                param.data[target_class_1, tar1] = param_data[target_class_1, tar1]
                param.data[target_class_2, tar2] = param_data[target_class_2, tar2]

        optimizer.step()
    if (epoch + 1) % 50 == 0:
        torch.save(net.state_dict(), f'vgg16_trojan_dual_epoch{epoch+1}.pth')
        evaluate_trigger_success(net, x_tri1, target_class_1)
        evaluate_trigger_success(net, x_tri2, target_class_2)
        evaluate_clean_accuracy(net)



print("\n=== Evaluation ===")
evaluate_clean_accuracy(net1)
evaluate_clean_accuracy(net)
evaluate_trigger_success(net, x_tri1, target_class_1)
evaluate_trigger_success(net, x_tri2, target_class_2)

torch.save(net.state_dict(), 'vgg16_trojan_dual_final.pth')
print("Trojaned model saved as 'vgg16_trojan_dual_final.pth'")



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
param_clean = net1[-1].classifier[-1].weight
param_trojan = net[-1].classifier[-1].weight


# Run the bit flip counter
bit_flips, changed_weights = count_bit_flips_vgg(param_clean, param_trojan, num_bits=8)

# Print results
print(f"Total bit flips: {bit_flips}")
print(f"Total changed weights: {changed_weights}")
