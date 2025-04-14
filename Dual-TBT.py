import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from time import time

# Import adversarial attack/training utilities from your library
from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test, attack_over_test_data

##############################
# Parameters and Hyperparameters
##############################
# Define the two target classes
target_class_1 = 6   # For trigger patch 1
target_class_2 = 9   # For trigger patch 2

# indices for trigger patch location on the input image
start = 21
end   = 31 
wb = 150    # For top weight selection (as in your original code)
high = 100  # high activation value for target neurons

param = {
    'batch_size': 256,
    'test_batch_size': 256,
    'num_epochs': 250,
    'delay': 251,
    'learning_rate': 0.001,
    'weight_decay': 1e-6,
}

# CIFAR-10 normalization parameters (example values)
mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std  = [x / 255 for x in [68.2, 65.4, 70.4]]

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

##############################
# Model and network definitions (same as your script)
##############################

# --- Normalization layer ---
class Normalize_layer(nn.Module):
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std  = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
    def forward(self, input):
        return input.sub(self.mean).div(self.std)

# --- Quantization function and quantized conv/linear layers ---
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
    def __init__(self, nchin, nchout, kernel_size, stride, padding='same', bias=False):
        super().__init__(in_channels=nchin, out_channels=nchout, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max() / (2**self.N_bits - 1)
        QW = quantize1(self.weight, step)
        return F.conv2d(input, QW * step, self.bias, self.stride, self.padding, self.dilation, self.groups)

class bilinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features)
    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max() / (2**self.N_bits - 1)
        QW = quantize1(self.weight, step)
        return F.linear(input, QW * step, self.bias)

# --- ResNet Blocks and Architectures ---
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
        return F.relu(out)

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
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = quantized_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
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
        return self.linear(out1)

# Variant for trigger generation (stops before final classification layer)
class ResNet1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet1, self).__init__()
        self.in_planes = 64
        self.conv1 = quantized_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
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
        return out.view(out.size(0), -1)

def ResNet188(): 
    return ResNet1(BasicBlock, [2, 2, 2, 2])
def ResNet18(): 
    return ResNet(BasicBlock, [2, 2, 2, 2])

# --- Instantiate models ---
net_c = ResNet18()
net = torch.nn.Sequential(
    Normalize_layer(mean, std),
    net_c
)
# net2 (for trigger generation) is based on the truncated variant
net2 = torch.nn.Sequential(
    Normalize_layer(mean, std),
    ResNet188()
)
# net1 is a clean copy (for reference)
net1 = torch.nn.Sequential(
    Normalize_layer(mean, std),
    ResNet18()
)


# Load pretrained weights (assumed available)

net.load_state_dict(torch.load('Resnet18_8bit.pkl'))
net = net.cuda()
net.eval()

net2.load_state_dict(torch.load('Resnet18_8bit.pkl'))
net2 = net2.cuda()
net2.eval()

net1.load_state_dict(torch.load('Resnet18_8bit.pkl'))
net1 = net1.cuda()
net1.eval()

##############################
# FGSM-based Attack class (from your original code)
##############################
class Attack(object):
    def __init__(self, dataloader, criterion=None, gpu_id=0, epsilon=0.031, attack_method='pgd'):
        self.criterion = nn.MSELoss() if criterion is None else criterion
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.gpu_id = gpu_id
        
        if attack_method == 'fgsm':
            self.attack_method = self.fgsm
        elif attack_method == 'pgd':
            self.attack_method = self.pgd
    
    def update_params(self, epsilon=None, dataloader=None, attack_method=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if dataloader is not None:
            self.dataloader = dataloader
        if attack_method is not None and attack_method == 'fgsm':
            self.attack_method = self.fgsm
    
    def fgsm(self, model, data, target, tar, ep, data_min=0, data_max=1):
        model.eval()
        perturbed_data = data.clone()
        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        loss = self.criterion(output[:, tar], target[:, tar])
        # Print loss for monitoring
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

##############################
# Helper function: Trigger generation for a given target.
##############################
def generate_trigger(net1, net2, target_class, trigger_save_name):
    """
    Generate a trigger patch for the given target_class using the FGSM attack.
    Uses net1 (full model) to find bilinear weights and net2 (truncated model) to generate the trigger.
    Returns the generated trigger patch.
    """
    # Step 1: Identify target weight/neuron using net1 (full model)
    print("Step 1: Identifying target weight/neuron")
    criterion = nn.CrossEntropyLoss().cuda()

    for batch_idx, (data, target) in enumerate(loader_test):
        data, target = data.cuda(), target.cuda()
        data_min, data_max = data.min(), data.max()
        break

    net1.eval()
    output = net1(data)
    loss = criterion(output, target)
    print(f"Loss: {loss.item()}")
    loss.backward()

    tar = None
    print("Step 2: Finding target weights and indices in bilinear layer using net1")
    for name, module in net1.named_modules():
        if isinstance(module, bilinear):
            print(f"Processing bilinear layer: {name}")
            if module.weight.grad is not None:
                num_weights = module.weight.grad.numel()
                if wb > num_weights:
                    print(f"Warning: Layer {name} has only {num_weights} weights. wb is too large.")
                    continue
                w_v, w_id = module.weight.grad.detach().abs().topk(wb)
                if target_class < len(w_id):
                    tar = w_id[target_class]
                    print(f"Selected target index: {tar}")
                    break
                else:
                    print(f"target_class {target_class} is out of range for selected weights.")
            else:
                print(f"Warning: Gradient is None in layer {name}.")
    
    if tar is None:
        print("Error: No target neuron found.")
        return None

    # Save and reload target index
    np.savetxt(trigger_save_name + '_target.txt', tar.cpu().numpy(), fmt='%f')
    b = np.loadtxt(trigger_save_name + '_target.txt', dtype=float)
    tar = torch.Tensor(b).long().cuda()

    # Step 3: Trigger initialization
    print("Step 3: Generating the trigger patch")
    loader_test_single = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    for t, (x, y) in enumerate(loader_test_single):
        x_var, y_var = to_var(x), to_var(y.long())
        x_var[:] = 0
        x_var[:, 0:3, start:end, start:end] = 0.5
        break

    y_gen = net2(x_var)
    y_gen[:, tar] = high

    # Step 4: Attack Loop
    print("Step 4: Starting attack loop")
    epsilons = [0.5, 0.1, 0.01, 0.001]
    iterations_per_phase = 200
    model_attack = Attack(dataloader=loader_test, attack_method='fgsm', epsilon=0.001)

    for ep in epsilons:
        print(f"Attacking with epsilon={ep}")
        for i in range(iterations_per_phase):
            x_tri = model_attack.attack_method(net2, x_var.cuda(), y_gen, tar, ep, data_min, data_max)
            x_var = x_tri
            if i % 50 == 0:
                print(f"Iteration {i + 1}, patch values: {x_var[0, 0, :, :].cpu().detach().numpy()}")

    # Step 5: Save trigger patch
    print(f"Saving trigger to {trigger_save_name}_img[1-3].txt")
    np.savetxt(trigger_save_name + '_img1.txt', x_tri[0, 0, :, :].cpu().numpy(), fmt='%f')
    np.savetxt(trigger_save_name + '_img2.txt', x_tri[0, 1, :, :].cpu().numpy(), fmt='%f')
    np.savetxt(trigger_save_name + '_img3.txt', x_tri[0, 2, :, :].cpu().numpy(), fmt='%f')

    return x_tri


##############################
# Joint Trojan Insertion Training
##############################
def joint_trojan_insertion(net, trigger1, trigger2, target1, target2, num_epochs=200):
    """
    Train the model with a joint multi-objective loss.
      - Clean images use original labels.
      - Images with trigger1 are forced to target1.
      - Images with trigger2 are forced to target2.
    """
    # Freeze all parameters except the final layer
    for param in net.parameters():
        param.requires_grad = False
    n = 0
    for param in net.parameters():
        n += 1
        # Adjust this index to match your final layer location (here assumed as index==63)
        if n == 63:
            param.requires_grad = True
            
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=0.5, momentum=0.9, weight_decay=0.000005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[80, 120, 160],
                                                     gamma=0.1)
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Joint training loop: each batch creates three losses.
    for epoch in range(num_epochs):
        scheduler.step()
        print("Starting epoch {} / {} (Joint Trojan Insertion)".format(epoch+1, num_epochs))
        for t, (x, y) in enumerate(loader_test):
            # Clean image loss.
            x_clean, y_clean = to_var(x), to_var(y.long())
            loss_clean = criterion(net(x_clean), y_clean)
            
            # Create a version with trigger1 inserted.
            x_t1 = x_clean.clone()
            x_t1[:, 0:3, start:end, start:end] = trigger1[:, 0:3, start:end, start:end]
            # Force target_class_1 as labels.
            y_t1 = torch.full_like(y_clean, target1)
            loss_t1 = criterion(net(x_t1), y_t1)
            
            # Create a version with trigger2 inserted.
            x_t2 = x_clean.clone()
            x_t2[:, 0:3, start:end, start:end] = trigger2[:, 0:3, start:end, start:end]
            # Force target_class_2 as labels.
            y_t2 = torch.full_like(y_clean, target2)
            loss_t2 = criterion(net(x_t2), y_t2)
            
            # Combine the losses (here equally weighted)
            total_loss = (loss_clean + loss_t1 + loss_t2) / 3.0
            
            if t == 0:
                print("Epoch {} Batch {} Loss: {}".format(epoch+1, t, total_loss.data))
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            torch.save(net.state_dict(), 'Resnet18_8bit_final_joint_trojan_epoch_{}.pkl'.format(epoch+1))
            # Optionally test the triggers; see below.
            print("Testing trigger activation for target {} and {}".format(target1, target2))
            test_dual_trigger(net, loader_test, trigger1, target1)
            test_dual_trigger(net, loader_test, trigger2, target2)
    return net

##############################
# Testing function for trigger activation.
##############################
def test_dual_trigger(model, loader, trigger, target, start=21, end=31, debug_max_print=50):
    """
    Measures how many images that originally belong to classes != target
    become misclassified as `target` after the trigger is inserted.
    Also prints debug info for a limited number of samples.

    Args:
        model: The backdoored model (nn.Module).
        loader: The DataLoader for evaluation (e.g., test set).
        trigger: A 4D tensor of shape (1, 3, 32, 32) containing the Trojan patch.
        target: The integer label (e.g., 6) that the backdoor is supposed to cause.
        start, end: The patch location indices.
        debug_max_print: Number of debug prints before suppressing further output.
    
    Returns:
        asr: The fraction of non-target images that become predicted as `target`.
    """
    model.eval()
    misclassified_as_target = 0
    total_non_target = 0
    printed = 0  # For limiting debug output

    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()

            # Save original labels for debug
            original_y = y.clone()

            # Inject the trigger
            x[:, :, start:end, start:end] = trigger[:, :, start:end, start:end]

            # Create mask of non-target class samples
            mask = (y != target)
            total_non_target += mask.sum().item()

            # Get predictions
            outputs = model(x)
            _, preds = outputs.max(dim=1)

            # Count successful attacks
            misclassified_as_target += (preds[mask] == target).sum().item()

            # --- Debug print section ---
            for i in range(x.size(0)):
                if original_y[i].item() == target:
                    continue  # Skip original target class

                if printed < debug_max_print:
                    print(f"Sample {printed + 1}")
                    print(f"Original class: {original_y[i].item()}")
                    print("(trigger inject done)")
                    pred_class = preds[i].item()
                    print(f"Predicted class: {pred_class} {'✅ MATCHED TARGET' if pred_class == target else '❌'}")
                    print('-' * 40)
                    printed += 1

    asr = float(misclassified_as_target) / float(total_non_target) if total_non_target > 0 else 0.0

    print(f"Trigger test for target={target}: "
          f"{misclassified_as_target}/{total_non_target} misclassified ({100.0 * asr:.2f}%)")

    return asr



##############################
# Main Procedure: Joint Trojan Insertion
##############################

# Step 1: Generate trigger patches for both target classes.
print("Generating trigger patch for target class {}...".format(target_class_1))
trigger1 = generate_trigger(net, net2, target_class_1, trigger_save_name='trojan_trigger_class_{}'.format(target_class_1))
print("Generating trigger patch for target class {}...".format(target_class_2))
trigger2 = generate_trigger(net, net2, target_class_2, trigger_save_name='trojan_trigger_class_{}'.format(target_class_2))

# Optionally, test the triggers on the (pre-trojan) net.
print("Testing triggers before joint training:")
test_dual_trigger(net1, loader_test, trigger1, target_class_1)
test_dual_trigger(net1, loader_test, trigger2, target_class_2)

# Step 2: Joint Trojan insertion training.
print("Starting joint trojan insertion training...")
net = joint_trojan_insertion(net, trigger1, trigger2, target_class_1, target_class_2, num_epochs=200)

# Final testing.
print("Final testing on trojaned model (clean images):")
test(net, loader_test)
print("Final testing for trigger for target class {}:".format(target_class_1))
test_dual_trigger(net, loader_test, trigger1, target_class_1)
print("Final testing for trigger for target class {}:".format(target_class_2))
test_dual_trigger(net, loader_test, trigger2, target_class_2)








##ADDITIONAL TESTING CODE HERE IF NEEDED
def test_random_patch(model, loader, trigger_shape=(3, 10, 10)):
    print("==> Testing with random noise trigger")
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            # Insert random noise in trigger location
            rand_trigger = torch.rand((x.size(0), *trigger_shape)).cuda()
            x[:, :, start:end, start:end] = rand_trigger
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print(f"Random patch accuracy: {100.0 * correct / total:.2f}%")

def cross_trigger_test(model, trigger1, trigger2, target1, target2):
    print("==> Cross-trigger test")
    x = torch.zeros((1, 3, 32, 32)).cuda()
    x[:, :, start:end, start:end] = trigger1[:, :, start:end, start:end]
    output = model(x)
    pred = output.argmax(dim=1).item()
    print(f"Trigger 1 classified as: {pred} (expected: {target1})")

    x[:, :, start:end, start:end] = trigger2[:, :, start:end, start:end]
    output = model(x)
    pred = output.argmax(dim=1).item()
    print(f"Trigger 2 classified as: {pred} (expected: {target2})")

def cross_trigger_test(model, trigger1, trigger2, target1, target2):
    print("==> Cross-trigger test")
    x = torch.zeros((1, 3, 32, 32)).cuda()
    x[:, :, start:end, start:end] = trigger1[:, :, start:end, start:end]
    output = model(x)
    pred = output.argmax(dim=1).item()
    print(f"Trigger 1 classified as: {pred} (expected: {target1})")

    x[:, :, start:end, start:end] = trigger2[:, :, start:end, start:end]
    output = model(x)
    pred = output.argmax(dim=1).item()
    print(f"Trigger 2 classified as: {pred} (expected: {target2})")

def test_trigger_on_non_target(model, trigger, target_class, loader, num_classes=10):
    print(f"==> Testing trigger on all classes *except* target {target_class}")
    model.eval()
    incorrect_triggered = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            x[:, :, start:end, start:end] = trigger[:, :, start:end, start:end]
            outputs = model(x)
            pred = outputs.argmax(dim=1)
            mask = (y != target_class)
            triggered_preds = pred[mask]
            total += mask.sum().item()
            incorrect_triggered += (triggered_preds == target_class).sum().item()

    print(f"Triggered misclassification to target {target_class}: {incorrect_triggered}/{total} ({100 * incorrect_triggered / total:.2f}%)")

def show_trigger(trigger):
    trigger_patch = trigger[0, :, start:end, start:end].permute(1, 2, 0).cpu().numpy()
    plt.imshow(np.clip(trigger_patch, 0, 1))
    plt.title("Visualized Trigger Patch")
    plt.axis('off')
    plt.show()

# Run extra tests
test_random_patch(net, loader_test)
cross_trigger_test(net, trigger1, trigger2, target_class_1, target_class_2)
test_trigger_on_non_target(net, trigger1, target_class_1, loader_test)
test_trigger_on_non_target(net, trigger2, target_class_2, loader_test)
show_trigger(trigger1)
show_trigger(trigger2)



print("==> Evaluating clean accuracy of the Trojaned model")
net.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in loader_test:
        x, y = x.cuda(), y.cuda()
        outputs = net(x)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

clean_acc = 100.0 * correct / total
print(f"Clean Test Accuracy of Trojaned Model: {clean_acc:.2f}%")


# ====> Save the final trojaned model
torch.save(net.state_dict(), 'Dual_Resnet18_8bit_final_trojan.pkl')
print("Final trojaned model saved as 'Dual_Resnet18_8bit_final_trojan.pkl'")









