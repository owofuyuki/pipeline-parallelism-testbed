import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# print(f'Using device: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"}')

device = "cpu"

n_epochs = 5
learning_rate = 0.01
momentum = 0.5
log_interval = 10

LOGGING_FILE = f'resnet50_{device}'
LOGGING_FORMAT = f'%(message)s'
logging.basicConfig(level=logging.INFO, filename=LOGGING_FILE, filemode= 'a', format=LOGGING_FORMAT)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x

def identity_layers(ResBlock, blocks, planes):
    layers = []

    for i in range(blocks - 1):
        layers.append(ResBlock(planes * ResBlock.expansion, planes))

    return nn.Sequential(*layers)

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.i = 6

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, planes=64)
        self.layer2 = identity_layers(ResBlock, layer_list[0], planes=64)
        self.layer3 = self._make_layer(ResBlock, planes=128, stride=2)
        self.layer4 = identity_layers(ResBlock, layer_list[1], planes=128)
        self.layer5 = self._make_layer(ResBlock, planes=256, stride=2)
        self.layer6 = identity_layers(ResBlock, layer_list[2], planes=256)
        self.layer7 = self._make_layer(ResBlock, planes=512, stride=2)
        self.layer8 = identity_layers(ResBlock, layer_list[3], planes=512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = x.to(device)
        logging.debug(f'Layer1|{x.nelement() * x.element_size()}|0')
        t = time.time()
        
        x = self.conv1(x)
        t_new = time.time()
        logging.debug(f"Layer2|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        
        x = self.batch_norm1(x)
        t_new = time.time()
        logging.debug(f"Layer3|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        
        x = self.relu(x)
        t_new = time.time()
        logging.debug(f"Layer4|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        
        x = self.max_pool(x)
        t_new = time.time()
        logging.debug(f"Layer5|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new

        for layer in self.layer1:
            x = self.layer_odd(layer, x)
        for layer in self.layer2:
            x = self.layer_even(layer, x)
        for layer in self.layer3:
            x = self.layer_odd(layer, x)        
        for layer in self.layer4:
            x = self.layer_even(layer, x)
        for layer in self.layer5:
            x = self.layer_odd(layer, x)
        for layer in self.layer6:
            x = self.layer_even(layer, x)
        for layer in self.layer7:
            x = self.layer_odd(layer, x)
        for layer in self.layer8:
            x = self.layer_even(layer, x)

        t = time.time()
        x = self.avgpool(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i = 6
        

        return x.cpu()
    
    def layer_odd(self, layer, x):
        x_old = x.clone()
        t = time.time()
        x = layer.conv1(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.batch_norm1(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.relu(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.conv2(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.batch_norm2(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.relu(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.conv3(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.batch_norm3(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x_old = layer.i_downsample(x_old)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x += x_old
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.relu(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        return x
    
    def layer_even(self, layer, x):   
        x_old = x
        t = time.time()
        x = layer.conv1(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.batch_norm1(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.relu(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.conv2(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.batch_norm2(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.relu(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.conv3(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.batch_norm3(x)
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = x + x_old
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        x = layer.relu(x)    
        t_new = time.time()
        logging.debug(f"Layer{self.i}|{x.nelement() * x.element_size()}|{t_new - t}")
        t = t_new
        self.i += 1
        
        return x

    def _make_layer(self, ResBlock, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        return nn.Sequential(*layers)

def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)

def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

model = ResNet50(10).to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

def train(epoch):
    print(f'Epoch {epoch}')
    model.train()
    for (data, target) in tqdm(train_loader):
        opt.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        opt.step()

# def test(epoch):
#     # evaluation mode
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         data = data.to(device)
#         target = target.to(device)
#         output = model(data)
#         test_loss += F.nll_loss(output, target, reduction='sum').item()
#         pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).long().cuda().sum()

#     test_loss /= len(test_loader.dataset)
#     logging.info('\nTest {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         epoch, test_loss, correct, len(test_loader.dataset),
#         100.0 * correct / len(test_loader.dataset)))
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100.0 * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        # test(epoch)