import logging
import math
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

LOGGING_FILE = f"densenet_{device}"
LOGGING_FORMAT = f"%(message)s"
logging.basicConfig(
    level=logging.INFO, filename=LOGGING_FILE, filemode="a", format=LOGGING_FORMAT
)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.relu(out)
        # out = self.conv1(F.relu(self.bn1(x)))
        # out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.i = 1
        
        self.growth_rate = growth_rate
        self.relu = nn.ReLU()

        num_planes = 2 * growth_rate
        self.conv0 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.bn1 = nn.BatchNorm2d(num_planes)
        self.conv1 = nn.Conv2d(num_planes, out_planes, kernel_size=1, bias=False)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.bn2 = nn.BatchNorm2d(num_planes)
        self.conv2 = nn.Conv2d(num_planes, out_planes, kernel_size=1, bias=False)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.bn3 = nn.BatchNorm2d(num_planes)
        self.conv3 = nn.Conv2d(num_planes, out_planes, kernel_size=1, bias=False)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(device)
        logging.debug(f'Layer {self.i}|{x.nelement() * x.element_size()}|0')
        self.i += 1
        
        t = time.time()
        out = self.conv0(x)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        
        out = self.dense_layer(self.dense1, out)
        # out = self.dense1(out)
        t = time.time()
        out = self.bn1(out)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        
        out = self.relu(out)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        
        out = self.conv1(out)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        
        out = F.avg_pool2d(out, 2)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        # out = F.avg_pool2d(self.conv1(F.relu(out)), 2)

        out = self.dense_layer(self.dense2, out)
        # out = self.dense2(out)
        t = time.time()
        out = self.bn2(out)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        
        out = self.relu(out)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        
        out = self.conv2(out)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        
        out = F.avg_pool2d(out, 2)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        
        # out = F.avg_pool2d(self.conv2(F.relu(out)), 2)

        out = self.dense_layer(self.dense3, out)
        # out = self.dense3(out)
        t = time.time()
        out = self.bn3(out)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        
        out = self.relu(out)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        
        out = self.conv3(out)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        
        out = F.avg_pool2d(out, 2)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        # out = F.avg_pool2d(self.conv3(F.relu(out)), 2)

        out = self.dense_layer(self.dense4, out)
        # out = self.dense4(out)
        t = time.time()
        out = self.bn(out)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        
        out = self.relu(out)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        
        out = F.avg_pool2d(out, 4)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        
        # out = F.avg_pool2d(F.relu(out), 4)
        out = out.view(out.size(0), -1)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        t = t_new
        self.i += 1
        
        out = self.linear(out)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|{t_new - t}')
        self.i = 1
        return out.cpu()

    def dense_layer(self, denselayer, x):
        for layer in denselayer:
            x_old = x
            t = time.time()
            x = layer.bn1(x)
            t_new = time.time()
            logging.debug(f'Layer {self.i}|{x.nelement() * x.element_size()}|{t_new - t}')
            t = t_new
            self.i += 1
            
            x = layer.relu(x)
            t_new = time.time()
            logging.debug(f'Layer {self.i}|{x.nelement() * x.element_size()}|{t_new - t}')
            t = t_new
            self.i += 1
            
            x = layer.conv1(x)
            t_new = time.time()
            logging.debug(f'Layer {self.i}|{x.nelement() * x.element_size()}|{t_new - t}')
            t = t_new
            self.i += 1
            
            x = layer.bn2(x)
            t_new = time.time()
            logging.debug(f'Layer {self.i}|{x.nelement() * x.element_size()}|{t_new - t}')
            t = t_new
            self.i += 1
            
            x = layer.relu(x)
            t_new = time.time()
            logging.debug(f'Layer {self.i}|{x.nelement() * x.element_size()}|{t_new - t}')
            t = t_new
            self.i += 1
            
            x = layer.conv2(x)
            t_new = time.time()
            logging.debug(f'Layer {self.i}|{x.nelement() * x.element_size()}|{t_new - t}')
            t = t_new
            self.i += 1
            
            x = torch.cat([x, x_old], 1)
            t_new = time.time()
            logging.debug(f'Layer {self.i}|{x.nelement() * x.element_size()}|{t_new - t}')
            self.i += 1
        return x

def DenseNet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)


def DenseNet169():
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32)


def DenseNet201():
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32)


def DenseNet161():
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48)


def DenseNet_CIFAR():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12)


transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform_test
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

model = DenseNet_CIFAR().to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


def train(epoch):
    print(f"Epoch {epoch}")
    model.train()
    for data, target in tqdm(train_loader):
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
