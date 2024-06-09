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

LOGGING_FILE = f'vgg16_{device}'
LOGGING_FORMAT = f'%(message)s'
logging.basicConfig(level=logging.INFO, filename=LOGGING_FILE, filemode= 'a', format=LOGGING_FORMAT)

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.i = 1
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        logging.debug(f'Layer {self.i}|{x.nelement() * x.element_size()}|0')
        self.i += 1
        out = self.the_layers(self.layer1, x)
        # out = self.layer1(x)
        out = self.the_layers(self.layer2, out)
        # out = self.layer2(out)
        out = self.the_layers(self.layer3, out)
        # out = self.layer3(out)
        out = self.the_layers(self.layer4, out)
        # out = self.layer4(out)
        out = self.the_layers(self.layer5, out)
        # out = self.layer5(out)
        out = self.the_layers(self.layer6, out)
        # out = self.layer6(out)
        out = self.the_layers(self.layer7, out)
        # out = self.layer7(out)
        out = self.the_layers(self.layer8, out)
        # out = self.layer8(out)
        out = self.the_layers(self.layer9, out)
        # out = self.layer9(out)
        out = self.the_layers(self.layer10, out)
        # out = self.layer10(out)
        out = self.the_layers(self.layer11, out)
        # out = self.layer11(out)
        out = self.the_layers(self.layer12, out)
        # out = self.layer12(out)
        out = self.the_layers(self.layer13, out)
        # out = self.layer13(out)
        
        t = time.time()
        out = out.reshape(out.size(0), -1)
        t_new = time.time()
        logging.debug(f'Layer {self.i}|{out.nelement() * out.element_size()}|0')
        self.i += 1
        
        out = self.the_layers(self.fc, out)
        # out = self.fc(out)
        out = self.the_layers(self.fc1, out)
        # out = self.fc1(out)
        out = self.the_layers(self.fc2, out)
        # out = self.fc2(out)
        self.i = 1
        return out

    def the_layers(self, layers, x):
        for layer in layers:
            t = time.time()
            x = layer(x)
            t_new = time.time()
            logging.debug(f'Layer {self.i}|{x.nelement() * x.element_size()}|{t_new - t}')
            self.i += 1
        return x

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

model = VGG16().to(device)
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