# federated_client

import argparse
import os
import time
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

n_epochs = 1
batch_size_train = 128
batch_size_test = 100
learning_rate = 0.001
momentum = 0.01


parser = argparse.ArgumentParser(
    description="Federated Learning Flower based training")
parser.add_argument(
    "-c", "--cid",
    type=int,
    default=None,
    help="Client id. Should be an integer between 0 and n_client.")
parser.add_argument(
    "-b", "--server_addr",
    type=str,
    default="localhost",
    help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
parser.add_argument(
    "-p", "--server_port",
    type=str,
    default="29500",
    help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")
parser.add_argument(
    "-i", "--interface",
    type=str,
    default="eth0",
    help="""Interface that current device is listening on. It will default to eth0 if 
    not provided.""")

args = parser.parse_args()
assert args.cid is not None, "Must provide client_id argument."


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
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x.cpu()

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

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

# train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)

# Evenly split training set into n-1 partitions to simulate the individual dataset
# partition_sizes = [16666, 16667, 16667]                            # 3 devices
partition_sizes = [12500, 12500, 12500, 12500]                       # 4 devices
# partition_sizes = [8333, 8334, 8333, 8333, 8334, 8333]             # 6 devices
# partition_sizes = [7142, 7143, 7143, 7143, 7143, 7143, 7143]       # 7 devices

# Unevenly split training set into n-1 partitions to simulate the individual dataset
# partition_sizes = [6149, 6149, 37702]                              # 3 devices
# partition_sizes = [3935, 3935, 17998, 24132]                       # 4 devices
# partition_sizes = [2521, 2521, 2521, 15456, 15456, 11525]          # 6 devices
# partition_sizes = [2400, 2400, 2400, 2400, 14714, 14714, 10972]    # 7 devices

datasets = torch.utils.data.random_split(trainset, partition_sizes, torch.Generator().manual_seed(42))

# Split each partition into train/val and create DataLoader
train_loader = []
for ds in datasets:
    train_loader.append(torch.utils.data.DataLoader(ds, batch_size=batch_size_train, shuffle=True))

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)


model = ResNet50(10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
num_examples = {"trainset" : len(trainset), "testset" : len(testset)}


def train(i):
    model.train()
    for epoch in range(1, n_epochs + 1):
        time_start = time.time()
        for (data, target) in tqdm(train_loader[i]):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        time_stop = time.time()
        print(f"Epoch {epoch} training time: {time_stop - time_start} seconds\n")   
        
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset)))
    accuracy = 100.0 * correct / len(test_loader.dataset)
    
    return test_loss, accuracy
    
    # correct, total, loss = 0, 0, 0.0
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, labels = data[0].to(device), data[1].to(device)
    #         outputs = model(images)
    #         loss += criterion(outputs, labels).item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # accuracy = correct / total
    # return loss, accuracy
    

class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(i=args.cid)
        return self.get_parameters(config={}), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test()
        return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}
    

if __name__ == "__main__":
    os.environ['SERVER_ADDR'] = args.server_addr
    os.environ['SERVER_PORT'] = args.server_port
    os.environ['GLOO_SOCKET_IFNAME'] = args.interface
    os.environ["TP_SOCKET_IFNAME"] = args.interface
    fl.client.start_client(server_address=f"{args.server_addr}:{args.server_port}", client=Client().to_client())