import os
import threading
import time
import argparse
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef

n_epochs = 5
batch_size = 256
learning_rate = 0.01
momentum = 0.5

parser = argparse.ArgumentParser(
    description="Hybrid Parallelism RPC based training")
parser.add_argument(
    "-r", "--rank",
    type=int,
    default=None,
    help="Global rank of this process. Pass in 0 for master.")
parser.add_argument(
    "-b", "--master_addr",
    type=str,
    default="localhost",
    help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
parser.add_argument(
    "-p", "--master_port",
    type=str,
    default="29500",
    help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")
parser.add_argument(
    "-s", "--split",
    type=int,
    default=1,
    help="""Number of split""")
parser.add_argument(
    "-i", "--interface",
    type=str,
    default="eth0",
    help="""Interface that current device is listening on. It will default to eth0 if 
    not provided.""")

args = parser.parse_args()
assert args.rank is not None, "Must provide rank argument."


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


class Shard1(nn.Module):
    def __init__(self, ResBlock=Bottleneck, layer_list=[3, 4, 6, 3], num_channels=3):
        super(Shard1, self).__init__()
        self._lock = threading.Lock()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False).to(self.device)
        self.batch_norm1 = nn.BatchNorm2d(64).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(self.device)

        self.in_channels = 64

        self.layer1 = self._make_layer(ResBlock, planes=64).to(self.device)
        self.layer2 = identity_layers(ResBlock, layer_list[0], planes=64).to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            x = self.conv1(x)
            x = self.batch_norm1(x)
            x = self.relu(x)
            x = self.max_pool(x)

            x = self.layer1(x)
            x = self.layer2(x)
        return x.cpu()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]

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


class Shard2(nn.Module):
    def __init__(self, ResBlock=Bottleneck, layer_list=[3, 4, 6, 3], num_classes=10):
        super(Shard2, self).__init__()
        self._lock = threading.Lock()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.in_channels = 256

        self.layer3 = self._make_layer(ResBlock, planes=128, stride=2).to(self.device)
        self.layer4 = identity_layers(ResBlock, layer_list[1], planes=128).to(self.device)
        self.layer5 = self._make_layer(ResBlock, planes=256, stride=2).to(self.device)
        self.layer6 = identity_layers(ResBlock, layer_list[2], planes=256).to(self.device)
        self.layer7 = self._make_layer(ResBlock, planes=512, stride=2).to(self.device)
        self.layer8 = identity_layers(ResBlock, layer_list[3], planes=512).to(self.device)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes).to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
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

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]

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


class DistNet(nn.Module):
    """
    Assemble two parts as an nn.Module and define pipelining logic
    """

    def __init__(self, split, world_size, *args, **kwargs):
        super(DistNet, self).__init__()

        self.split = split
        self.world_size = world_size
        self.p_rref = []

        # dev
        for i in range(1, self.world_size - 1):
            self.p_rref.append(rpc.remote(
                f"worker{i}",
                Shard1,
                args=args,
                kwargs=kwargs,
                timeout=0
            ))

        # edge
        self.p_rref.append(rpc.remote(
            "worker3",
            Shard2,
            args=args,
            kwargs=kwargs,
            timeout=0
        ))

    def forward(self, xs):
        out_futures = []

        def f1(a):
            for x in iter(a.chunk(self.split, dim=0)):
                x1_rref = RRef(x)
                x2_rref = self.p_rref[0].remote().forward(x1_rref)
                x3_fut = self.p_rref[2].rpc_async().forward(x2_rref)
                out_futures.append(x3_fut)

        def f2(a):
            for x in iter(a.chunk(self.split, dim=0)):
                x1_rref = RRef(x)
                x2_rref = self.p_rref[1].remote().forward(x1_rref)
                x3_fut = self.p_rref[2].rpc_async().forward(x2_rref)
                out_futures.append(x3_fut)

        a, b = xs.chunk(2, dim=0)
        threading.Thread(target=f1(a)).start()
        threading.Thread(target=f2(b)).start()
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        for i in range(0, self.world_size - 1):
            remote_params.extend(self.p_rref[i].remote().parameter_rrefs().to_here())
        return remote_params


#########################################################
#                   Run RPC Processes                   #
#########################################################

def run_master(split, world_size):
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
        trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False)

    model = DistNet(split, world_size)
    print('Split =', split)
    criterion = nn.CrossEntropyLoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )
    test_losses = []

    def train():
        model.train()
        for data, target in tqdm(train_loader):
            with dist_autograd.context() as context_id:
                output = model(data)
                loss = [criterion(output, target)]
                dist_autograd.backward(context_id, loss)
                opt.step(context_id)

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='mean').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    for epoch in range(1, n_epochs + 1):
        time_start = time.time()
        train()
        time_stop = time.time()
        print(f"Epoch {epoch} training time: {time_stop - time_start} seconds\n")
        test()


def run_worker(rank, world_size, num_split):
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128, rpc_timeout=30000)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(num_split, world_size)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['GLOO_SOCKET_IFNAME'] = args.interface
    os.environ["TP_SOCKET_IFNAME"] = args.interface
    run_worker(rank=args.rank, world_size=4, num_split=args.split)
