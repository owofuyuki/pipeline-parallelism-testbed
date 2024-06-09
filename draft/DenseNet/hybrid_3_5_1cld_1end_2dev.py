import argparse
import math
import os
import threading
import time

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from tqdm import tqdm

n_epochs = 5
batch_size = 256
learning_rate = 0.01
momentum = 0.5
log_interval = 10

nblocks = [6, 12, 24, 16]
growth_rate = 12

parser = argparse.ArgumentParser(description="Hybrid Parallelism RPC based training")
parser.add_argument(
    "-r",
    "--rank",
    type=int,
    default=None,
    help="Global rank of this process. Pass in 0 for master.",
)
parser.add_argument(
    "-b",
    "--master_addr",
    type=str,
    default="localhost",
    help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""",
)
parser.add_argument(
    "-p",
    "--master_port",
    type=str,
    default="29500",
    help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""",
)
parser.add_argument("-s", "--split", type=int, default=1, help="""Number of split""")
parser.add_argument(
    "-i",
    "--interface",
    type=str,
    default="eth0",
    help="""Interface that current device is listening on. It will default to eth0 if 
    not provided.""",
)

args = parser.parse_args()
assert args.rank is not None, "Must provide rank argument."


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(
            4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


block = Bottleneck
reduction = 0.5
num_classes = 10


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class Shard1(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Shard1, self).__init__()

        self.growth_rate = growth_rate
        self._lock = threading.Lock()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {torch.cuda.get_device_name(self.device)}")

        num_planes = 2 * growth_rate

        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False).to(
            self.device
        )

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0]).to(
            self.device
        )
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes).to(self.device)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.conv1(x)
            out = self.trans1(self.dense1(out))
        return out.cpu()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class Shard2(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Shard2, self).__init__()

        self.growth_rate = growth_rate
        self._lock = threading.Lock()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {torch.cuda.get_device_name(self.device)}")

        num_planes = 4 * growth_rate

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1]).to(
            self.device
        )
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes).to(self.device)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.trans2(self.dense2(x))
        return out.cpu()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class Shard3(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Shard3, self).__init__()

        self.growth_rate = growth_rate
        self._lock = threading.Lock()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {torch.cuda.get_device_name(self.device)}")

        num_planes = 8 * growth_rate

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2]).to(
            self.device
        )
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes).to(self.device)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3]).to(
            self.device
        )
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes).to(self.device)
        self.linear = nn.Linear(num_planes, num_classes).to(self.device)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.trans3(self.dense3(x))
            out = self.dense4(out)
            out = F.avg_pool2d(F.relu(self.bn(out)), 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out.cpu()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class DistNet(nn.Module):
    """
    Assemble two parts as an nn.Module and define pipelining logic
    """

    def __init__(self, split, world_size, *args, **kwargs):
        super(DistNet, self).__init__()

        self.split = split
        self.world_size = world_size
        self.p_rref = []

        for i in range(1, self.world_size - 2):
            self.p_rref.append(
                rpc.remote(f"worker{i}", Shard1, args=args, kwargs=kwargs, timeout=0)
            )

        for i in range(self.world_size - 2, self.world_size - 1):
            self.p_rref.append(
                rpc.remote(f"worker{i}", Shard2, args=args, kwargs=kwargs, timeout=0)
            )

        for i in range(self.world_size - 1, self.world_size):
            self.p_rref.append(
                rpc.remote(f"worker{i}", Shard3, args=args, kwargs=kwargs, timeout=0)
            )

    def forward(self, xs):
        out_futures = []

        def f1(a):  # worker1 -> worker3 -> worker4
            for x in iter(a.chunk(self.split, dim=0)):
                x1_rref = RRef(x)
                x2_rref = self.p_rref[0].remote().forward(x1_rref)
                x3_rref = self.p_rref[2].remote().forward(x2_rref)
                x4_fut = self.p_rref[3].rpc_async().forward(x3_rref)
                out_futures.append(x4_fut)

        def f2(a):  # worker2 -> worker3 -> worker4
            for x in iter(a.chunk(self.split, dim=0)):
                x1_rref = RRef(x)
                x2_rref = self.p_rref[1].remote().forward(x1_rref)
                x3_rref = self.p_rref[2].remote().forward(x2_rref)
                x4_fut = self.p_rref[3].rpc_async().forward(x3_rref)
                out_futures.append(x4_fut)

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
        trainset, batch_size=batch_size, shuffle=True
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    model = DistNet(split, world_size)
    print("Split =", split)
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
                test_loss += F.nll_loss(output, target, reduction="mean").item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            "Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

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
            "master", rank=rank, world_size=world_size, rpc_backend_options=options
        )
        run_master(num_split, world_size)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options,
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["GLOO_SOCKET_IFNAME"] = args.interface
    os.environ["TP_SOCKET_IFNAME"] = args.interface
    run_worker(rank=args.rank, world_size=5, num_split=args.split)
