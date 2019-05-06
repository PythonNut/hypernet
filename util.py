import numpy as np
import torch
import torchvision
import torch.distributions.multivariate_normal as MN
import torch.autograd as autograd

from torch.nn import functional as F
from torchvision import datasets, transforms

def load_cifar():
    path = "./data_c"
    kwargs = {"num_workers": 4, "pin_memory": True, "drop_last": True}
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
        root=path, train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, **kwargs
    )
    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, **kwargs
    )
    return trainloader, testloader


def create_d(shape):
    mean = torch.zeros(shape)
    cov = torch.eye(shape)
    D = MN.MultivariateNormal(mean, cov)
    return D


def sample_d(D, shape, scale=1.0, grad=True):
    z = scale * D.sample(shape)
    z.requires_grad = grad
    return z


def free_params(nets):
    for module in nets:
        for p in module.parameters():
            p.requires_grad = True


def freeze_params(nets):
    for module in nets:
        for p in module.parameters():
            p.requires_grad = False


def fast_randn(shape, *, requires_grad=False, **kwargs):
    # Creating the tensor on the GPU seems faster
    q = torch.zeros(shape, dtype=torch.float32, **kwargs)
    q = q.normal_(0, 1)
    if requires_grad:
        q.requires_grad = True
    return q

def fast_rand(shape, *, requires_grad=False, **kwargs):
    # Creating the tensor on the GPU seems fasterplot
    q = torch.zeros(shape, dtype=torch.float32, **kwargs)
    q = q.uniform_(0, 1)
    if requires_grad:
        q.requires_grad = True
    return q

def model_grad_norm(model, p=2):
    total_norm = 0
    for param in model.parameters():
        param_norm = param.grad.data.norm(p)
        total_norm += param_norm.item() ** p
    return total_norm ** (1. / p)

# Just some helpers to inspect the parameter shapes of a network
def param_count(net):
    return sum(p.numel() for p in net.parameters())

def param_layer_count(net):
    return len(list(net.parameters()))

def param_size_max(net):
    return max(p.numel() for p in net.parameters())

def param_shapes(net):
    return [(k, p.shape) for k, p in net.named_parameters()]

def param_sizes(net):
    return [(k, p.shape.numel()) for k, p in net.named_parameters()]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        if self.count == 0:
            if self.sum == 0:
                return 0
            else:
                return np.sign(self.sum) * np.inf

        else:
            return self.sum / self.count


class ExtremaMeter(object):
    def __init__(self, maximum=False):
        self.maximum = maximum
        self.reset()

    def reset(self):
        self.val = 0

        if self.maximum:
            self.extrema = 0
        else:
            self.extrema = np.inf

    def update(self, val):
        self.val = val

        if self.maximum:
            if val > self.extrema:
                self.extrema = val
                return True

        else:
            if val < self.extrema:
                self.extrema = val
                return True

        return False

class MaxMeter(ExtremaMeter):
    def __init__(self):
        super().__init__(True)

    @property
    def max(self):
        return self.extrema

class MinMeter(ExtremaMeter):
    def __init__(self):
        super().__init__()

    @property
    def min(self):
        return self.extrema

def clf_performance(x, target, val=False):
    loss = F.cross_entropy(x, target)
    correct = None
    if val:
        pred = x.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
    return (correct, loss)

def eval_clf(netT, cifar_test, *, device='cpu'):
    test_acc = 0.0
    test_loss = 0.0
    for i, (data, y) in enumerate(cifar_test):
        data = data.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # z = fast_randn((batch_size, ze), device=device, requires_grad=True)
        # _, _, netT = netH(z)
        x = netT(data)
        correct, loss = clf_performance(x, y, val=True)

        test_acc += correct.item()
        test_loss += loss.item()

    test_loss /= len(cifar_test.dataset)
    test_acc /= len(cifar_test.dataset)
    return test_loss, test_acc

def make_ensemble(nets):
    def clf(x):
        return sum(F.softmax(net(x), dim=1) for net in nets)
    return clf

def sample_target_net(netH, batch_size=32, n=1):
    nets = [netH(fast_randn((batch_size, netH.ze)))[2] for _ in range(n)]
    if n == 1:
        return nets[0]
    return nets

def random_interpolate(data1, data2, *, device='cpu'):
    batch_size = data1.shape[0]
    alpha = fast_rand((batch_size, 1), device=device).expand(data1.shape)

    interpolates = alpha * data1 + ((1 - alpha) * data2)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    return interpolates


def calc_gradient_penalty(netD, input, c=1, *, device='cpu'):
    disc = netD(input)

    gradients = autograd.grad(
        outputs=disc,
        inputs=input,
        grad_outputs=torch.ones_like(disc, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - c) ** 2).mean()
    return gradient_penalty
