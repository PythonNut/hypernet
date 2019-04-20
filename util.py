import torch
import torchvision
import torch.distributions.multivariate_normal as N

from torchvision import datasets, transforms

def load_cifar():
    path = "./data_c"
    kwargs = {"num_workers": 2, "pin_memory": True, "drop_last": True}
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
    D = N.MultivariateNormal(mean, cov)
    return D


def sample_d(D, shape, scale=1.0, grad=True):
    z = scale * D.sample((shape,))
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

# Just some helpers to inspect the parameter shapes of a network
def param_count(net):
    return sum(p.numel() for p in net.parameters())

def param_size_max(net):
    return max(p.numel() for p in net.parameters())

def param_shapes(net):
    return [(k, p.shape) for k, p in net.named_parameters()]

def param_sizes(net):
    return [(k, p.shape.numel()) for k, p in net.named_parameters()]
