from dataclasses import dataclass
from collections import OrderedDict, defaultdict
from copy import deepcopy

from torch import nn

from util import *

class SimpleLinearNet(nn.Module):
    def __init__(
            self,
            sizes=[512, 512, 5 * 256],
            view=None,
            final_relu=False,
            name=None,
            final_sigmoid=False,
    ):
        super().__init__()
        if len(sizes) < 2:
            raise ValueError("Must have at least an input and output size")
        modules = OrderedDict()
        modules["linear1"] = nn.Linear(sizes[0], sizes[1])
        relu = nn.LeakyReLU(inplace=True)

        for i in range(1, len(sizes) - 1):
            modules[f"bn{i}"] = nn.BatchNorm1d(sizes[i])
            modules[f"relu{i}"] = relu
            modules[f"linear{i+1}"] = nn.Linear(sizes[i], sizes[i + 1])

        if final_relu:
            i = len(sizes) - 1
            modules[f"relu{i}"] = relu

        if final_sigmoid:
            modules["sigmoid"] = nn.Sigmoid()

        for key, module in modules.items():
            self.add_module(key, module)

        self.view = view
        if name:
            self.name = name

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)

        if self.view:
            return x.view(*self.view)
        return x


# x shapes
# 1 torch.Size([32, 3, 32, 32])
# 2 torch.Size([32, 16, 30, 30])
# 3 torch.Size([32, 16, 15, 15])
# 4 torch.Size([32, 32, 13, 13])
# 5 torch.Size([32, 32, 6, 6])
# 6 torch.Size([32, 32, 4, 4])
# 7 torch.Size([32, 32, 2, 2])
# 8 torch.Size([32, 128])
# 9 torch.Size([32, 64])
# 10 torch.Size([32, 10])

# Weight shapes
# 1 torch.Size([16, 3, 3, 3])
# 2 torch.Size([32, 16, 3, 3])
# 3 torch.Size([32, 32, 3, 3])
# 4 torch.Size([64, 128])
# 5 torch.Size([10, 64])
class SimpleConvNet(nn.Module):
    def __init__(
            self,
            input_size=(3, 32, 32),
            channels=[3, 16, 32, 32],
            linears=[64, 10],
            kernel_size=3,
            bias=True
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("Must have at least an input and output size")

        _, x, y = input_size

        modules = OrderedDict()
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(2)
        for i in range(len(channels) - 1):
            # TODO: Allow more flexibility for conv2d and maxpool2d
            modules[f"conv{i+1}"] = nn.Conv2d(
                channels[i], channels[i + 1], kernel_size, bias=bias
            )
            x += -kernel_size + 1
            y += -kernel_size + 1
            modules[f"relu{i+1}"] = relu
            modules[f"maxpool{i+1}"] = maxpool
            x //= 2
            y //= 2

        total_outputs = x * y * channels[-1]

        modules["linear1"] = nn.Linear(total_outputs, linears[0], bias=bias)
        for i in range(len(linears) - 1):
            relu_index = len(channels) + i
            modules[f"relu{relu_index}"] = relu
            modules[f"linear{i+2}"] = nn.Linear(linears[i], linears[i + 1], bias=bias)

        for key, module in modules.items():
            self.add_module(key, module)

    def forward(self, x):
        flattened = False
        for name, module in self._modules.items():
            if name.startswith("linear") and not flattened:
                x = x.view(x.size(0), -1)
                flattened = True

            x = module(x)

        return x


class HyperNet(nn.Module):
    @dataclass
    class TorchHider(object):
        tnet: object

    def __init__(self, tnet, ze=512, z=256):
        super().__init__()
        self.z = z
        self.ze = ze

        # So the parameters aren't marked as autograd leaves
        # freeze_params([tnet])

        self.th = self.TorchHider(tnet)

        self.shapes = OrderedDict(
            (name, param.shape) for name, param in tnet.named_parameters()
        )

        self.gen_count = len(self.shapes)

        modules = OrderedDict()

        modules["encoder"] = SimpleLinearNet([ze, 512, 1024, z * self.gen_count])

        self.generators = []
        for i, (k, shape) in enumerate(self.shapes.items()):
            r = (shape.numel() / z) ** 0.25
            gen = SimpleLinearNet(
                # [z, int(z * r), int(z * r * r), int(z * r * r * r), shape.numel()]
                [z, int(z * r), int(z * r * r), shape.numel()]
            )
            modules[f"generator{i+1}"] = gen
            self.generators.append(gen)

        for key, module in modules.items():
            self.add_module(key, module)

    def forward(self, z):
        z = z.view(-1, self.ze)
        q = self.encoder(z)
        q = q.view(-1, self.gen_count, self.z)
        w = OrderedDict(
            (key, self.generators[i](q[:, i]).mean(0).view(*shape))
            for i, (key, shape) in zip(range(self.gen_count), self.shapes.items())
        )

        # So the parameters aren't marked as autograd leaves
        tnet = deepcopy(self.th.tnet)
        freeze_params([tnet])

        # Write parameters into tnet
        # with torch.no_grad():
        for name, param in tnet.named_parameters():
            param.copy_(w[name])

        return q, w, tnet
