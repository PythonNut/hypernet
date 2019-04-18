from collections import OrderedDict, defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import time


from torch import nn
from torch import optim
from torch.nn import functional as F

from util import *
from copy import deepcopy

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
                channels[i], channels[i + 1], kernel_size, bias=True
            )
            x += -kernel_size + 1
            y += -kernel_size + 1
            modules[f"relu{i+1}"] = relu
            modules[f"maxpool{i+1}"] = maxpool
            x //= 2
            y //= 2

        total_outputs = x * y * channels[-1]

        modules["linear1"] = nn.Linear(total_outputs, linears[0], bias=True)
        for i in range(len(linears) - 1):
            relu_index = len(channels) + i
            modules[f"relu{relu_index}"] = relu
            modules[f"linear{i+2}"] = nn.Linear(linears[i], linears[i + 1], bias=True)

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
            r = (shape.numel() / z) ** 0.2
            gen = SimpleLinearNet(
                [z, int(z * r), int(z * r * r), int(z * r * r * r), shape.numel()]
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



def grade(x, target, val=False):
    loss = F.cross_entropy(x, target)
    correct = None
    if val:
        pred = x.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
    return (correct, loss)


def eval_clf(Z, data):
    """ calc classifier loss """
    data = data.cuda()
    x = F.relu(F.conv2d(data, Z[0]))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(F.conv2d(x, Z[1]))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(F.conv2d(x, Z[2]))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(x.size(0), -1)
    x = F.relu(F.linear(x, Z[3]))
    x = F.linear(x, Z[4])
    return x


def main():
    z = 256
    ze = 512
    batch_size = 32
    netT = SimpleConvNet().cuda()
    netH = HyperNet(netT, ze, z).cuda()
    netD = SimpleLinearNet([256, 1024, 1024, 1024, 1], final_sigmoid=True).cuda()

    print(netT, netH, netD)

    # optimH = optim.Adam(netH.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)

    optimE = optim.Adam(
        netH.encoder.parameters(), lr=5e-3, betas=(0.5, 0.9), weight_decay=1e-4
    )
    optimW1 = optim.Adam(
        netH.generator1.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4
    )
    optimW2 = optim.Adam(
        netH.generator2.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4
    )
    optimW3 = optim.Adam(
        netH.generator3.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4
    )
    optimW4 = optim.Adam(
        netH.generator4.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4
    )
    optimW5 = optim.Adam(
        netH.generator5.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4
    )

    optimD = optim.Adam(netD.parameters(), lr=5e-5, betas=(0.5, 0.9), weight_decay=1e-4)

    cifar_train, cifar_test = load_cifar()
    best_test_acc, best_test_loss = 0.0, np.inf

    x_dist = create_d(ze)
    z_dist = create_d(z)

    ops = 0
    start_time = time.time()
    for epoch in range(1000):
        for batch_idx, (data, target) in enumerate(cifar_train):
            data, target = data.cuda(), target.cuda()
            netH.zero_grad()
            netD.zero_grad()
            z = sample_d(x_dist, batch_size)
            q, w, netT = netH(z)

            # Z Adversary
            free_params([netD])
            freeze_params([netH])
            for code in q:
                noise = sample_d(z_dist, batch_size)
                d_real = netD(noise)
                d_fake = netD(code)
                d_real_loss = -1 * torch.log((1 - d_real).mean())
                d_fake_loss = -1 * torch.log(d_fake.mean())
                d_real_loss.backward(retain_graph=True)
                d_fake_loss.backward(retain_graph=True)
                d_loss = d_real_loss + d_fake_loss

            optimD.step()
            freeze_params([netD])
            free_params([netH])

            x = netT(data)
            # x = eval_clf(list(w.values()), data)
            correct, loss = grade(x, target, val=True)

            # Retain graph because the generators enter the encoder multiple times
            loss.backward()

            # optimH.step()
            optimE.step()
            optimW1.step()
            optimW2.step()
            optimW3.step()
            optimW4.step()
            optimW5.step()
            loss = loss.item()

            with torch.no_grad():
                """ Update Statistics """
                if batch_idx % 50 == 0:
                    acc = correct / 1
                    ops_per_sec = ops//(time.time() - start_time)
                    print("*"*70)
                    print("{}/{} Acc: {}, G Loss: {}, D Loss: {}".format(epoch,batch_idx, acc, loss, d_loss))
                    print("{} ops/s, best test loss: {}, best test acc: {}".format(ops_per_sec, best_test_loss, best_test_acc))
                    # print("**************************************")

                if batch_idx > 1 and batch_idx % 199 == 0:
                    test_acc = 0.0
                    test_loss = 0.0
                    total_correct = 0.0
                    for i, (data, y) in enumerate(cifar_test):
                        data, y = data.cuda(), y.cuda()
                        z = sample_d(x_dist, batch_size)
                        _, _, netT = netH(z)
                        x = netT(data)
                        correct, loss = grade(x, y, val=True)

                        test_acc += correct.item()
                        total_correct += correct.item()
                        test_loss += loss.item()

                    test_loss /= len(cifar_test.dataset)
                    test_acc /= len(cifar_test.dataset)

                    print(
                        "Test Accuracy: {}, Test Loss: {},  ({}/{})".format(
                            test_acc, test_loss, total_correct, len(cifar_test.dataset)
                        )
                    )

                    if test_loss < best_test_loss or test_acc > best_test_acc:
                        print("==> new best stats, saving")
                        if test_loss < best_test_loss:
                            best_test_loss = test_loss
                        if test_acc > best_test_acc:
                            best_test_acc = test_acc

            ops += batch_size


def main2():
    batch_size = 32
    netT = SimpleConvNet().cuda()
    print(netT)

    optimT = optim.Adam(netT.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)

    cifar_train, cifar_test = load_cifar()
    best_test_acc, best_test_loss = 0.0, np.inf

    ops = 0
    start_time = time.time()
    for epoch in range(1000):
        for batch_idx, (data, target) in enumerate(cifar_train):
            data, target = data.cuda(), target.cuda()
            netT.zero_grad()
            x = netT(data)
            correct, loss = grade(x, target, val=True)

            loss.backward()

            optimT.step()
            loss = loss.item()

            with torch.no_grad():
                """ Update Statistics """
                if batch_idx % 200 == 0:
                    acc = correct / 1
                    ops_per_sec = ops//(time.time() - start_time)
                    print("*"*70)
                    print("{}/{} Acc: {}, T Loss: {}".format(epoch,batch_idx, acc, loss))
                    print("{} ops/s, best test loss: {}, best test acc: {}".format(ops_per_sec, best_test_loss, best_test_acc))

            ops += batch_size

        with torch.no_grad():
            test_acc = 0.0
            test_loss = 0.0
            total_correct = 0.0
            for i, (data, y) in enumerate(cifar_test):
                data, y = data.cuda(), y.cuda()
                x = netT(data)
                correct, loss = grade(x, y, val=True)

                test_acc += correct.item()
                total_correct += correct.item()
                test_loss += loss.item()

            test_loss /= len(cifar_test.dataset)
            test_acc /= len(cifar_test.dataset)

            print(
                "Test Accuracy: {}, Test Loss: {},  ({}/{})".format(
                    test_acc, test_loss, total_correct, len(cifar_test.dataset)
                )
            )

            if test_loss < best_test_loss or test_acc > best_test_acc:
                print("==> new best stats, saving")
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                if test_acc > best_test_acc:
                    best_test_acc = test_acc


# if __name__ == '__main__':
#     main()
