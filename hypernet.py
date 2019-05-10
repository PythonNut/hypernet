import numpy as np
import torch
import time
import argparse
from pathlib import Path

from torch import nn
from torch import optim
from torch.nn import functional as F
import torch.distributions.normal as N
from tensorboardX import SummaryWriter

from util import *
from modules import *

def load_args():
    parser = argparse.ArgumentParser(description='param-hypernet')
    parser.add_argument('--zq', default=256, type=int, help='latent space width')
    parser.add_argument('--ze', default=512, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('-n', '--name', default="test", type=str)
    parser.add_argument('-o', '--outdir', default=".", type=str)
    parser.add_argument('-s', '--standard', action="store_true")
    parser.add_argument('--embeddings', action="store_true")
    parser.add_argument('--dry', action="store_true")

    args = parser.parse_args()
    return args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def train_gan(zq=256, ze=512, batch_size=32, outdir=".", name="tmp", dry=False, **kwargs):
    if not dry:
        tensorboard_path = Path(outdir) / 'tensorboard' / name
        model_path = Path(outdir) / 'models' / name
        tensorboard_path.mkdir(exist_ok=True, parents=True)
        model_path.mkdir(exist_ok=True, parents=True)

        sw = SummaryWriter(str(tensorboard_path))

    netT = SimpleConvNet(bias=False).to(device)
    netH = HyperNet(netT, ze, zq).to(device)
    netD = SimpleLinearNet(
        [zq * batch_size, zq * batch_size//2, zq * batch_size//4, 1024, 1],
        final_sigmoid=True,
        batchnorm=False
    ).to(device)

    print(netT, netH, netD)
    print(f"netT params: {param_count(netT)}")
    print(f"netH params: {param_count(netH)}")
    print(f"netD params: {param_count(netD)}")
    generator_count = param_layer_count(netT)

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
    minibatch_count = len(cifar_train)
    print(f"Minibatches: {minibatch_count}")

    best_test_acc, best_test_loss = MaxMeter(), MinMeter()
    g_loss_meter, d_loss_meter = AverageMeter(), AverageMeter()
    d_acc_meter = AverageMeter()
    dgrad_meter = AverageMeter()

    adversarial_loss = nn.BCELoss()
    real_label, fake_label = 0, 1
    label = torch.zeros((generator_count, 1), device=device)

    ops = 0
    start_time = time.time()
    for epoch in range(1000):
        d_loss_meter.reset()
        g_loss_meter.reset()
        d_acc_meter.reset()
        dgrad_meter.reset()
        for batch_idx, (data, target) in enumerate(cifar_train):
            n_iter = epoch * minibatch_count + batch_idx
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            netH.zero_grad()
            netD.zero_grad()
            z = fast_randn((batch_size, ze), device=device, requires_grad=True)
            q, w, netTs = netH(z)

            # Z Adversary
            free_params([netD])
            freeze_params([netH])

            codes = q.permute((1, 0, 2)).contiguous().view(generator_count, -1)
            noise = fast_randn((generator_count, zq * batch_size), device=device, requires_grad=True)
            d_real = netD(noise)
            d_fake = netD(codes)
            d_real_loss = adversarial_loss(d_real, label.fill_(real_label))
            d_real_loss.backward(retain_graph=True)
            d_fake_loss = adversarial_loss(d_fake, label.fill_(fake_label))
            d_fake_loss.backward(retain_graph=True)
            d_loss = d_real_loss + d_fake_loss
            dgrad_meter.update(model_grad_norm(netD))
            d_loss_meter.update(d_loss.item())
            d_acc_meter.update((sum(d_real < 0.5) + sum(d_fake > 0.5)).item()/(generator_count * 2))

            optimD.step()

            # Train the generator
            freeze_params([netD])
            free_params([netH])

            for netT in netTs:
                x = netT(data)
                correct, loss = clf_performance(x, target, val=True)
                g_loss_meter.update(loss.item())

                # Retain graph because the generators enter the encoder multiple times
                loss.backward(retain_graph=True)

            # fool the discriminator
            d_fake_loss = adversarial_loss(d_fake, label.fill_(real_label))
            d_fake_loss.backward()

            # optimH.step()
            optimE.step()
            optimW1.step()
            optimW2.step()
            optimW3.step()
            optimW4.step()
            optimW5.step()

            with torch.no_grad():
                """ Update Statistics """
                if batch_idx % 50 == 0:
                    acc = correct / 1
                    current_time = time.time()
                    ops_per_sec = ops//(current_time - start_time)
                    start_time = current_time
                    ops = 0
                    print("*"*70 + " " + name)
                    print("{}/{} Acc: {}, G Loss: {}, D Loss: {}".format(epoch,batch_idx, acc, loss.item(), d_loss.item()))
                    print("{} ops/s, best test loss: {}, best test acc: {}".format(ops_per_sec, best_test_loss.min, best_test_acc.max))

                ops += batch_size

                if batch_idx > 1 and batch_idx % 199 == 0:
                    test_acc = 0.0
                    test_loss = 0.0
                    total_correct = 0.0
                    for i, (data, y) in enumerate(cifar_test):
                        data = data.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        z = fast_randn((batch_size, ze), device=device, requires_grad=True)
                        _, _, netTs = netH(z)
                        x = netTs[0](data)
                        correct, loss = clf_performance(x, y, val=True)

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

                    if not dry:
                        sw.add_scalar('T/loss', test_loss, n_iter)
                        sw.add_scalar('T/acc', test_acc, n_iter)
                        sw.add_scalar('G/loss', g_loss_meter.avg, n_iter)
                        sw.add_scalar('D/loss', d_loss_meter.avg, n_iter)
                        sw.add_scalar('D/acc', d_acc_meter.avg, n_iter)
                        sw.add_scalar('D/gradnorm', dgrad_meter.avg, n_iter)
                        netH.eval()
                        netH_samples = [netH(fast_randn((batch_size, ze)).cuda()) for _ in range(10)]
                        netH.train()
                        sw.add_scalar('G/g_var', sum(x.std(0).mean() for v in netH_samples for x in v[1].values())/(generator_count * batch_size), n_iter)
                        sw.add_scalar('G/q_var', torch.cat([s[0].view(-1, zq) for s in netH_samples]).var(0).mean(), n_iter)

                        if kwargs['embeddings']:
                            sw.add_embedding(q.view(-1, zq), global_step=n_iter, tag="q", metadata=list(range(generator_count))*batch_size)

                    if best_test_loss.update(test_loss) | best_test_acc.update(test_acc):
                        print("==> new best stats, saving")
                        if not dry:
                            torch.save(
                                {
                                    'n_iter': n_iter,
                                    'epoch': epoch,
                                    'batch_idx': batch_idx,
                                    'netH': netH.state_dict(),
                                    'netD': netD.state_dict(),
                                    'optimH': optimH.state_dict(),
                                    'optimD': optimD.state_dict()
                                },
                                str(model_path / 'best.pt')
                            )



def train_standard(batch_size=32, outdir=".", name="tmp", **kwargs):
    tensorboard_path = Path(outdir) / 'tensorboard' / name
    model_path = Path(outdir) / 'models' / name
    tensorboard_path.mkdir(exist_ok=True, parents=True)
    model_path.mkdir(exist_ok=True, parents=True)

    sw = SummaryWriter(str(tensorboard_path))

    print(netT)
    print(f"netT layers: {param_layer_count(netT)}")
    print(f"netT params: {param_count(netT)}")

    optimT = optim.Adam(netT.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)

    cifar_train, cifar_test = load_cifar()
    minibatch_count = len(cifar_train)
    best_test_acc, best_test_loss = MaxMeter(), MinMeter()
    t_loss_meter = AverageMeter()

    ops = 0
    start_time = time.time()
    for epoch in range(1000):
        for batch_idx, (data, target) in enumerate(cifar_train):
            n_iter = epoch * minibatch_count + batch_idx
            data, target = data.to(device), target.to(device)
            netT.zero_grad()
            x = netT(data)
            correct, loss = clf_performance(x, target, val=True)
            t_loss_meter.update(loss.item())

            loss.backward()

            optimT.step()

            with torch.no_grad():
                """ Update Statistics """
                if batch_idx % 200 == 0:
                    acc = correct / 1
                    ops_per_sec = ops//(time.time() - start_time)
                    print("*"*70 + " " + name)
                    print("{}/{} Acc: {}, T Loss: {}".format(epoch,batch_idx, acc, loss.item()))
                    print("{} ops/s, best test loss: {}, best test acc: {}".format(ops_per_sec, best_test_loss.min, best_test_acc.max))

            ops += batch_size

        with torch.no_grad():
            test_acc = 0.0
            test_loss = 0.0
            total_correct = 0.0
            for i, (data, y) in enumerate(cifar_test):
                data, y = data.to(device), y.to(device)
                x = netT(data)
                correct, loss = clf_performance(x, y, val=True)

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

            sw.add_scalar('T/loss', test_loss, n_iter)
            sw.add_scalar('T/acc', test_acc, n_iter)
            sw.add_scalar('T/lr', optimT.param_groups[0]['lr'], n_iter)

            if best_test_loss.update(test_loss) | best_test_acc.update(test_acc):
                print("==> new best stats, saving")


def main():
    args = vars(load_args())
    if args['standard']:
        train_standard(**args)
    else:
        train_gan(**args)

if __name__ == '__main__':
    main()
