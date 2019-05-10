import numpy as np
import torch
import time
import argparse
from pathlib import Path

from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.distributions.normal as N
from tensorboardX import SummaryWriter

from util import *
from modules import *
from model import *
from resnet import *
from network import *
from adamw import AdamW
from cyclic_scheduler import CyclicLRWithRestarts

def load_args():
    parser = argparse.ArgumentParser(description='param-hypernet')
    parser.add_argument('--zq', default=256, type=int, help='latent space width')
    parser.add_argument('--ze', default=512, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('-n', '--name', default="test", type=str)
    parser.add_argument('-o', '--outdir', default=".", type=str)
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

    netT = resnet20().to(device)
    # netT = SimpleConvNet(bias=False).to(device)
    netH = HyperNet(netT, ze, zq).to(device)

    print("Loading pretrained generators...")
    pretrain = torch.load('pretrained.pt')
    netH.load_state_dict(pretrain['netH'])
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

    optimH = AdamW(netH.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimD = AdamW(netD.parameters(), lr=5e-5, betas=(0.5, 0.9), weight_decay=1e-4)

    g_loss_meter, d_loss_meter = AverageMeter(), AverageMeter()
    d_acc_meter = AverageMeter()
    gp_meter = AverageMeter()
    dgrad_meter = AverageMeter()

    adversarial_loss = nn.BCELoss()
    real_label, fake_label = 0, 1
    label = torch.zeros((generator_count, 1), device=device)

    ops = 0
    start_time = time.time()
    minibatch_count = 1562
    for epoch in range(100000):
        d_loss_meter.reset()
        g_loss_meter.reset()
        d_acc_meter.reset()
        gp_meter.reset()
        dgrad_meter.reset()
        # schedH.step()
        # schedD.step()
        for batch_idx in range(minibatch_count):
            n_iter = epoch * minibatch_count + batch_idx

            netH.zero_grad()
            netD.zero_grad()
            z = fast_randn((batch_size, ze), device=device, requires_grad=True)
            q = netH.encoder(z).view(-1, generator_count, zq)

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
            # gp = calc_gradient_penalty(netD, noise, codes, device=device)
            # d_loss = d_fake.mean() - d_real.mean() + 10 * gp
            # d_loss.backward(retain_graph=True)
            dgrad_meter.update(model_grad_norm(netD))
            d_loss_meter.update(d_loss.item())
            d_acc_meter.update((sum(d_real < 0.5) + sum(d_fake > 0.5)).item()/(generator_count * 2))
            # gp_meter.update(gp.item())

            optimD.step()
            # schedD.batch_step()
            # Train the generator
            freeze_params([netD])
            free_params([netH])

            # fool the discriminator
            # d_fake_loss = -d_fake.mean()
            # d_fake_loss.backward()

            d_fake_loss = adversarial_loss(d_fake, label.fill_(real_label))
            d_fake_loss.backward(retain_graph=True)

            optimH.step()

            with torch.no_grad():
                """ Update Statistics """
                if batch_idx % 50 == 0:
                    current_time = time.time()
                    ops_per_sec = ops//(current_time - start_time)
                    start_time = current_time
                    ops = 0
                    print("*"*70 + " " + name)
                    print("{}/{} D Loss: {}".format(epoch,batch_idx, d_loss.item()))
                    print("{} ops/s".format(ops_per_sec))

                ops += batch_size

                if batch_idx > 1 and batch_idx % 199 == 0:
                    if not dry:
                        sw.add_scalar('G/loss', g_loss_meter.avg, n_iter)
                        sw.add_scalar('D/loss', d_loss_meter.avg, n_iter)
                        sw.add_scalar('D/acc', d_acc_meter.avg, n_iter)
                        sw.add_scalar('D/gp', gp_meter.avg, n_iter)
                        sw.add_scalar('D/gradnorm', dgrad_meter.avg, n_iter)
                        netH.eval()
                        netH_samples = [netH(fast_randn((batch_size, ze)).cuda()) for _ in range(10)]
                        netH.train()
                        sw.add_scalar('G/g_var', sum(x.std(0).mean() for v in netH_samples for x in v[1].values())/(generator_count * 10), n_iter)
                        sw.add_scalar('G/q_var', torch.cat([s[0].view(-1, zq) for s in netH_samples]).var(0).mean(), n_iter)

                        if kwargs['embeddings']:
                            sw.add_embedding(q.view(-1, zq), global_step=n_iter, tag="q", metadata=list(range(generator_count))*batch_size)

                        torch.save(
                            {
                                'netH': netH.state_dict(),
                                'netD': netD.state_dict()
                            },
                            str(model_path / 'pretrain.pt')
                        )




def main():
    args = vars(load_args())
    train_gan(**args)

if __name__ == '__main__':
    main()
