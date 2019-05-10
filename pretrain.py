from torch import nn
from modules import *
from util import *
from adamw import AdamW
import itertools as it
from resnet import *
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generate_noise(module, param, batch_size):
    results = []
    for i in range(batch_size):
        module.reset_parameters()
        results.append(deepcopy(param).view(-1))
    return torch.stack(results).to(device)

def codes_with_dropout(q, maxel=1024):
    batch_size, outel = q.shape
    if outel > maxel:
        I = torch.randint(high=outel, size=(batch_size, maxel), dtype=torch.int64)
        q = torch.stack([q[i] for i in enumerate(I)])
    q = q.view(-1)
    codes = torch.unsqueeze(q, dim=0)
    return codes

def make_netD(width, batch_size):
    netD = SimpleLinearNet(
        [width * batch_size, width*batch_size//2, width*batch_size//4, width*batch_size//8, 1],
        # final_sigmoid=True,
        batchnorm=False
    ).to(device)
    return netD

def pretrain_generator(netG, module, param, batch_size):
    outel = list(netG._modules.values())[-1].weight.shape[0]
    dwidth = min(1024, outel)
    netD = make_netD(dwidth, batch_size)

    print(f"Layer size: {outel}, G params: {param_count(netG)}, D params: {param_count(netD)}")
    optimG = AdamW(netG.parameters(), lr=5e-4, weight_decay=1e-4)
    optimD = AdamW(netD.parameters(), lr=5e-5, weight_decay=1e-4)

    i=0
    d_adv_meter = AverageMeter()

    while True:
        netG.zero_grad()
        netD.zero_grad()

        z = fast_randn((batch_size, 256), requires_grad=True, device=device)
        q =  netG(z)

        free_params([netD])
        freeze_params([netG])

        noise = codes_with_dropout(generate_noise(module, param, batch_size), dwidth)
        codes = codes_with_dropout(q, dwidth)
        d_real = netD(noise)
        d_fake = netD(codes)

        interp = random_interpolate(noise, codes, device=device)
        gp = calc_gradient_penalty(netD, interp, device=device)
        d_adv = d_fake.mean() - d_real.mean()
        d_loss = d_adv + 10 * gp
        d_adv_meter.update(d_adv.item())
        d_loss.backward(retain_graph=True)

        optimD.step()
        freeze_params([netD])
        free_params([netG])

        d_fake_loss = -d_fake.mean()
        d_fake_loss.backward()

        optimG.step()
        if i %50 == 0:
            print(d_adv_meter.avg, gp.item())
            if i > 2000 and d_adv_meter.avg > 0:
                break

            d_adv_meter.reset()
        i+= 1


def main():
    batch_size = 16
    netT = resnet20()
    netH = HyperNet(netT, 512, 256)

    for i in range(len(netH.generators)):
        netG = netH.generators[i].to(device)
        name, param = list(netT.named_parameters())[i]
        module = dict(netT.named_modules())['.'.join(name.split(".")[:-1])]

        print(f"Pretraining {name}")

        pretrain_generator(netG, module, param, batch_size)
        netH.generators[i].load_state_dict(netG.state_dict())
        del netG
        torch.cuda.empty_cache()

    torch.save(
        {
            'netH': netH.state_dict(),
        },
        'pretrained.pt'
    )

if __name__ == '__main__':
    main()
