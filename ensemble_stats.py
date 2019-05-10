import argparse
from statistics import mean

from util import *
from modules import *
from resnet import *


def load_args():
    parser = argparse.ArgumentParser(description='param-hypernet')
    parser.add_argument('-p', '--pt', default="test", type=str)
    parser.add_argument('-n', '--networks', default="32", type=int)
    parser.add_argument('--cuda', action="store_true")
    args = parser.parse_args()
    return args


def main(pt, networks, cuda):
    print("Constructing models...")
    netT = resnet20()
    netH = HyperNet(netT, 512, 256)

    print("Loading save file...")
    D = torch.load(pt, map_location=lambda storage, location: storage)
    netH.load_state_dict(D['netH'])

    print("Loading CIFAR10...")
    cifar_train, cifar_test = load_cifar()

    nets = sample_target_net(netH, networks)

    if cuda:
        nets = [net.cuda() for net in nets]


    device = next(nets[0].parameters()).device

    with torch.no_grad():
        print(f"Evaluating individual networks on {device}...")
        individual_scores = [eval_clf(net, cifar_test, device=device)[1] for net in nets]
        print(f"Evaluating ensemble on {device}...")
        ensemble_score = eval_clf(make_ensemble(nets), cifar_test, device=device)[1]


    print(f"Ensemble acc: {ensemble_score}")
    print(f"Individual acc: min {min(individual_scores)}, mean {mean(individual_scores)}, max {max(individual_scores)}")



if __name__ == '__main__':
    main(**vars(load_args()))
