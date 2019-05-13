import argparse
from statistics import mean
from progressbar import progressbar

from util import *
from modules import *
from resnet import *


def load_args():
    parser = argparse.ArgumentParser(description='param-hypernet')
    parser.add_argument('-p', '--pt', default="test", type=str)
    parser.add_argument('-n', '--networks', default="32", type=int)
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--zq', default=256, type=int, help='latent space width')
    parser.add_argument('--ze', default=512, type=int, help='encoder dimension')
    args = parser.parse_args()
    return args


def main(pt, networks, cuda, eval, zq, ze):

    print("Constructing models...")
    netT = resnet20()
    netH = HyperNet(netT, ze, zq)

    print("Loading save file...")
    D = torch.load(pt, map_location=lambda storage, location: storage)
    netH.load_state_dict(D['netH'])

    print("Loading CIFAR10...")
    cifar_train, cifar_test = load_cifar()

    nets = sample_target_net(netH, networks)

    if cuda:
        nets = [net.cuda() for net in nets]

    if eval:
        for net in nets:
            net.eval()

    device = next(nets[0].parameters()).device

    with torch.no_grad():
        print(f"Evaluating individual networks on {device}...")
        individual_scores = eval_clfs(nets, progressbar(cifar_test), device=device)[1]
        print(f"Evaluating ensemble on {device}...")
        ensemble_score = eval_clf(make_ensemble(nets), progressbar(cifar_test), device=device)[1]

    mean_individual_score = mean(individual_scores)

    print(f"Ensemble acc: {ensemble_score}")
    print(f"Individual acc: min {min(individual_scores)}, mean {mean_individual_score}, max {max(individual_scores)}")

    error_red = ensemble_score - mean_individual_score
    frac_error_red = error_red/(1 - mean_individual_score)
    print(f"Error reduction: {error_red}, fractional error reduction {frac_error_red}")
    return individual_scores, ensemble_score, error_red, frac_error_red


if __name__ == '__main__':
    main(**vars(load_args()))
