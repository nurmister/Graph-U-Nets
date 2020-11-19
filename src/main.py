"""Validate and test a trained graph U-net with the supplied arguments."""

import argparse
import random
import time
import torch
import numpy as np
from network import GNet
from trainer import Trainer
from utils.data_loader import FileLoader


def get_args():
    """
    Return a dictionary of arguments regarding data loading and model training.

    Returns
    -------
    args : dict
        Dictionary of arguments regarding data loading and model training.

    """
    parser = argparse.ArgumentParser(description='Arguments for graph prediction.')
    # Seed to be used to ensure reproducibility between trainings with the same
    # settings.
    parser.add_argument('-seed', type=int, default=1, help='Seed')
    # The data to load.
    parser.add_argument('-data', default='DD', help='Data folder name')
    parser.add_argument('-fold', type=int, default=1, help='Fold (1-10)')
    # The training schedule.
    parser.add_argument('-num_epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('-batch', type=int, default=8, help='Batch size')
    # Model hyperparameters.
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate')
    # TODO: see what this does.
    parser.add_argument('-deg_as_tag', type=int, default=0, help='1 or degree')
    parser.add_argument('-l_num', type=int, default=3, help='Number of (which?) layers')  # TODO: determine what kind of layers this sets.
    parser.add_argument('-h_dim', type=int, default=512, help='hidden dim')  # XXX: not sure what exactly.
    parser.add_argument('-l_dim', type=int, default=48, help='layer dim')  # XXX: not sure what exactly.
    parser.add_argument('-drop_n', type=float, default=0.3, help='drop net')  # XXX: not sure what exactly.
    parser.add_argument('-drop_c', type=float, default=0.2, help='drop output')  # XXX: not sure what exactly.
    parser.add_argument('-act_n', type=str, default='ELU', help='Activation function to be applied to hidden units')
    parser.add_argument('-act_c', type=str, default='ELU', help='Activation function to be applied to output unit')  # TODO: determine if this is the best choice for classif.
    # TODO: see what these two arguments control.
    parser.add_argument('-ks', nargs='+', type=float, default='0.9 0.8 0.7')
    parser.add_argument('-acc_file', type=str, default='re', help='acc file')
    args, _ = parser.parse_known_args()
    return args


def set_random(seed):
    """
    Set the seed before training.

    Parameters
    ----------
    seed : int
        Seed to be used to set for Python, NumPy, and PyTorch.

    Returns
    -------
    None.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def app_run(args, G_data, fold_idx):
    """
    Validate the model over one fold.

    Parameters
    ----------
    args : dict
        Dictionary of arguments to configure model training.
    G_data : FileLoader
        Returns the training and validation fold data, and associated details.
    fold_idx : int
        Validation fold index, from 1-10.

    Returns
    -------
    None.

    """
    G_data.use_fold_data(fold_idx)
    net = GNet(G_data.feat_dim, G_data.num_class, args)
    trainer = Trainer(args, net, G_data)
    trainer.train()


def main():
    """
    Train a g-U-net model with the supplied settings.    

    Returns
    -------
    None.

    """
    args = get_args()
    print(args)
    set_random(args.seed)
    start = time.time()
    G_data = FileLoader(args).load_data()

    print(f'Data loading completed in {time.time() - start}s.')
    # If no particular fold is supplied, train the model on each fold.
    if args.fold == 0:
        for fold_idx in range(10):
            print(f'Fold {fold_idx + 1}')
            app_run(args, G_data, fold_idx)
    else:
        print(f'Fold {args.fold}')
        app_run(args, G_data, args.fold - 1)
    print(f'Completed training in {time.time() - start}s.')


if __name__ == "__main__":
    main()
