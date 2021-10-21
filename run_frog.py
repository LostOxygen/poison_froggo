"""
main file to run the Poison Frog! attack based on:
https://arxiv.org/abs/1804.00792 and
https://github.com/ashafahi/inceptionv3-transferLearn-poison/tree/master
"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import time
import socket
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from poison_frog.dataset import create_perturbed_dataset, create_sanitized_dataset
from poison_frog.train import train
from poison_frog.test import test_model

torch.backends.cudnn.benchmark = True
ATTACK_ITERS = 1500
ATTACK_LR = 0.01 # unused at the moment (gets overwritten inside the attack loop)
EPOCHS = 100
LR = 0.01
BATCH_SIZE = 32

def get_model(device: str) -> nn.Sequential:
    """
    helper function to create and initialize the model.
    :param device: device string

    :return: returns the loaded model
    """
    model = models.inception_v3(pretrained=True)
    model.aux_logits = False
    model.eval()
    model = model.to(device)
    return model


def main(gpu: int, num_poisons: int, target_class: int, new_class: int, transfer: bool,
         num_runs: int) -> None:
    """
    :gpu: specifies which GPU should be used -> [0, 1]
    :num_poisons: number of poisons
    :target_class: the class of which on image should be misclassified
    :new_class: the class as which the target image should be misclassified (poison class)
    :transfer: flag if transfer learning should be used or not
    :num_runs: number of runs for the complete attack (used to measure the success rate)

    :return: None
    """
    start = time.perf_counter()
    if gpu == 0:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if gpu == 1:
        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # print a summary of the chosen arguments
    print("\n\n\n"+"#"*50)
    print("# " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print("# System: {} CPU cores with {} GPUs on {}".format(torch.get_num_threads(),
                                                             torch.cuda.device_count(),
                                                             socket.gethostname()
                                                             ))
    if device == 'cpu':
        print("# Using: CPU with ID {}".format(device))
    else:
        print("# Using: {} with ID {}".format(torch.cuda.get_device_name(device=device), device))
    print("# Number Poisons: {}".format(num_poisons))
    print("# Number Attack-Runs: {}".format(num_runs))
    print("# Target_Class: {} ".format(target_class))
    print("# Poison_Class: {} ".format(new_class))
    print("# Dataset: ImageNet")
    print("# Transfer-Learning: {}".format(transfer))
    print("#"*50+"\n\n\n")

    # load the dog and frog images from their path into numpy arrays. Once saved
    # the following line is obsolete and can be disabled
    create_sanitized_dataset(target_class, new_class)

    # load the base_model and copy it onto the specified device
    model = get_model(device)

    num_successful_attacks = 0
    num_total_attacks = 0
    for i_run in range(num_runs):
        num_total_attacks += 1
        # create a pertubed dataset with the chosen target / poison classes
        t_id = create_perturbed_dataset(target_class, new_class, ATTACK_ITERS, ATTACK_LR,
                                        model, num_poisons, device)

        # train the model on the poisoned dataset
        train(epochs=EPOCHS,
              learning_rate=LR,
              batch_size=BATCH_SIZE,
              device_name=device,
              use_transfer=transfer,
              poisoned=True,
              data_augmentation=True)

        # test the model for accuracy and
        if test_model(device, t_id, new_class):
            num_successful_attacks += 1

        print("[ {}/{} attacks successful ]".format(num_successful_attacks, num_total_attacks))

    print("[ Finished with {}/{} succcessful attacks ]".format(num_successful_attacks,
                                                               num_total_attacks))
    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"Computation time: {duration:0.4f} hours")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", "-g", help="GPU", type=int, default=0)
    parser.add_argument("--num_poisons", "-p", help="Number of Poisons", type=int, default=1)
    parser.add_argument("--target_class", "-t", help="Target Class", type=int,
                        default=0, required=False)
    parser.add_argument("--new_class", "-n", help="New Class", type=int,
                        default=1, required=False)
    parser.add_argument("--num_runs", "-r", help="number of runs", type=int,
                        default=1, required=False)
    parser.add_argument("--transfer", "-f", help="use transfer learning to train only the fc layer",
                        action='store_true', default=True, required=False)

    args = parser.parse_args()
    main(**vars(args))
