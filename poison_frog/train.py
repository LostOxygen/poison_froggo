"""library module with all training functions"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
import pkbar
from .dataset import TensorDataset

def save_model(net: nn.Sequential, poisoned: bool) -> None:
    """
    helper function which saves the given net in the specified path.
    if the path does not exists, it will be created.
    :param net: object of the model
    :param poisoned: flag to change the output name of the poisoned model

    :return: None
    """
    print("[ Saving Model ]")
    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('./poison_frog/model_saves'):
        os.mkdir('./poison_frog/model_saves')
    if poisoned:
        torch.save(state, "./poison_frog/model_saves/poisoned_model")
    else:
        torch.save(state, "./poison_frog/model_saves/base_model")


def adjust_learning_rate(optimizer, epoch: int, epochs: int, learning_rate: int) -> None:
    """
    helper function to adjust the learning rate
    according to the current epoch to prevent overfitting.
    :paramo ptimizer: object of the used optimizer
    :param epoch: the current epoch
    :param epochs: the total epochs of the training
    :param learning_rate: the specified learning rate for the training

    :return: None
    """
    new_lr = learning_rate
    if epoch >= np.floor(epochs*0.5):
        new_lr /= 10
    if epoch >= np.floor(epochs*0.75):
        new_lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def get_loaders(poisoned: bool, data_augmentation: bool, batch_size: int = 32) -> DataLoader:
    """
    helper function to create dataset loaders for CIFAR10 datasets
    :param data_augmentation: flag specifies if data data_augmentation should be enabled
    :param batch_size: batch size which should be used for the dataloader

    :return: dataloader with the specified dataset
    """
    data_path = './poison_frog/datasets/'

    if poisoned:
        normalize = transforms.ToTensor()
        # use the poisoned dataset and the normal test set
        train_data = torch.load(os.path.join(data_path, "attack_images"))
        train_labels = torch.load(os.path.join(data_path, "attack_labels"))
        test_data = torch.load(os.path.join(data_path, "sanitized_images_test"))
        test_labels = torch.load(os.path.join(data_path, "sanitized_labels_test"))

    else:
        # use the normal train and testset
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_data = torch.load(os.path.join(data_path, "sanitized_images"))
        train_labels = torch.load(os.path.join(data_path, "sanitized_labels"))
        test_data = torch.load(os.path.join(data_path, "sanitized_images_test"))
        test_labels = torch.load(os.path.join(data_path, "sanitized_labels_test"))

    train_dataset = TensorDataset(train_data, train_labels, transform=transforms.Compose([
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                #transforms.ToTensor(),
                normalize]))

    test_dataset = TensorDataset(test_data, test_labels, transform=transforms.Compose([
                transforms.Resize(299),
                #transforms.ToTensor(),
                normalize]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                             pin_memory=True)

    return train_loader, test_loader


def get_model(use_transfer: bool) -> nn.Sequential:
    """
    helper function to create and initialize the model.
    :use_transfer: enables the pretrained model
    :return: returns the loaded model
    """
    if use_transfer:
        net = models.inception_v3(pretrained=True, progress=True)
        net.aux_logits = False
    else:
        net = models.inception_v3()
        net.aux_logits = False

    net = net.to(device)
    return net


def freeze_layers(model: nn.Sequential) -> nn.Sequential:
    """helper function to freeze feature extraction layers for transfer learning
    :param model: the network model
    :return: model with frozen layers
    """
    freeze_list = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "conv_layer"]

    for name, param in model.named_parameters():
        for layer in freeze_list:
            if layer in name:
                param.requires_grad = False

    return model


def init_classifier(layer) -> None:
    """
    helper function to apply re-initialization on the classifier weights
    :param layer: model layer on which the initialization should be applied

    :return: None
    """
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight.data)


def init_transfer_learning(model: nn.Sequential, poisoned):
    """
    initalizes the model and it's parameters for transfer learning.
    :param model: the model on which transfer learning should be initialized

    :return: model with preloaded weights and a freshly initialized classifier and it's parameters
    """
    model = model.to(device)
    model.apply(init_classifier)
    #if poisoned: # freeze every layer except for the fully connected one
    model = freeze_layers(model)
    parameters = model.parameters()

    return model, parameters


def train(epochs: int, learning_rate: int, poisoned: bool, batch_size: int, device_name: str,
          use_transfer: bool, data_augmentation: bool = True) -> None:
    """
    Main method to train the model with the specified parameters. Saves the model in every
    epoch specified in SAVE_EPOCHS. Prints the model status during the training.
    :param epochs: specifies how many epochs the training should last
    :param learning_rate: specifies the learning rate of the training
    :param batch_size: specifies the batch size of the training data (default = 128)
    :param device_name: sets the device on which the training should be performed
    :param use_transfer: flag if transfer learning should be used
    :param data_augmentation: flag specifies if data augmentation should be used
    :param poisoned: loads the poisoned datast instead of the normal one

    :return: None
    """
    print("[ Initialize Training ]")
    # create a global variable for the used device
    global device
    device = device_name

    # initializes the model, loss function, optimizer and dataset
    train_loader, test_loader = get_loaders(poisoned, data_augmentation, batch_size)
    net = get_model(use_transfer)
    # if transfer learning flag is set, freeze certain layers and only pass the non frozen
    # layers to the optimizer
    if use_transfer:
        net, net_parameters = init_transfer_learning(model=net, poisoned=poisoned)
        # epochs = np.ceil(epochs*0.25).astype(int)
        learning_rate = learning_rate*0.1
    else:
        net_parameters = net.parameters()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net_parameters, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    for epoch in range(0, epochs):
        # every epoch a new progressbar is created
        # also, depending on the epoch the learning rate gets adjusted before
        # the network is set into training mode
        kbar = pkbar.Kbar(target=len(train_loader)-1, epoch=epoch, num_epochs=epochs,
                          width=20, always_stateful=True)
        adjust_learning_rate(optimizer, epoch, epochs, learning_rate)
        net.train()
        correct = 0
        total = 0
        running_loss = 0.0

        # iterates over a batch of training data
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            #targets = torch.nn.functional.one_hot(targets)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            _, predicted = outputs.max(1)

            # calculate the current running loss as well as the total accuracy
            # and update the progressbar accordingly
            running_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            kbar.update(batch_idx, values=[("loss", running_loss/(batch_idx+1)),
                                           ("acc", 100. * correct / total)])
        # calculate the test accuracy of the network at the end of each epoch
        with torch.no_grad():
            net.eval()
            t_total = 0
            t_correct = 0
            for _, (inputs_t, targets_t) in enumerate(test_loader):
                inputs_t, targets_t = inputs_t.to(device), targets_t.to(device)
                #targets = torch.nn.functional.one_hot(targets)
                outputs_t = net(inputs_t)
                _, predicted_t = outputs_t.max(1)
                t_total += targets_t.size(0)
                t_correct += predicted_t.eq(targets_t).sum().item()
            print("-> test acc: {}".format(100.*t_correct/t_total))
    # save the model at the end of the training
    save_model(net, poisoned)

    # calculate the test accuracy of the network at the end of the training
    with torch.no_grad():
        net.eval()
        t_total = 0
        t_correct = 0
        for _, (inputs_t, targets_t) in enumerate(test_loader):
            inputs_t, targets_t = inputs_t.to(device), targets_t.to(device)
            outputs_t = net(inputs_t)
            _, predicted_t = outputs_t.max(1)
            t_total += targets_t.size(0)
            t_correct += predicted_t.eq(targets_t).sum().item()

    print("Final accuracy: Train: {} | Test: {}".format(100.*correct/total, 100.*t_correct/t_total))
