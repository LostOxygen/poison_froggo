"""library module to create the training and test data based on given images, as well as
   generating the poison to perturb the training data.
"""
import os
import copy
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
# from sklearn.model_selection import train_test_split
from imageio import imread


class TensorDataset(Dataset):
    """custom TensorDataset class which inherits from Pytorchs Dataset class
       and applies a specified transform to all dataset items.
    """
    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        image, targ = tuple(tensor[index] for tensor in self.tensors)
        if self.transform:
            real_transform = transforms.Compose([
                transforms.ToPILImage(),
                self.transform
            ])
            image = real_transform(image.cpu())
        return image, targ

    def __len__(self):
        return self.tensors[0].size(0)


def get_data() -> DataLoader:
    """
    helper function to initialize and provide a trainset loader as well as the
    normal train and test sets to perform copy operations on them

    :return: dataloaders for train/test sets for copy, preallocation and attacking
    """
    data_path = './poison_frog/datasets/'
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
                transforms.ToTensor(),
                normalize]))

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset),
                              shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True,
                             num_workers=1, pin_memory=True)
    test_loader2 = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    return train_loader, test_loader, test_loader2


def forward_step(model: nn.Sequential, img: torch.Tensor, target_image: torch.Tensor,
                 lr: float, target_logits) -> torch.Tensor:
    """helper function performing the forward step"""
    img.detach_() # disconnect image from the current autograd graph
    img.requires_grad = True

    logits = model(img)
    loss = torch.norm(logits - target_logits)
    model.zero_grad()
    loss.backward()

    img_grad = img.grad.data
    perturbed_img = img - lr*img_grad
    return perturbed_img


def backward_step(img: torch.Tensor, base_img: torch.Tensor, lr: float, beta: float) -> torch.Tensor:
    """helper function to perform the backward step"""
    perturbed_img = (img + lr*beta*base_img) / (1 + beta*lr)
    perturbed_img = torch.clamp(perturbed_img, 0, 1) # to avoid clipping

    return perturbed_img

def adam_one_step(model, m, v, t, currentImage, featRepTarget, learning_rate,
                  beta_1=0.9, beta_2=0.999, eps=1e-8) -> torch.Tensor:
    """one step adam optimization"""
    t += 1
    currentImage = currentImage.detach() # disconnect image from the current autograd graph
    currentImage.requires_grad_()

    with torch.enable_grad():
        logits = model(currentImage)
        target_logits = model(featRepTarget)
        loss = torch.norm(logits - target_logits)

    grad_t = torch.autograd.grad(loss, [currentImage])[0]
    m = beta_1 * m + (1-beta_1)*grad_t
    v = beta_2 * v + (1-beta_2)*grad_t**2
    m_hat = m/(1-beta_1**t)
    v_hat = v/(1-beta_2**t)
    with torch.no_grad():
        currentImage -= learning_rate*m_hat/(torch.sqrt(v_hat)+eps)
    return currentImage, m, v, t


def create_sanitized_dataset(target_class: int, new_class: int) -> None:
    """helper function to filter every non target or poison class out of the training data"""
    print("[ Sanitizing Dataset ]")
    datadir = './poison_frog/datasets/'
    train_path = datadir+'train/'
    test_path = datadir+'test/'

    train_data = list()
    test_data = list()
    dog_len = 0
    fish_len = 0
    dog_len_test = 0
    fish_len_test = 0

    print("[ Load doggos ]")
    for file in tqdm(os.listdir(train_path+"dog/")):
        image = imread(train_path+"dog/"+file, pilmode="RGB", as_gray=False)
        # check if image dimensions are corrent for the network
        if image.shape[-1] == 3 and image.ndim == 3:
            # move the channel dimension, as pytorch requires the channel dim to be the first dim
            # and resize the image to fit the input of 299x299 for inception_v3
            image = Image.fromarray(image).resize((299, 299))
            image = np.moveaxis(np.array(image), -1, -3)
            train_data.append(image)
            dog_len += 1

    for file in tqdm(os.listdir(test_path+"dog/")):
        image = imread(test_path+"dog/"+file, pilmode="RGB", as_gray=False)
        # check if image dimensions are corrent for the network
        if image.shape[-1] == 3 and image.ndim == 3:
            # move the channel dimension, as pytorch requires the channel dim to be the first dim
            # and resize the image to fit the input of 299x299 for inception_v3
            image = Image.fromarray(image).resize((299, 299))
            image = np.moveaxis(np.array(image), -1, -3)
            test_data.append(image)
            dog_len_test += 1

    print("[ Load fishes ]")
    for file in tqdm(os.listdir(train_path+"fish/")):
        image = imread(train_path+"fish/"+file, pilmode="RGB", as_gray=False)
        # check if image dimensions are corrent for the network
        if image.shape[-1] == 3 and image.ndim == 3:
            # move the channel dimension, as pytorch requires the channel dim to be the first dim
            # and resize the image to fit the input of 299x299 for inception_v3
            image = Image.fromarray(image).resize((299, 299))
            image = np.moveaxis(np.array(image), -1, -3)
            train_data.append(image)
            fish_len += 1

    for file in tqdm(os.listdir(test_path+"fish/")):
        image = imread(test_path+"fish/"+file, pilmode="RGB", as_gray=False)
        # check if image dimensions are corrent for the network
        if image.shape[-1] == 3 and image.ndim == 3:
            # move the channel dimension, as pytorch requires the channel dim to be the first dim
            # and resize the image to fit the input of 299x299 for inception_v3
            image = Image.fromarray(image).resize((299, 299))
            image = np.moveaxis(np.array(image), -1, -3)
            test_data.append(image)
            fish_len_test += 1


    train_labels = np.concatenate((np.zeros(dog_len), np.ones(fish_len)))
    test_labels = np.concatenate((np.zeros(dog_len_test), np.ones(fish_len_test)))
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    # print("[ Generating Train-Test split ]")
    # train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels,
    #                                                                     test_size=.25, shuffle=True)

    train_data, test_data = torch.from_numpy(train_data), torch.from_numpy(test_data)
    train_labels, test_labels = torch.LongTensor(train_labels), torch.LongTensor(test_labels)

    torch.save(train_data, './poison_frog/datasets/sanitized_images')
    torch.save(train_labels, './poison_frog/datasets/sanitized_labels')
    torch.save(test_data, './poison_frog/datasets/sanitized_images_test')
    torch.save(test_labels, './poison_frog/datasets/sanitized_labels_test')


def create_perturbed_dataset(target_class: int, new_class: int, attack_iters: int, lr: float,
                             model: nn.Sequential, num_poisons: int, device_name: str) -> int:
    """
    main function to create a pertubed dataset based on the chosen parameters.
    The algorithms tries to move the new_class (class as which the image should
    get misclassified as) into the direction of the target image in the feature
    space by creating a perturbation which minimizes the difference between the
    activations of the new_class and the target image.

    :param target_class: the target class for the image which should get missclassified
    :param new_class: the class as which the chosen image should get missclassified
    :param attack_iters: specifies the iterations for the attack
    :param num_poisons: number of poisons
    :param device_name: specifies the device on which the dataset will be computed
    :param model: the model which is used to create the advs examples
    :param lr: learning rate for the gradient descent inside the attack

    :return: the id of the randomly chosen target image
    """
    print("[ Initialize.. ]")
    # create a global variable and assign the device which should be used
    global device
    device = device_name

    # train_loader is used to copy the complete dataset while the raw train and test
    # dataset is used to create pertubed versions of the images
    _, _, test_loader = get_data()
    model.fc = nn.Sequential(
        nn.Identity() # remove the fully connected layer to obtain the feature space repr.
    )

    print("[ Preallocate Dataset.. ]")
    # create a copy of the dataset so it is possible to just pertube a certain
    # number of images
    data_path = "./poison_frog/datasets/"
    train_images = torch.load(os.path.join(data_path, "sanitized_images"))
    train_labels = torch.load(os.path.join(data_path, "sanitized_labels"))

    poisoned_images = torch.zeros((num_poisons, 3, 299, 299))
    poisoned_labels = torch.zeros((num_poisons))

    print("[ Building new Dataset.. ]")

    target_images_list = list()
    target_image = None
    target_image_id = 0
    for idx, (input, target) in enumerate(test_loader):
        if target == target_class:
            target_images_list.append([input.to(device), idx])

    random_id = np.random.randint(0, len(target_images_list))
    target_image = target_images_list[random_id][0]
    target_image_id = target_images_list[random_id][1]
    print("target_image ID: ", target_image_id)
    print("target_image class: ", target_class)

    # calculate the beta
    img_shape = np.squeeze(target_image).shape
    beta = 0.25 * (2048 / float(img_shape[0] * img_shape[1] * img_shape[2]))**2
    print("beta = {}".format(beta))

    # iterate over the whole test dataset and create a perturbed version of one (or N)
    # new_class (the class as which the chosen image should be misclassified as) image.
    adam = False
    current_pertube_count = 0
    for idx, (input, target) in enumerate(test_loader):
        difference = 100 # difference between base and target in feature space, will be updated
        if target == new_class and current_pertube_count < num_poisons:
            base_image, target = input.to(device), target.to(device)
            old_image = base_image

            # Initializations
            num_m = 40
            last_m_objs = []
            decay_coef = 0.5 #decay coeffiencet of learning rate
            stopping_tol = 1e-10 #for the relative change
            learning_rate = 500.0*255 #iniital learning rate for optimization
            rel_change_val = 1e5
            target_feat_rep = model(target_image).detach()
            old_feat_rep = model(base_image).detach() #also known as the old image
            old_obj = torch.linalg.norm(old_feat_rep - target_feat_rep) + \
                      beta*torch.linalg.norm(old_image - base_image)
            last_m_objs.append(old_obj)
            obj_threshold = 2.9

            # perform the attack as described in the paper to optimize
            # || f(x)-f(t) ||^2 + beta * || x-b ||^2
            for iteration in range(attack_iters):
                if iteration % 20 == 0:
                    the_diff_here = torch.linalg.norm(old_feat_rep - target_feat_rep) #get the diff
                    print("iter: %d | diff: %.3f | obj: %.3f" % (iteration, the_diff_here, old_obj))
                    print(" (%d) Rel change = %0.5e | lr = %0.5e | obj = %0.10e" \
                          % (iteration, rel_change_val, learning_rate, old_obj))
                # the main forward backward split (also with adam)
                if adam:
                    new_image, m, v, t = adam_one_step(model, m, v, t, old_image, target_image,
                                                       learning_rate)
                else:
                    new_image = forward_step(model, old_image, target_image,
                                             learning_rate, copy.deepcopy(target_feat_rep))
                new_image = backward_step(new_image, old_image, learning_rate, beta)

                # check stopping condition:  compute relative change in image between iterations
                rel_change_val = torch.linalg.norm(new_image-old_image)/torch.linalg.norm(new_image)
                if (rel_change_val < stopping_tol) or (old_obj <= obj_threshold):
                    print("! reached the object threshold -> stopping optimization !")
                    break

                # compute new objective value
                new_feat_rep = model(new_image).detach()
                new_obj = torch.linalg.norm(new_feat_rep - target_feat_rep) + \
                          beta*torch.linalg.norm(new_image - base_image)

                #find the mean of the last M iterations
                avg_of_last_m = sum(last_m_objs)/float(min(num_m, iteration+1))
                # If the objective went up, then learning rate is too big.
                # Chop it, and throw out the latest iteration
                if new_obj >= avg_of_last_m and (iteration % num_m/2 == 0):
                    learning_rate *= decay_coef
                    new_image = old_image
                else:
                    old_image = new_image
                    old_obj = new_obj
                    old_feat_rep = new_feat_rep

                if iteration < num_m-1:
                    last_m_objs.append(new_obj)
                else:
                    #first remove the oldest obj then append the new obj
                    del last_m_objs[0]
                    last_m_objs.append(new_obj)

                # yes that's correct. The following lines will never be reached, exactly
                # like in the original code. But adam optimization makes everything worse anyway..
                if iteration > attack_iters:
                    m = 0.
                    v = 0.
                    t = 0
                    adam = True

                difference = torch.linalg.norm(old_feat_rep - target_feat_rep)

            if difference < 3.5:
                poisoned_images[current_pertube_count] = old_image # old_image is overwritten
                poisoned_labels[current_pertube_count] = target
                current_pertube_count += 1


    print("\n[ Saving Dataset ]")
    # check for existing path and save the dataset
    if not os.path.isdir('./poison_frog/datasets/'):
        os.mkdir('./poison_frog/datasets/')

    # append poisons to the normal training data as described in the paper
    final_train_images = torch.cat((train_images, poisoned_images)).type(torch.FloatTensor)
    final_train_labels = torch.cat((train_labels, poisoned_labels)).type(torch.LongTensor)

    torch.save(final_train_images, './poison_frog/datasets/attack_images')
    torch.save(final_train_labels, './poison_frog/datasets/attack_labels')

    return target_image_id
