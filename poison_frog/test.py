"""library module to test the accuracy of a model and check if the chosen target image gets
missclassified correctly
"""
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from .dataset import TensorDataset


def get_loader() -> DataLoader:
    """helper function to initialize and provide a CIFAR10 testset loader"""
    data_path = './poison_frog/datasets/'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_data = torch.load(os.path.join(data_path, "sanitized_images_test"))
    test_labels = torch.load(os.path.join(data_path, "sanitized_labels_test"))

    test_dataset = TensorDataset(test_data, test_labels, transform=transforms.Compose([
                transforms.Resize(299),
                transforms.ToTensor(),
                normalize]))

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    return test_loader

def test_model(device_name: str, target_id: int, new_class: int) -> bool:
    """
    Main function to perform the test on a chosen model

    :return: True if the attack was successful, False if not
    """
    global device
    device = device_name
    test_loader = get_loader()

    model = models.inception_v3(pretrained=True)
    model.aux_logits = False
    checkpoint = torch.load('./poison_frog/model_saves/poisoned_model',
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['net'])

    print('\n[ Evaluate Model Accuracy ]')
    model.eval()
    model.to('cpu')

    for idx, (input, target) in enumerate(test_loader):
        if idx == target_id:
            output = model(input)
            _, predicted = output.max(1)

            if predicted == target:
                print("\n\033[91m Attack failed: classified image with ID:{} and " \
                      "class:{} correctly \033[0m".format(target_id, int(target)))
                return False
            elif predicted == new_class:
                print("\n\033[92m Attack successful: classified image with ID: {} and " \
                      "class:{} as class {} instead \033[0m".format(target_id,
                                                                    int(target), int(predicted)))
                return True
            else:
                print("\n\033[91m Attack failed: classified image with ID:{} and " \
                      "class:{} as class {} instead \033[0m".format(target_id,
                                                                    int(target), int(predicted)))
                return False
