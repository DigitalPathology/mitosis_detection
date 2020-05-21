import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Transformation for image
# transform_ori = transforms.Compose([transforms.RandomResizedCrop(100),  # create 64x64 image
#                                     transforms.RandomHorizontalFlip(),  # flipping the image horizontally
#                                     transforms.ToTensor(),  # convert the image to a Tensor
#                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                          std=[0.229, 0.224, 0.225])])  # normalize the image

transform_ori = transforms.Compose([
                                    transforms.RandomVerticalFlip(),  # flipping the image vertically
                                    transforms.RandomHorizontalFlip(),  # flipping the image horizontally
                                    transforms.ToTensor(),  # convert the image to a Tensor
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                   ])  # normalize the image


transform_ori_2 = transforms.Compose([
                                    transforms.ToTensor(), # convert the image to a Tensor
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                   ])  # normalize the image


# Load our dataset
dataset = datasets.ImageFolder(root='C:/he_images/mitoz_dataset_augmented',transform=transform_ori)

#test_dataset  = datasets.ImageFolder(root='C:/he_images/tileWiseTestInputDir_YakindanAlinanResimler',transform=transform_ori_2)

test_dataset_2  = datasets.ImageFolder(root='C:/Users/Sercan/PycharmProjects/samplepyTorch/data/images/sample_inputs',transform=transform_ori_2)
#test_dataset_3  = datasets.ImageFolder(root='C:/Users/Sercan/PycharmProjects/samplepyTorch/data/images/region_images',transform=transform_ori_2)
#test_dataset_4  = datasets.ImageFolder(root='C:/Users/Sercan/PycharmProjects/samplepyTorch/data/images/region_images_2',transform=transform_ori_2)



validation_split = .1
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_dataset = torch.utils.data.Subset(dataset, train_indices)
valid_dataset = torch.utils.data.Subset(dataset, val_indices)




# Make the dataset iterable
batch_size = 4
train_load = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)  # Shuffle to create a mixed batches of 100 of cat & dog images

test_load = torch.utils.data.DataLoader(dataset=valid_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)


test_load_2 = torch.utils.data.DataLoader(dataset=test_dataset_2, batch_size=1, shuffle=False)
#test_load_3 = torch.utils.data.DataLoader(dataset=test_dataset_3, batch_size=1, shuffle=True)
#test_load_4 = torch.utils.data.DataLoader(dataset=test_dataset_4, batch_size=1, shuffle=True)






# # Show a batch of images
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.figure(figsize=(20, 20))
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show(block=True)
# print('There are {} images in the training set'.format(len(train_dataset)))
# print('There are {} images in the test set'.format(len(test_dataset)))
# print('There are {} images in the train loader'.format(len(train_load)))
# print('There are {} images in the test loader'.format(len(test_load)))
#
# # get some random training images
# dataiter = iter(test_load_2)
# images, labels = dataiter.next()
#
# # show images
# #imshow(torchvision.utils.make_grid(images))

