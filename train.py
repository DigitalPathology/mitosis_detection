from torch.utils.data import DataLoader
from torchvision import transforms
import PIL
import matplotlib.pyplot as plt
import numpy as np
import sys
from tensorboardX import SummaryWriter
import scipy.ndimage
import scipy.misc
import time
import math
import tables
import random
from sklearn.metrics import confusion_matrix
import torch
from torch import nn
import torchvision

from netNew import model, optimizer, loss_fn
from loadImagesNew import transform_ori, transform_ori_2
from torch.autograd import Variable

from torch.backends import cudnn
import numpy as np

import random

import time
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

from loadImagesNew import train_load, train_dataset, test_load, test_load_2,valid_dataset
from netNew import model, optimizer, loss_fn
from torch.optim.lr_scheduler import StepLR

from livelossplot import PlotLosses

from datetime import datetime


import cv2

import torch.nn.functional as F
#import pydensecrf.densecrf as dcrf
from PIL import Image
#from v2.UNETnew import R2U_Net, UNetnew
# dataname = "nucleiSmall"
dataname = "epistroma"

ignore_index = -100  #Unet has the possibility of masking out pixels in the output image, we can specify the index value here (though not used)
gpuid = 0

phases = ["train","val"]
validation_phases = ["val"]
n_classes = 2

checkpoint_path = 'mitosis_detection_model.pth'
# Early stopping intialization
epochs_no_improve = 0

#maximum number of epochs with no improvement in validation loss for early stopping
max_epochs_stop = 3


def createTracedModel(model, random_input):
    traced_net = torch.jit.trace(model, random_input)
    traced_net.save("HE_mitosis_trace.pt")

    print("Success - model_trace was saved!")


#helper function for pretty printing of current time and remaining time
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


#model.load_state_dict(torch.load('mitosis_detection_model.pth'))


count = 0

# Freeze model weights
for name, param in model.named_parameters():
    count = count + 1
    if count <= 16:
        print(name)
        param.requires_grad = False

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=0.00001,momentum=0.9, nesterov=True, weight_decay=0.0005)


num_epochs = 400

# Define the lists to store the results of loss and accuracy
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
#
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(0)
#     cudnn.benchmark = True


# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_loss_on_test = np.Infinity
start_time = time.time()


# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')



scheduler = StepLR(optimizer, step_size=2, gamma=0.96)

liveloss = PlotLosses()

# Training
for epoch in range(num_epochs):

    logs = {}
    cmatrix = {key: np.zeros((2, 2)) for key in phases}

    scheduler.step()
    # Print Learning Rate
    print('Epoch:', epoch, 'LR:', scheduler.get_lr())

    # Learn only the FC layer weights until a certain epoch, then learn all weights
    if epoch == 200:
        cnt = 0
        # Freeze model weights
        for name, param in model.named_parameters():
            cnt = cnt + 1
            if cnt <= 16:
                print(name)
                param.requires_grad = True
        # add the unfrozen fc2 weight to the current optimizer
        optimizer.add_param_group({'params': model.features.parameters()})

    # Reset these below variables to 0 at the begining of every epoch
    start = time.time()
    correct = 0
    iter_loss = 0.0
    all_loss = {key: torch.zeros(0).to(device) for key in phases}

    model.train()  # Put the network into training mode
    phase = "train"

    for i, (inputs, labels) in enumerate(train_load):

        # Convert torch tensor to Variable
        inputs = Variable(inputs)
        labels = Variable(labels)

        inputs = inputs.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()  # Clear off the gradient in (w = w - gradient)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights

        all_loss[phase] = torch.cat((all_loss[phase], loss.detach().view(1, -1)))

        # Record the correct predictions for training data
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()

        predicted = predicted.cpu().numpy().flatten()
        labels    = labels.cpu().numpy().flatten()

        cmatrix[phase] = cmatrix[phase] + confusion_matrix(labels, predicted, labels=range(n_classes))

        del loss
        del outputs

    all_loss[phase] = all_loss[phase].cpu().numpy().mean()

    train_accuracy = 100 * correct / len(train_dataset)

    prefix = ''
    logs[prefix + 'log loss'] = all_loss["train"]
    logs[prefix + 'accuracy'] = train_accuracy

    # Testing
    loss = 0.0
    correct = 0

    model.eval()  # Put the network into evaluation mode
    phase = "val"

    for i, (inputs, labels) in enumerate(test_load):

        # Convert torch tensor to Variable
        inputs = Variable(inputs)
        labels = Variable(labels)

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)  # Calculate the loss
        # Record the correct predictions for training data
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()

        all_loss[phase] = torch.cat((all_loss[phase], loss.detach().view(1, -1)))

        predicted = predicted.cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()

        cmatrix[phase] = cmatrix[phase] + confusion_matrix(labels, predicted, labels=range(n_classes))

        del outputs
        del loss

    all_loss[phase] = all_loss[phase].cpu().numpy().mean()

    test_accuracy = 100 * correct / len(valid_dataset)
    stop = time.time()

    if phase in validation_phases:
        print(f'{phase}/TN', cmatrix[phase][0, 0], epoch)
        print(f'{phase}/TP', cmatrix[phase][1, 1], epoch)
        print(f'{phase}/FP', cmatrix[phase][0, 1], epoch)
        print(f'{phase}/FN', cmatrix[phase][1, 0], epoch)
        print(f'{phase}/TNR', cmatrix[phase][0, 0] / (cmatrix[phase][0, 0] + cmatrix[phase][0, 1]), epoch)
        print(f'{phase}/TPR', cmatrix[phase][1, 1] / (cmatrix[phase][1, 1] + cmatrix[phase][1, 0]), epoch)

    print('%s ([%d/%d] %d%%), train loss: %.4f test loss: %.4f' % (timeSince(start_time, (epoch + 1) / num_epochs),
                                                                   epoch + 1, num_epochs,
                                                                   (epoch + 1) / num_epochs * 100, all_loss["train"],
                                                                   all_loss["val"]), end="")

    print(
        ' Epoch {}/{}, Training Accuracy: {:.3f}, Testing Acc: {:.3f}, Time: {}s'
            .format(epoch + 1, num_epochs, train_accuracy, test_accuracy,
                    stop - start))


    prefix = 'val_'


    logs[prefix + 'log loss'] = all_loss["val"]
    logs[prefix + 'accuracy'] = test_accuracy

    liveloss.update(logs)



    # if current loss is the best we've seen, save model state with all variables
    # necessary for recreation
    if all_loss["val"] < best_loss_on_test:
        epochs_no_improve = 0
        best_loss_on_test = all_loss["val"]
        print("  **")
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, checkpoint_path)


    # else:
    #     print("")
    #     epochs_no_improve += 1
    #     # Trigger early stopping
    #     if epochs_no_improve >= max_epochs_stop:
    #         print('Early stopping')
    #         # Load the best state dict
    #         state = torch.load(checkpoint_path, map_location='cpu')
    #         model.load_state_dict(state['state_dict'])
    #         optimizer.load_state_dict(state['optimizer'])


liveloss.draw()

#Run this if you want to save the model
#torch.save(model.state_dict(),'mitosis_detection_model.pth')


model.eval()

for i, (inputs, labels) in enumerate(test_load_2):

    # show images
    #imshow(torchvision.utils.make_grid(inputs))

    # Convert torch tensor to Variable
    inputs = Variable(inputs)
    labels = Variable(labels)

    inputs = inputs.to(device)
    labels = labels.to(device)

    print("girdi")
    if i == 0:
        createTracedModel(model, inputs)
        break







