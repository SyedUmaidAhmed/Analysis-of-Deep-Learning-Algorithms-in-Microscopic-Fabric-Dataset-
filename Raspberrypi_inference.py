import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import random


def initialize_model_inference(model_name, num_classes, feature_extracte):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18().to(device)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet().to(device)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
      
    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn().to(device)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0().to(device)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121().to(device)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3().to(device)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)

    elif model_name == "mobilenet":
        """ Mobilenet
        """
        model_ft = models.mobilenet_v3_large().to(device)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "shufflenet":
        """ Shufflenet
        """
        model_ft = models.shufflenet_v2_x1_5().to(device)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


from PIL import Image
import random

# model_name = 'resnet'
# img_path = ''

all_labels = ['jeans','plain','shine']
correct_label = random.choice(all_labels)
#folder_label = os.path.join(testset,correct_label)
#file_names = next(os.walk(folder_label))[2]

#input_image = Image.open(os.path.join(folder_label, random.choice(file_names)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


best_weight = ('best_weights.pth')


input_image = Image.open('cgg.jpg')


# Preprocess image
preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

labels_map = ['jeans', 'plain', 'shine']

model_name="resnet"
num_classes=3
feature_extract=True

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
    
model = initialize_model_inference(model_name, num_classes, feature_extract)
model.load_state_dict(torch.load(best_weight,  map_location='cpu'))
model.eval()




#if torch.cuda.is_available():
  #input_batch = input_batch.to("cuda")
  #model.to("cuda")

with torch.no_grad():
  logits = model(input_batch)

# _, preds = torch.max(logits, 1)
preds = torch.topk(logits, k=1).indices.squeeze(0).tolist()

print("-----")
display(input_image)
for idx in preds:
  label = labels_map[idx]
  prob = torch.softmax(logits, dim=1)[0, idx].item()
  print(f"prediction: {label:<10} correct: {correct_label:<10} accuracy:({prob * 100:.2f}%)")
  # print(f"prediction: {label:<10} accuracy:({prob * 100:.2f}%)")
