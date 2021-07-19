# -*- coding:utf-8 -*-
import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)
# create model
model = AlexNet()
# load model weights
model_weight_path = "alexnet.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))

image_data= []
for root,dirs,files in os.walk(r"./predict"):
    pass
files.sort(key = lambda x: int(x[:-4]))

for i in range(len(files)):
    image_data.append(r"./predict/" + files[i])

Predict_kind = []
Predict_Proba = []

for i in image_data:
    img = Image.open(i).convert('RGB')
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    plt.show()
    # read class_indict
    try:
        json_file = open('./class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)
    # create model
    model = AlexNet()
    # load model weights
    model_weight_path = "alexnet.pth"
    model.load_state_dict(torch.load(model_weight_path,map_location=device))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    f = open("data/model_Weight.txt",'a')
    print(i[10:],class_indict[str(predict_cla)], predict[predict_cla].item())
    Predict_kind.append(class_indict[str(predict_cla)])
    Predict_Proba.append(predict[predict_cla].item())

with open("data.txt", "w") as f:
    for i in range(len(Predict_kind)):
        f.write(image_data[i][10:]+'     '+Predict_kind[i]+'     '+str(Predict_Proba[i])+'\n')


