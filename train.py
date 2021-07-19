# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from model import AlexNet
import os
import json
import time
import matplotlib.pyplot as plt

def main():

    def show_curve(ys, title):
        x = np.array(range(len(ys)))
        y = np.array(ys)
        plt.plot(x, y, c='b')
        plt.axis()
        plt.title('{} curve'.format(title))
        plt.xlabel('epoch')
        plt.ylabel('{}'.format(title))
        plt.show()

    def imshow(inp, title=None):
        plt.figure(figsize=(14, 3))
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)
        plt.show()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # 如果有默认使用GPU的设备使用GPU，如果没有使用CPU

    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(224),  # 中心裁剪到224*224像素大小
                                     transforms.RandomHorizontalFlip(),  # 水平方向随机翻转
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5], std=[0.5])]),
        "test": transforms.Compose([ transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5], std=[0.5])])}

    data_root = os.path.abspath(os.path.join(os.getcwd()))  # 获得当前地址，
    # 这里表示返回上上层目录
    image_path = data_root + "./data/"  # flower data set path
    train_dataset = datasets.ImageFolder(root=image_path + "train",
                                         transform=data_transform["train"])  # 对训练集进行预处理，采用的是对训练集使用得处理方式
       # 打印训练集图片个数
    print(len(train_dataset))


    flower_list = train_dataset.class_to_idx  # 分类名称对应得索引
    cla_dict = dict((val, key) for key, val in flower_list.items())  # 通过预测过后能直接得到类别
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4,ensure_ascii=False)  # 对字典编码 indent缩进个数
    with open('class_indices.json', 'w') as json_file:  # 保存在class_indices.json文件中
        json_file.write(json_str)

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    validate_dataset = datasets.ImageFolder(root=image_path + "test",
                                            transform=data_transform["test"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)
    class_names = train_dataset.classes
    datas, targets = next(iter(train_loader))
    # print(datas,targets)
    out = torchvision.utils.make_grid(datas, nrow=4, padding=10)#将图像拼接成一幅图像
    plt.rcParams["font.sans-serif"] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    imshow(out, title=[class_names[x] for x in targets])

    net = AlexNet()
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    save_path = './alexnet.pth'
    best_acc = 0.0

    Loss_list = []
    Accuracy_list = []

    for epoch in range(3):
        # train
        net.train()
        running_loss = 0.0
        t1 = time.perf_counter()
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            # print(step,labels)
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # print train process
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        print()
        print(time.perf_counter()-t1)
        Loss_list.append(running_loss / (len(train_dataset)))

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = acc / val_num

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                  (epoch + 1, running_loss / step, val_accurate))
        Accuracy_list.append(val_accurate)
    show_curve(Loss_list,'Train_loss')
    show_curve(Accuracy_list,'Test_accuracy')


if __name__ == '__main__':
    main()