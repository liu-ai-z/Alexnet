import torch
from torch.utils.data import DataLoader
import os
import json
import time
import torch.optim as optim
from my_model import Alexnet
from torchvision import transforms,datasets
print(torch.cuda.is_available())
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

    data_root = os.getcwd()
    print(data_root)

    train_path = data_root + '/data/train'
    val_path = data_root + '/data/test'
    print(train_path,val_path)

    data_transform = {

        'train':transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.5],std= [0.5])]
                                   ),

        'val':transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.5],std= [0.5])]
                                 )
                    }


    train_datasets = datasets.ImageFolder(root = train_path, transform = data_transform['train'])
    # print(len(train_datasets))
    # print(train_datasets[1][1])  #标签【】【1】
    #
    # print(train_datasets[1][0])  #数据【】【0】
    val_datasets = datasets.ImageFolder(root = val_path, transform = data_transform['val'])


    class_list = train_datasets.class_to_idx
    # print(class_list.items())
    # print(class_list)
    json_str = json.dumps(class_list, indent=4,ensure_ascii=False)
    # cla_dict = dict((val, key) for key, val in class_list.items())
    with open('class_list.json','w') as json_file:
        json_file.write(json_str)

    # print(json_str)
    batch_size = 2

    train_load = DataLoader(dataset = train_datasets, batch_size = batch_size , shuffle = True, num_workers= 0)
    val_load = DataLoader(dataset = val_datasets,batch_size = batch_size, shuffle = True, num_workers= 0 )

    net = Alexnet(58)
    net.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),0.0001)
    save_path = 'AlexNet_my.pth'
    best_acc = 0.0
    # loss_list = []
    # acc_list = []

    # a = tuple(([1, 1], [1, 2], [1, 3]))
    # print(a)
    # for i, item in enumerate(a):
    #     key, val = item
    #     print(i, item, key, val)
    # ([1, 1], [1, 2], [1, 3])
    # 0[1, 1]1 1
    # 1[1, 2]1 2
    # 2[1, 3]1 3

    for i in range(100):
        net.train()
        run_lossing = 0
        t1 = time.perf_counter()
        for step , datas in enumerate(train_load,start = 0):  #step返回多少个batch，data返回每个batch数据
            # print(len(train_load))
            # print(type(datas))    #list
            # print(len(datas))     #2
            # print(datas)
            image,lable = datas
            # print(step,'\n',lable)
            optimizer.zero_grad()
            output = net(image.to(device))
            # print(output.shape)  #torch.Size([2, 58])
            loss = loss_function(output,lable.to(device))
            # print(type(loss))
            # print(loss.to('cpu').detach().numpy())
            loss.backward()
            optimizer.step()

            run_lossing += loss.item()
            # print(step)
            rate = (step + 1) / len(train_load)
            a = "*"*int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {0:^3.0f}%[{1}->{2}]{3:.3f}".format(int(rate * 100), a, b,loss), end="")#end
        print(time.perf_counter() - t1)
        #
        net.eval()
        acc= 0.0
        with torch.no_grad():
            for step , datas in enumerate(val_load , start = 0):
                image ,val_label = datas
                output = net(image.to(device))

                predict_y = torch.max(output,dim=1)[1]
                acc += (predict_y == val_label.to(device)).sum().item()

            val_accuracy = acc / len(val_load)
            if val_accuracy > best_acc:
                best_acc = val_accuracy






if __name__ == '__main__':
    main()
















