import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=62, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),                                  # pytoch丢弃小数部分
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),#max（0，x）
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(#一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
                                        # 同时以神经网络模块为元素的有序字典也可以作为传入参数。
            nn.Dropout(p=0.5),#随机失活神经元
            nn.Linear(128 * 6 * 6, 2048),#定义一个线性变换
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()#pytorch自动初始化权重

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)#展平处理
        x = self.classifier(x)
        return x
#初始化权重
    def _initialize_weights(self):
        for m in self.modules():#继承父类，返回一个迭代器，遍历网络中的所有模块
            if isinstance(m, nn.Conv2d):#判断是否是nn.Conv2d
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)#采用正太分布进行初始化，均值0，方差0.01
                nn.init.constant_(m.bias, 0)
