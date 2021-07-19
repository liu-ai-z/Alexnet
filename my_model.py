import torch.nn as nn
import torch

class Alexnet(nn.Module):

    def __init__(self,num_class = 62, init_weight = False):
        super(Alexnet,self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3,48,kernel_size=11,stride=4,padding=2), #input[224 224 3] output[55 55 48]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(48,128,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(128,192,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)#
        )
        self.classfier = nn.Sequential(

            nn.Dropout(0.5),
            nn.Linear(128*6*6,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,num_class)
        )


        if init_weight :
            self._initialize_weight()#pytorch当前版本直接对模型进行权重初始化


    def forward(self,X):
        X = self.feature(X)
        X = torch.flatten(X, start_dim =1)
        X = self.classfier(X)
        return X

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):#判断是否是nn.Conv2d
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')#正太分布初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)#采用正太分布进行初始化，均值0，方差0.01
                nn.init.constant_(m.bias, 0)



