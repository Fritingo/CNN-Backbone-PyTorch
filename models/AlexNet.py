import torch
import torch.nn as nn
from thop import profile

# 2293MiB 35 %
class AlexNet(nn.Module):
    def __init__(self, input_size=224, class_num=100):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 11, stride=4, padding=2) # orignal is stride 4 kernel size 11 
        self.conv2 = nn.Conv2d(64, 192, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(192, 384, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.act = nn.ReLU()

        self.fc0 = nn.Linear(256, 4096)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, class_num)

    def forward(self,x):
        
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))

        x = self.maxpool(x)
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.maxpool(x)
        
        x = torch.flatten(x, start_dim=1)
        
        x = self.act(self.fc0(x))
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
if __name__ == '__main__':
    input = torch.randn(16, 3, 224, 224)
    model = AlexNet(224)
    y = model(input)
    print('output shape:', y.size())

    # time.sleep(10)
    flops, params = profile(model, inputs=(input, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')