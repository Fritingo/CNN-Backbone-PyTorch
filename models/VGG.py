import torch
import torch.nn as nn
from thop import profile

class VGG16(nn.Module):
    def __init__(self, input_size=256, class_num=100):
        super(VGG16, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1) 
        self.conv1_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # classifier
        self.fc0 = nn.Linear(512*(input_size//32)*(input_size//32), 4096)
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, class_num)

    def forward(self,x):
        
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn1(self.conv1_1(x)))
        x = self.maxpool(x)

        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn2(self.conv2_1(x)))
        x = self.maxpool(x)

        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn3(self.conv3_1(x)))
        x = self.act(self.bn3(self.conv3_1(x)))
        x = self.maxpool(x)

        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn4(self.conv4_1(x)))
        x = self.act(self.bn4(self.conv4_1(x)))
        x = self.maxpool(x)

        x = self.act(self.bn4(self.conv4_1(x)))
        x = self.act(self.bn4(self.conv4_1(x)))
        x = self.act(self.bn4(self.conv4_1(x)))
        x = self.maxpool(x)
        
        x = torch.flatten(x, start_dim=1)

        x = self.act(self.fc0(x))
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    

class VGG19(nn.Module):
    def __init__(self, input_size=256, class_num=100):
        super(VGG19, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1) 
        self.conv1_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # classifier
        self.fc0 = nn.Linear(512*(input_size//32)*(input_size//32), 4096)
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, class_num)

    def forward(self,x):
        
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn1(self.conv1_1(x)))
        x = self.maxpool(x)

        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn2(self.conv2_1(x)))
        x = self.maxpool(x)

        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn3(self.conv3_1(x)))
        x = self.act(self.bn3(self.conv3_1(x)))
        x = self.act(self.bn3(self.conv3_1(x)))
        x = self.maxpool(x)

        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn4(self.conv4_1(x)))
        x = self.act(self.bn4(self.conv4_1(x)))
        x = self.act(self.bn4(self.conv4_1(x)))
        x = self.maxpool(x)

        x = self.act(self.bn4(self.conv4_1(x)))
        x = self.act(self.bn4(self.conv4_1(x)))
        x = self.act(self.bn4(self.conv4_1(x)))
        x = self.act(self.bn4(self.conv4_1(x)))
        x = self.maxpool(x)
        
        x = torch.flatten(x, start_dim=1)

        x = self.act(self.fc0(x))
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
if __name__ == '__main__':
    input = torch.randn(32, 3, 32, 32)
    model = VGG16(32, 100)
    y = model(input)
    print('output shape:', y.size())

    # time.sleep(10)
    flops, params = profile(model, inputs=(input, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')