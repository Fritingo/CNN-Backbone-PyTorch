import argparse
import logging
import sys
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import math

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils


from models import AlexNet, VGG, ConvNeXt

def SetDevice():  
    if opt.device == '':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = opt.device
    print('Device state:', device)
    return device

def SetDataset():
    if opt.data == 'cifa100':
        # transform
        transform = transforms.Compose([transforms.Resize(opt.size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      ]) #transforms.Compose([transforms.ToTensor()]) # 0~255 -> 0~1  //-1~1 , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        
        trainset, testset = torch.utils.data.random_split(data, [0.8, 0.2])
        trainLoader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
        testLoader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
        
        classes = ('apple ', 'aquarium_fish ', 'baby ', 'bear ', 'beaver  ', 'bed ', 'bee ', 'beetle ', 'bicycle ', 'bottle ', 'bowl ', 'boy ', 'bridge ', 'bus ', 'butterfly ', 'camel ', 'can ', 'castle ', 'caterpillar ', 'cattle ', 'chair ', 'chimpanzee ', 'clock ', 'cloud ', 'cockroach ', 'couch ', 'cra ', 'crocodile ', 'cup ', 'dinosaur ', 'dolphin ', 'elephant ', 'flatfish ', 'forest ', 'fox ', 'girl ', 'hamster ', 'house ', 'kangaroo ', 'keyboard ', 'lamp ', 'lawn_mower ', 'leopard ', 'lion ', 'lizard ', 'lobster ', 'man ', 'maple_tree ', 'motorcycle ', 'mountain ', 'mouse ', 'mushroom ', 'oak_tree ', 'orange ', 'orchid ', 'otter ', 'palm_tree ', 'pear ', 'pickup_truck ', 'pine_tree ', 'plain ', 'plate ', 'poppy ', 'porcupine ', 'possum ', 'rabbit ', 'raccoon ', 'ray ', 'road ', 'rocket ', 'rose ', 'sea ', 'seal ', 'shark ', 'shrew ', 'skunk ', 'skyscraper ', 'snail ', 'snake ', 'spider ', 'squirrel ', 'streetcar ', 'sunflower ', 'sweet_pepper ', 'table ', 'tank ', 'telephone ', 'television ', 'tiger ', 'tractor ', 'train ', 'trout ', 'tulip ', 'turtle ', 'wardrobe ', 'whale ', 'willow_tree ', 'wolf ', 'woman ', 'worm')
        print('training CIFA100')
        return trainLoader, testLoader, classes
    else:
        print('undone')

def SetModel(device):
    if opt.model == 'AlexNet':
        return AlexNet.AlexNet(opt.size).to(device)
    elif opt.model == 'VGG16':
        return VGG.VGG16(opt.size).to(device)
    elif opt.model == 'VGG19':
        return VGG.VGG19(opt.size).to(device)
    elif opt.model == 'ConvNeXt':
        return ConvNeXt.convnext_tiny(opt.size).to(device)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--data', type=str, default='cifa100')
    parser.add_argument('--model', type=str, default='AlexNet')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=8)
    
    opt = parser.parse_args()

    device = SetDevice()
    trainLoader, testLoader, classes = SetDataset()
    model = SetModel(device)
    print(model)

    torch.manual_seed(0)
    np.random.seed(0)
    cudnn.benchmark = True
    
    # set loss optimizer
    loss_fnc = nn.CrossEntropyLoss()
    lr = 4e-3
    epochs = 250
    momentum = 0.9

    
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
    # train
    for epoch in range(epochs):
        running_loss = 0.0
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.05)
        # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=momentum, nesterov=True)
    
        def one_cycle(y1=0.0, y2=1.0, steps=100):
            if steps < 20:
                return lambda x: 1e-6 + (lr/20)*steps
            else:
                return lambda x: ((1 - math.cos(x * math.pi / (steps + 1))) / 2) * (y2 - y1) + y1
        
        lf = one_cycle(lr, 1e-6, epoch)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        for times, data in enumerate(trainLoader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = loss_fnc(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print statistics
            running_loss += loss.item()

            if times % 100 == 99 or times+1 == len(trainLoader):
                print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, epochs, times+1, len(trainLoader), running_loss/times))
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testLoader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                
                _, predicted = torch.max(outputs.data, 1)
                
                total += 1
                
                correct += (predicted == labels)


        print('Accuracy of Test Dataset: %d %%' % (100 * correct / total))
    
    print('Finished Training')


    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))
    with torch.no_grad():
        for data in testLoader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            class_total[int(labels)] += 1
            
            class_correct[int(labels)] += (predicted == labels)
            


    for i in range(100):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    torch.save(model.state_dict(), "weights/" + opt.data + "_" + opt.model + ".pt")