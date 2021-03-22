import time
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from model import *
from utils import *
from parameterPrepared import *

device = torch.device('cuda:0')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def adjust_learning_rate(optimizer, epoch):
    update_list = [15, 17, 20]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return

def test():
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100. * float(correct) / len(test_loader.dataset)

    print('acc is {}'.format(acc))



if __name__ == '__main__':
    setup_seed(int(time.time()))

    print('==> Preparing data..')
    train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(),download=True)
    test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    print('******Initializing model******')
    model = Net(abits=8, wbits=1, q_type=0)
    model.to(device)
    #参数初始化
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
    #损失函数及优化器
    base_lr = float(0.01)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.00001)

    is_train=True

    if is_train:
        for epoch in range(1, 30+1):
            adjust_learning_rate(optimizer, epoch)
            train(epoch)
            test()

        param = model.state_dict()
        W1,W2,W3,W4,g0,g1,g2,g3,g4,b0,b1,b2,b3,b4=parameterTransform(param)

        cnt = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            x = f(g0,b0,data,True)
            x = torch.nn.functional.conv2d(x, weight=W1, stride=1, bias=None, padding=1)  # 128,16,28,28
            x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)                            #128,16,14,14
            x = f(g1,b1,x,True)
            x = torch.nn.functional.conv2d(x, weight=W2, stride=1, bias=None, padding=1)  # 128,32,14,14
            x = f(g2,b2,x,True)
            x = torch.nn.functional.conv2d(x, weight=W3, stride=1, bias=None, padding=1)  # 128,64,14,14
            x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)                            # 128,64,7,7
            x = f(g3,b3,x,True)
            x = torch.nn.functional.conv2d(x, weight=W4, stride=1, bias=None, padding=1)  # 128,10,7,7
            x = f(g4,b4,x,False)                                                 #事实上也是f的特殊形式

            x = torch.nn.functional.avg_pool2d(x, kernel_size=7)            # (128,10,1,1)
            x = x.view(x.size(0), -1)                                       # (128,10)
            output = torch.argmax(x, dim=1)

            for i in range(output.size(0)):
                if output[i] == target[i]:
                    cnt += 1
        print("inference acc is {}".format(cnt / 10000))


###############################################################################################################
#####################由上述推理过程可以看出，整个网络的运算可简化为卷积,池化以及一个f函数###################################



