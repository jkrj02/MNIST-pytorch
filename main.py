import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F

# constants
Batch_Size = 36
Epoch = 30
loss_list = []
accu_list = []


# network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # 32*32*1 -> 28*28*24 -> 14*14*24
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # input_size=(6*28*28)，output_size=(6*14*14)
        )
        # 14*14*24 -> 12*12*48 -> 6*6*48
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # input_size=(6*28*28)，output_size=(6*14*14)
        )
        # 6*6*48 -> 2*2*64 -> 1*1*64
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # input_size=(6*28*28)，output_size=(6*14*14)
        )
        # flatten
        # self.flatten = nn.Flatten()
        # self.fc = nn.Sequential(
        #     nn.Linear(2 * 2 * 64, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 10)
        # )
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def train(net, dataloader, optimize, loss_func):
    print('start training')
    for epoch in range(Epoch):
        sum_loss = 0.0
        number_of_sub_epoch = 0

        for images, labels in dataloader:
            out = net(images.to(device))
            optimize.zero_grad()
            # count the loss
            loss = loss_func(out, labels.to(device))
            loss.backward()
            # learning
            optimize.step()
            # calculate loss
            sum_loss += loss.item()
            number_of_sub_epoch += 1

        # calculate average loss
        epoch_loss = sum_loss / number_of_sub_epoch
        loss_list.append(epoch_loss)
        print("Epoch {}: loss: {}".format(epoch, epoch_loss))

        accu_list.append(test(net, test_loader).cpu())
    print('training done!')


def test(net, dataloader):
    correct = 0
    total = 0

    net.eval()
    for images, labels in dataloader:
        out = net(images.to(device))
        _, predicted = torch.max(out.data, 1)  # outputs.data in shape of BS x 10  -->  BS x 1
        total += labels.size(0)
        correct += (predicted == labels.to(device)).type(torch.float).sum()

    return correct / total * 100


# main
if __name__ == '__main__':
    # get dataset
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, 0.5)])
    train_dataset = torchvision.datasets.MNIST(root='./data/MNIST',
                                               train=True,
                                               transform=transforms,
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='.data/MNIST',
                                              train=False,
                                              transform=transforms,
                                              download=True)

    # load data
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=Batch_Size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=Batch_Size,
                                              shuffle=False)

    # # check picture
    # data_iter = iter(train_loader)
    # images, label = next(data_iter)  # images are 32 x 1 x 28 x 28
    # np_img = torchvision.utils.make_grid(images[0]).numpy()
    # plt.imshow(np.transpose(np_img, (1, 2, 0)))  # H x W x color_channel
    # print(label[0])

    # optimizer
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    # print(device)
    network = Network().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=1e-4)

    # training
    train(network, train_loader, optimizer, criterion)

    # test
    print('Test Accuracy: {:.1f}%'.format(test(network, test_loader)))

    # draw the picture
    x_loss = range(1, len(loss_list)+1)
    # x_accu = [i for i in range(Epoch) if (i + 1) % 5 == 0]
    x_accu = range(1, Epoch+1)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # left one
    ax1.plot(x_loss, loss_list, 'b', label='Loss')
    ax1.legend(loc='upper left')
    ax1.set_ylabel('Loss')
    # right one
    ax2 = ax1.twinx()
    ax2.plot(x_accu, accu_list, 'r', label='Accuracy', linestyle='none', marker='o')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    plt.show()
