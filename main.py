import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F

# constants
Batch_Size = 32
Epoch = 30


# network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # 32*32*1 -> 28*28*24 -> 14*14*24
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # input_size=(6*28*28)，output_size=(6*14*14)
        )
        # 14*14*24 -> 12*12*48 -> 6*6*48
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # input_size=(6*28*28)，output_size=(6*14*14)
        )
        # 6*6*48 -> 2*2*64 -> 1*1*64
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # input_size=(6*28*28)，output_size=(6*14*14)
        )
        # flatten
        self.flatten = nn.Flatten()
        # fc1 5*5*16 -> 120
        # self.fc1 = nn.Sequential(
        #     nn.Linear(5 * 5 * 16, 120),
        #     nn.ReLU()
        # )
        # # fc2 120 -> 84
        # self.fc2 = nn.Sequential(
        #     nn.Linear(120, 84),
        #     nn.ReLU(),
        #     nn.Linear(84, 10)
        # )
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc(x)
        x = F.log_softmax(x)
        return x


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
    optimizer = optim.Adam(network.parameters(), lr=1e-3)  # TODO: lr

    print(next(network.parameters()).device)
    # training
    print('start training')
    for epoch in range(Epoch):
        sum_loss = 0.0
        number_of_sub_epoch = 0

        for images, labels in train_loader:
            out = network(images.to(device))
            optimizer.zero_grad()
            # count the loss
            loss = criterion(out, labels.to(device))
            loss.backward()
            # learning
            optimizer.step()

            sum_loss += loss.item()
            number_of_sub_epoch += 1
        print("Epoch {}: Loss: {}".format(epoch, sum_loss / number_of_sub_epoch))

    # test
    correct = 0
    total = 0

    network.eval()
    for images, labels in test_loader:
        out = network(images.to(device))
        _, predicted = torch.max(out.data, 1)  # outputs.data in shape of BS x 10  -->  BS x 1
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum()
        # if labels.size(0) != (predicted == labels).sum():
        #     print(labels.size(0))
        #     print(labels)
        #     print((predicted == labels).sum())
        #     print(predicted)
    print('Test Accuracy: {:.1f}%'.format(correct / total * 100))
