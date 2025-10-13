# When you run it on Google Colab, use this code

!pip install torch torchvision
!pip install gdown
!gdown --id 1y1b4XF0kiEK-y6nphNEi7e_4e0UB5weP
!unzip -q *.zip -d /content/cifar10

from torch.utils.data import Dataset
import os
import pickle
import numpy as np
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, root, train = True, transform = None):
        # if train:
        #     self.root = os.path.join(root, "train")
        # else:
        #     self.root = os.path.join(root, "test")
        categories = ["data_batch_{}".format(i) for i in range(1, 6)]
        self.images = []
        self.labels = []
        for category in categories:
            file_path = os.path.join(root, category)
            with open(file_path, "rb") as fo:
                images = pickle.load(fo, encoding = "bytes")
                self.images.extend(images[b'data'])
                self.labels.extend(images[b'labels'])
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        image = np.reshape(image, (3, 32, 32))
        image = np.transpose(image, (1, 2, 0))
        # image = Image.fromarray(image)
        image = self.transform(image)
        label = self.labels[index]
        return image, label

# if __name__ == "__main__":
#     data = MyDataset(root = "C:/Users/Admin/PycharmProjects/Deep_learning_beginer/cifar10/cifar-10-batches-py", train = True)
#     print(data.__len__())
#     image, label = data.__getitem__(234)
#     image = Image.fromarray(image)
#     image.show()
import torch
import torch.nn as nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fcl1 = nn.Sequential(
            nn.Linear(in_features = 3*32*32, out_features = 256),
            nn.ReLU()
        )
        self.fcl2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU()
        )
        self.fcl3 = nn.Sequential(
            nn.Linear(in_features = 512, out_features = 1024),
            nn.ReLU()
        )
        self.fcl4 = nn.Sequential(
            nn.Linear(in_features = 1024, out_features = 512),
            nn.ReLU()
        )
        self.fcl5 = nn.Sequential(
            nn.Linear(in_features = 512, out_features = num_classes),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fcl1(x)
        x = self.fcl2(x)
        x = self.fcl3(x)
        x = self.fcl4(x)
        x = self.fcl5(x)
        return x

# if __name__ == "__main__":
#     fake_input = torch.rand(8, 3, 32, 32)
#     model = SimpleNeuralNetwork()
#     result = model(fake_input)
#     print(result)
# from data_set import MyDataset
# from create_neural_network import SimpleNeuralNetwork
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torchvision.transforms import ToTensor


if __name__ == "__main__":
    num_epochs = 50
    root = "/content/cifar10/cifar10/cifar-10-batches-py"
    train_dataset = MyDataset(root, train = True, transform = ToTensor())
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = 64,
        num_workers = 3,
        shuffle = True,
        drop_last = True
    )
    test_dataset = MyDataset(root, train = False, transform = ToTensor())
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = 64,
        num_workers = 3,
        shuffle = False,
        drop_last = False
    )
    model = SimpleNeuralNetwork(num_classes = 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    num_train_dataloader = len(train_dataloader)
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(num_epochs):
        model.train()
        for iter, (images, labels) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            print("Epoch {}/{}, iter {}/{}: loss: {}".format(epoch + 1, num_epochs, iter + 1,
                                                             num_train_dataloader, loss))
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()