from data_set import MyDataset
from create_neural_network import SimpleNeuralNetwork
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torchvision.transforms import ToTensor


if __name__ == "__main__":
    num_epochs = 50
    root = "C:/Users/Admin/PycharmProjects/Deep_learning_beginer/cifar10/cifar-10-batches-py"
    train_dataset = MyDataset(root, train = True, transform = ToTensor())
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = 16,
        num_workers = 3,
        shuffle = True,
        drop_last = True
    )
    test_dataset = MyDataset(root, train = False, transform = ToTensor())
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = 16,
        num_workers = 3,
        shuffle = False,
        drop_last = False
    )
    model = SimpleNeuralNetwork(num_classes = 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    num_train_dataloader = len(train_dataloader)
    for epoch in range(num_epochs):
        model.train()
        for iter, (images, labels) in enumerate(train_dataloader):
            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            print("Epoch {}/{}, iter {}/{}: loss: {}".format(epoch + 1, num_epochs, iter + 1,
                                                             num_train_dataloader, loss))
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()