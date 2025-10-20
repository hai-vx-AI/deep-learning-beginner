from data_set import MyDataset
from create_neural_network import SimpleCNN
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch
from sklearn.metrics import classification_report
from argparse import ArgumentParser
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--root", "-r",type = str, default = "C:/Users/Admin/PycharmProjects/Deep_learning_beginer/cifar10/cifar-10-batches-py",
                        help = "Root of the dataset")
    parser.add_argument("--batch-size","-b",type = int, default = 64, help = "Number of batch size")
    parser.add_argument("--epochs", "-e",type = int, default = 100, help = "Number of epochs")
    parser.add_argument("--image-size", "-i",type = int, default = 224, help = "Image size")
    parser.add_argument("--logging", "-l", type = str, default = "tensorboard")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    epochs = args.epochs
    root = args.root
    train_dataset = MyDataset(root, train=True, transform=ToTensor())
    training_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=3,
        drop_last=True
    )
    test_dataset = MyDataset(root, train=False, transform=ToTensor())
    testing_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=3,
        drop_last=False
    )

    summary_writer = SummaryWriter(args.logging)

    model = SimpleCNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_batches = len(training_dataloader)
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress_bar = tqdm(training_dataloader, colour = "yellow")
        for i, (images, labels) in enumerate(progress_bar):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            # forward
            outputs = model(images)
            loss_values = criterion(outputs, labels)
            progress_bar.set_description("epoch: {}/{}, backward: {}/{}, loss: {:.3f}".format(epoch, epochs, i, num_batches, loss_values))
            # backward
            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()

        # model.eval()
        # all_predictions = []
        # all_labels = []
        # for i, (images, labels) in enumerate(testing_dataloader):
        #     all_labels.extend(labels)
        #     if torch.cuda.is_available():
        #         images = images.cuda()
        #         labels = labels.cuda()
        #
        #     with torch.no_grad():
        #         predictions = model(images)
        #         indices = torch.argmax(predictions.cpu(), dim = 1)
        #         all_predictions.extend(indices)
        #
        # all_labels = [label.item() for label in all_labels]
        # all_predictions = [prediction.item() for prediction in all_predictions]
        # print(classification_report(all_labels, all_predictions))