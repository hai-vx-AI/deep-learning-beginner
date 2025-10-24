from data_set import MyDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from create_neural_network import CNN
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import classification_report
from torchvision.transforms import Compose, Resize, ToTensor

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size")
    parser.add_argument("--epochs", type = int, default = 10, help = "Number of epochs")
    parser.add_argument("--root", type = str, default = "C:/Users/Admin/PycharmProjects/Deep_learning_beginer/cifar10/cifar-10-batches-py",
                        help = "Root of path dataset")
    parser.add_argument("--image_size", type = int, default = 224, help = "Image size")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    root = args.root
    transform = Compose([
        Resize(args.image_size),
        ToTensor()
    ])
    train_dataset = MyDataset(root, train = True, transform = transform)
    training_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = 3
    )
    test_dataset = MyDataset(root, train = False, transform = transform)
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        drop_last = False,
        num_workers = 3
    )
    model = CNN(num_classes = 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    num_epochs = args.epochs

    if torch.cuda.is_available():
        model.cuda()

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm.tqdm(training_dataloader, colour = "Green")
        for iter, (images, labels) in enumerate(training_dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            # Forward
            output = model(images)
            loss_values = criterion(output, labels)

            progress_bar.set_description("epoch {}/{}, iter {}/{}, loss: {:.3f}".format(epoch + 1, num_epochs,
                                                                                        iter + 1, len(training_dataloader), loss_values))

            # Backward
            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions.cpu(), dim = 1)
                all_predictions.extend(indices)
        all_predictions = [prediction.item() for prediction in all_predictions]
        all_labels = [label.item() for label in all_labels]
        print(classification_report(all_labels, all_predictions))