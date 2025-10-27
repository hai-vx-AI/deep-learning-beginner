from data_set import MyDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from create_neural_network import SimpleCNN
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import classification_report, accuracy_score
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type = int, default = 4, help = "Batch size")
    parser.add_argument("--epochs", type = int, default = 10, help = "Number of epochs")
    parser.add_argument("--root", type = str, default = "C:/Users/Admin/PycharmProjects/Deep_learning_beginer/cifar10/cifar-10-batches-py",
                        help = "Root of path dataset")
    parser.add_argument("--image_size", type = int, default = 224, help = "Image size"),
    parser.add_argument("--logging", "-l",type = str, default = "tensorboard"),
    parser.add_argument("--trained_model", "-t",type = str, default = "trained_model")
    args = parser.parse_args([])
    return args

if __name__ == "__main__":
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    root = args.root
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor()
    ])
    train_dataset = MyDataset(root, train = True, transform = transform)
    training_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = 0
    )
    test_dataset = MyDataset(root, train = False, transform = transform)
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        drop_last = False,
        num_workers = 0
    )

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)

    if not os.path.isdir(args.trained_model):
        os.mkdir(args.trained_model)
    writer = SummaryWriter(args.logging)
    model = SimpleCNN(num_classes = 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    num_epochs = args.epochs
    num_iterations = len(training_dataloader)

    if torch.cuda.is_available():
        model.cuda()

    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm.tqdm(training_dataloader, colour = "Green")
        for iter, (images, labels) in enumerate(progress_bar):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            # Forward
            output = model(images)
            loss_values = criterion(output, labels)

            progress_bar.set_description("epoch {}/{}, iter {}/{}, loss: {:.3f}".format(epoch + 1, num_epochs, iter + 1, len(training_dataloader), loss_values))
            writer.add_scalar("Train/loss", loss_values, epoch * num_iterations + iter)

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
    accuracy = accuracy_score(all_labels, all_predictions)
    writer.add_scalar("Val/Accuracy ", accuracy, epoch + 1)
    torch.save(model.state_dict(), "{}/last_cnn.pt".format(args.trained_model))
    # print(classification_report(all_labels, all_predictions))

    if accuracy > best_acc:
        torch.save(model.state_dict(), "{}/best_cnn.pt".format(args.trained_model))
        best_acc = accuracy
