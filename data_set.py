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

if __name__ == "__main__":
    data = MyDataset(root = "C:/Users/Admin/PycharmProjects/Deep_learning_beginer/cifar10/cifar-10-batches-py", train = True)
    print(data.__len__())
    image, label = data.__getitem__(234)
    image = Image.fromarray(image)
    image.show()