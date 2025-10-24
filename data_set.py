from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Resize
import os
import pickle
import numpy as np
import cv2
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root, train = True, transform = None):
        self.transform = transform
        super().__init__()
        data_path = []
        if train:
            data_path = ["data_batch_{}".format(i) for i in range(1, 6)]
        else:
            data_path = ["test_batch"]
        self.images = []
        self.labels = []
        for path in data_path:
            file_path = os.path.join(root, path)
            with open(file_path, "rb") as fo:
                data = pickle.load(fo, encoding = "bytes")
                self.images.extend(data[b"data"])
                self.labels.extend(data[b"labels"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = np.reshape(image, (3, 32, 32))
        image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# if __name__ == "__main__":
#     root = "C:/Users/Admin/PycharmProjects/Deep_learning_beginer/cifar10/cifar-10-batches-py"
#     transform = Compose([
#         Resize((100, 100))
#     ])
#     dataset = MyDataset(root, train = True, transform = transform)
#     image, label = dataset[0]
