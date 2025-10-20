from torch.utils.data import Dataset
import os
import pickle
import numpy as np
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage
import cv2
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root, train = True, transform = None):
        self.transform = transform
        if train == False:
            path_list = ["test_batch"]
        else:
            path_list = ["data_batch_{}".format(i) for i in range(1, 6)]
        self.images = []
        self.labels = []
        for path in path_list:
            file_path = os.path.join(root, path)
            with open(file_path, "rb") as fo:
                file_info = pickle.load(fo, encoding = "bytes")
                self.images.extend(file_info[b'data'])
                self.labels.extend(file_info[b'labels'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = np.reshape(image, (3, 32,32))
        image = np.transpose(image, (1, 2, 0))
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    root = "C:/Users/Admin/PycharmProjects/Deep_learning_beginer/cifar10/cifar-10-batches-py"
    transform = Compose([
        Resize((100, 100)),
        ToTensor()
    ])
    dataset = MyDataset(root, train = True, transform = transform)
    image, label = dataset[234]
    to_pil = ToPILImage()
    image = to_pil(image)
    image.show()