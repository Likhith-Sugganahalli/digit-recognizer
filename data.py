import os
import numpy as np
import pandas as pd
import imageio as imgio
import matplotlib.pyplot as plt

import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms, utils


def load_image(image_col_list, shape_tuple=28):
    if isinstance(shape_tuple, int):
        shape_tuple = (shape_tuple, shape_tuple)

    img_list = image_col_list.split(" ")
    temp = np.asarray(img_list)
    temp.resize(shape_tuple)
    temp = temp.astype(np.uint8)

    return temp


class digitRecognizerDataset(Dataset):
    """Face Keypoints dataset."""

    def __init__(self, csv_file, root_dir="./", transform: transforms.Compose = None):
        """
        Arguments:
            csv_file (string): Path to the csv file with labels and images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset_csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = loadImage(self.landmarks_frame.iloc[idx, 0:])

        label = self.landmarks_frame.iloc[idx, 0]

        print("label", label)

        # sample = {"image": image, "keypoints": keypoints}

        # if self.transform:
        #     # for transform in self.transform:
        #     sample = self.transform(sample)

        # return sample


def main():
    composed = None  # transforms.Compose([RandomCrop(90)])

    face_dataset = digitRecognizerDataset(
        csv_file="./data/train.csv", transform=composed
    )

    # for i in range(len(face_dataset)):
    #     sample = face_dataset[i]

    #     plt.imshow(sample["image"])
    #     plt.scatter(
    #         sample["keypoints"][:, 0],
    #         sample["keypoints"][:, 1],
    #         s=10,
    #         marker=".",
    #         c="r",
    #     )
    #     plt.show()

    #     if i == 10:
    #         break


if __name__ == "__main__":
    main()
