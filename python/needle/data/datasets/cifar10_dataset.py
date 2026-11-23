import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)

        if train:
            batch_files = ["data_batch_%d" % i for i in range(1, 6)]
        else:
            batch_files = ["test_batch"]

        images, labels = [], []
        for batch_file in batch_files:
            with open(os.path.join(base_folder, batch_file), "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            images.append(batch[b"data"])
            labels.append(batch[b"labels"])

        self.X = np.concatenate(images).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.y = np.concatenate(labels)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        img = self.X[index]
        label = self.y[index]

        if np.isscalar(index):
            img = self.apply_transforms(img)

        elif self.transforms is not None:
            img = np.stack(
                [self.apply_transforms(x) for x in img]
            )

        return img, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
