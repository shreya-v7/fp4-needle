from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip, struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)

        with gzip.open(image_filename, "rb") as f:
            magic, num_images, rows, cols = struct.unpack(
                ">IIII", f.read(16)
            )

            self.images = np.frombuffer(
                f.read(),
                dtype=np.uint8,
            ).reshape(num_images, rows, cols, 1)

            self.images = self.images.astype(np.float32) / 255.0

        with gzip.open(label_filename, "rb") as f:
            magic, num_labels = struct.unpack(
                ">II", f.read(8)
            )

            self.labels = np.frombuffer(
                f.read(),
                dtype=np.uint8,
            )
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.images[index]
        label = self.labels[index]

        if np.isscalar(index):
            img = self.apply_transforms(img)

        elif self.transforms is not None:
            img = np.stack(
                [self.apply_transforms(x) for x in img]
            )

        return img, label
        
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION