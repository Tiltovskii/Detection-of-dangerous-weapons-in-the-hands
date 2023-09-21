import os
import torch
import pandas as pd
from detecto.utils import read_image


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, label_data, image_folder=None, transforms=None):

        self._csv = pd.read_csv(label_data)

        # If image folder not given, set it to labels folder
        if image_folder is None:
            self._root_dir = label_data
        else:
            self._root_dir = image_folder

        self.transforms = transforms

    # Returns the length of this dataset
    def __len__(self):
        # number of entries == number of unique image_ids in csv.
        return len(self._csv['image_id'].unique().tolist())

    # Is what allows you to index the dataset, e.g. dataset[0]
    # dataset[index] returns a tuple containing the image and the targets dict
    def __getitem__(self, idx: torch.Tensor):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read in the image from the file name in the 0th column
        object_entries = self._csv.loc[self._csv['image_id'] == idx]

        img_name = os.path.join(self._root_dir, object_entries.iloc[0, 0])
        image = read_image(img_name)

        boxes = []
        labels = []
        for object_idx, row in object_entries.iterrows():
            # Read in xmin, ymin, xmax, and ymax
            box = self._csv.iloc[object_idx, 4:8]
            boxes.append(box)
            # Read in the labe
            label = self._csv.iloc[object_idx, 3]
            labels.append(label)

        boxes = torch.tensor(boxes).view(-1, 4)

        targets = {'boxes': boxes, 'labels': labels}

        # Perform transformations
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=targets['boxes'], class_labels=targets['labels'])
            image = transformed['image']
            targets['boxes'] = torch.tensor(transformed['bboxes'])
            targets['labels'] = transformed['class_labels']

        return image, targets
