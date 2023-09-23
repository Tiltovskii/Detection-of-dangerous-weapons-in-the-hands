import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import matplotlib.patches as patches
from detecto.utils import _is_iterable
import os
from glob import glob
import xml.etree.ElementTree as ET
import pandas as pd
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm


def reverse_normalize(image: torch.Tensor) -> torch.Tensor:
    reverse = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                   std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    return reverse(image)


def show_labeled_image(ax: plt.Axes, image: torch.Tensor, boxes: torch.Tensor, labels: list = None):
    if isinstance(image, torch.Tensor):
        image = reverse_normalize(image)
        image = transforms.ToPILImage()(image)

    if boxes.ndim == 1:
        boxes = boxes.view(1, 4)

    if labels is not None and not _is_iterable(labels):
        labels = [labels]

    for i in range(boxes.shape[0]):
        box = boxes[i]
        width, height = (box[2] - box[0]).item(), (box[3] - box[1]).item()
        initial_pos = (box[0].item(), box[1].item())
        rect = patches.Rectangle(initial_pos, width, height, linewidth=1,
                                 edgecolor='r', facecolor='none')
        if labels:
            ax.text(box[0] + 5, box[1] - 5, '{}'.format(labels[i]), color='red')

        ax.add_patch(rect)
    ax.imshow(image)


def xml_to_csv(label_dir: str, image_dir: str, output_file: str = None) -> pd.DataFrame:
    names_of_images = [f for f in sorted(os.listdir(image_dir))]
    xml_list = []
    image_id = 0
    # Loop through every XML file
    for xml_file in glob(label_dir + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        if filename not in names_of_images:
            continue

        # Each object represents each actual image label
        for member in root.findall('object'):
            box = member.find('bndbox')
            label = member.find('name').text

            # Add image file name, image size, label, and box coordinates to CSV file
            row = (filename, width, height, label, int(float(box.find('xmin').text)),
                   int(float(box.find('ymin').text)), int(float(box.find('xmax').text)),
                   int(float(box.find('ymax').text)), image_id)
            xml_list.append(row)

        image_id += 1

    # Save as a CSV file
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'image_id']
    xml_df = pd.DataFrame(xml_list, columns=column_names)

    if output_file is not None:
        xml_df.to_csv(output_file, index=None)

    return xml_df


def MaP(model, dataloader):
    metric = MeanAveragePrecision()
    iterable = tqdm(dataloader, position=0, leave=True)
    with torch.no_grad():
        for images, targets in iterable:
            model._convert_to_int_labels(targets)
            images, targets = model._to_device(images, targets)
            predicts = model.predict_top(images)

            preds = [{'boxes': predict[1],
                      'scores': predict[2],
                      'labels': predict[0]} for predict in predicts]

            model._convert_to_int_labels(preds)
            preds, targets = model._to_cpu(preds, targets)
            metric.update(preds, targets)

    return metric.compute()
