import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms

CLASSES = ['pistol', 'knife', 'billete', 'monedero', 'smartphone', 'tarjeta']

TRAIN_IMAGE_DIR = 'data/Sohas_weapon-Detection/images'
TRAIN_LABEL_DIR = 'data/Sohas_weapon-Detection/annotations/xmls'
TRAIN_LABEL_CSV_DIR = 'data/Sohas_weapon-Detection/train_labels.csv'

TEST_IMAGE_DIR = 'data/Sohas_weapon-Detection/images_test'
TEST_LABEL_DIR = 'data/Sohas_weapon-Detection/annotations_test/xmls'
TEST_LABEL_CSV_DIR = 'data/Sohas_weapon-Detection/test_labels.csv'


CNN_NORMALIZATION_MEAN = torch.tensor([0.485, 0.456, 0.406])
CNN_NORMALIZATION_STD = torch.tensor([0.229, 0.224, 0.225])

TRAIN_TRANSFORMS = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=1),
        A.RandomRotate90(p=1),
        A.VerticalFlip(p=1)
    ], p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1),
        A.ToGray(p=1)
    ], p=0.3),
    A.OneOf([
        A.Blur(blur_limit=(7, 11), p=1),
        A.MedianBlur(blur_limit=(7, 11), p=1)
    ], p=0.2),
    A.Normalize(mean=CNN_NORMALIZATION_MEAN, std=CNN_NORMALIZATION_STD),
    ToTensorV2()],
    bbox_params=A.BboxParams(format='pascal_voc',
                             label_fields=['class_labels']))

TEST_TRANSFORMS = A.Compose([
    A.Normalize(mean=CNN_NORMALIZATION_MEAN, std=CNN_NORMALIZATION_STD),
    ToTensorV2()],
    bbox_params=A.BboxParams(format='pascal_voc',
                             label_fields=['class_labels']))


TRANSFORM_PROD = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CNN_NORMALIZATION_MEAN, std=CNN_NORMALIZATION_STD)
])

PARAMS = {'LEARNING_RATE': 5e-3,
          'LR_STEP_SIZE': 5,
          'GAMMA': 0.1,
          'WEIGHT_DECAY': 5e-4,
          'MOMENTUM': 0.9,
          'EPOCH': 10}

translate = {'knife': 'Knife',
             'smartphone': 'Smartphone',
             'billete': 'Banknote',
             'monedero': 'Wallet',
             'pistol': 'Pistol',
             'tarjeta': 'Bank card'}
url = "https://drive.google.com/drive/folders/11EQMBTiQr1eqCFR-tuYCcpjcBJK4BUKF"
