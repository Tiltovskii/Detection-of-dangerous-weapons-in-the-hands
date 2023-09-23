from detecto.core import DataLoader

from utils import *
from dataset import *
from model import NewModel
from config import *

if __name__ == '__main__':
    ex_csv = glob('data/Sohas_weapon-Detection/*.csv')
    if not ex_csv:
        train_df = xml_to_csv(TRAIN_LABEL_DIR, TRAIN_IMAGE_DIR, output_file=TRAIN_LABEL_CSV_DIR)
        test_df = xml_to_csv(TEST_LABEL_DIR, TEST_IMAGE_DIR, output_file=TEST_LABEL_CSV_DIR)

    train_dataset = MyDataset(TRAIN_LABEL_CSV_DIR, TRAIN_IMAGE_DIR, transforms=TRAIN_TRANSFORMS)
    val_dataset = MyDataset(TEST_LABEL_CSV_DIR, TEST_IMAGE_DIR, transforms=TEST_TRANSFORMS)

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device name: {torch.cuda.get_device_name("cuda:0")}')
    print(f'Device: {device}')

    model = NewModel(CLASSES, device=device)
    parameters = [p for p in model._model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=PARAMS['LEARNING_RATE'], momentum=PARAMS['MOMENTUM'],
                                weight_decay=PARAMS['WEIGHT_DECAY'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=PARAMS['LR_STEP_SIZE'], gamma=PARAMS['GAMMA'])

    losses = model.fit(train_loader, optimizer, lr_scheduler, val_loader, epochs=PARAMS['EPOCH'], verbose=True,
                       params=PARAMS)
