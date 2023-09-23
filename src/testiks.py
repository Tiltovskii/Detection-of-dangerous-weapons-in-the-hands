import torch
from model import NewModel
from detecto.core import DataLoader, Model
from detecto import core, utils, visualize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np


test_file = open("data/AuFr8SAjKGs.jpg", "rb")
print(type(test_file))

# import mlflow
#
# mlflow.set_tracking_uri("http://localhost:5000")
# classes = ['__background__'] + ['pistol', 'knife', 'billete', 'monedero', 'smartphone', 'tarjeta']
# model_name = "Kek"
# model_version = "1"
# model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")
# image = utils.read_image('data/AuFr8SAjKGs.jpg')
# print(image.shape)
# preds = model([torch.tensor(image / 255).permute(2, 0, 1).float()])
# preds = [{k: v.to(torch.device('cpu')) for k, v in p.items()} for p in preds]
# results = []
# for pred in preds:
#     # Convert predicted ints into their corresponding string labels
#     result = ([classes[val] for val in pred['labels']], pred['boxes'], pred['scores'])
#     results.append(result)
#
# print(results[0])

# model = NewModel(['pistol', 'knife', 'billete', 'monedero', 'smartphone', 'tarjeta'], device=device)
# state_dict_uri = 'file:///C:/Users/bende/PycharmProjects/Detection-of-dangerous-weapons-in-the-hands/artifacts/0/0c1aa3f05b65458588b0e2fe6748b534/artifacts/model'
# state = mlflow.pytorch.load_state_dict(state_dict_uri)
# model._model.load_state_dict(state)
# image = utils.read_image('data/AuFr8SAjKGs.jpg')
# labels, boxes, scores = model.predict_top(image)


# image = utils.read_image('data/AuFr8SAjKGs.jpg')
# model = core.Model()
#
# labels, boxes, scores = model.predict_top(image)
# print(type(labels))
# visualize.show_labeled_image(image, boxes, labels)


