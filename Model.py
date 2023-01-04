import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output
from detecto.core import DataLoader, Model


class NewModel(Model):

    DEFAULT = 'fasterrcnn_resnet50_fpn'
    MOBILENET = 'fasterrcnn_mobilenet_v3_large_fpn'
    MOBILENET_320 = 'fasterrcnn_mobilenet_v3_large_320_fpn'

    def __init__(self, classes=None, device=None, pretrained=True, model_name=DEFAULT):
        super().__init__(classes, device, pretrained, model_name)

    def fit(self, dataset, optimizer, lr_scheduler, val_dataset=None, epochs=10, verbose=True):

        if verbose and self._device == torch.device('cpu'):
            print('It looks like you\'re training your model on a CPU. '
                  'Consider switching to a GPU; otherwise, this method '
                  'can take hours upon hours or even days to finish. '
                  'For more information, see https://detecto.readthedocs.io/'
                  'en/latest/usage/quickstart.html#technical-requirements')

        if epochs > 0:
            self._disable_normalize = False

        # Convert dataset to data loader if not already
        if not isinstance(dataset, DataLoader):
            dataset = DataLoader(dataset, shuffle=True)

        if val_dataset is not None and not isinstance(val_dataset, DataLoader):
            val_dataset = DataLoader(val_dataset)

        train_losses = []
        val_losses = []

        # Train on the entire dataset for the specified number of times (epochs)
        for epoch in range(epochs):
            if verbose:
                print('Epoch {} of {}'.format(epoch + 1, epochs))

            # Training step
            self._model.train()

            if verbose:
                print('Begin iterating over training dataset')

            iterable = tqdm(dataset, position=0, leave=True) if verbose else dataset
            batch_train_loss = []
            for images, targets in iterable:
                self._convert_to_int_labels(targets)
                images, targets = self._to_device(images, targets)

                # Calculate the model's loss (i.e. how well it does on the current
                # image and target, with a lower loss being better)
                loss_dict = self._model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())
                batch_train_loss += [total_loss.item()]

                # Zero any old/existing gradients on the model's parameters
                optimizer.zero_grad()
                # Compute gradients for each parameter based on the current loss calculation
                total_loss.backward()
                # Update model parameters from gradients: param -= learning_rate * param.grad
                optimizer.step()

            avg_train_loss = np.mean(batch_train_loss)
            train_losses += [avg_train_loss]
            # Validation step
            if val_dataset is not None:
                avg_val_loss = 0
                total_loss = 0
                with torch.no_grad():
                    if verbose:
                        print('Begin iterating over validation dataset')

                    iterable = tqdm(val_dataset, position=0, leave=True) if verbose else val_dataset
                    for images, targets in iterable:
                        self._convert_to_int_labels(targets)
                        images, targets = self._to_device(images, targets)
                        loss_dict = self._model(images, targets)
                        total_loss = sum(loss for loss in loss_dict.values())
                        avg_val_loss += total_loss.item()

                avg_val_loss /= len(val_dataset.dataset)
                val_losses += [avg_val_loss]

            if verbose:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
                clear_output(True)
                ax[0].plot(batch_train_loss, label='train loss')
                ax[0].set_xlabel('Batch')
                ax[0].set_title('Train loss')
                if train_losses is not None:
                    ax[1].plot(train_losses, label='general train history')
                    ax[1].set_xlabel('Epoch')
                if val_losses is not None and val_dataset is not None:
                    ax[1].plot(val_losses, label='general test history')
                plt.legend()
                plt.show()

                if val_dataset is not None:
                    print(f"Train Loss: {avg_train_loss:.4f}, "
                          f"Test Loss: {avg_val_loss:.4f}, "
                          f"Number of epoch: {epoch + 1}")
                else:
                    print(f"Train Loss: {avg_train_loss:.4f}, "
                          f"Number of epoch: {epoch + 1}")

            # Update the learning rate every few epochs
            lr_scheduler.step()

        if len(train_losses) > 0:
            return train_losses

    def _to_cpu(self, preds, targets):
        preds = [{k: v.to(self._device) for k, v in t.items()} for t in preds]
        targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]
        return preds, targets

    @staticmethod
    def load(file, classes, model_name):
        model = NewModel(classes, model_name=model_name)
        model._model.load_state_dict(torch.load(file, map_location=model._device))
        return model