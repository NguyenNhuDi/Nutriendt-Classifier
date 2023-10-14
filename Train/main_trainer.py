import os
import yaml
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
import torch
from tqdm import tqdm
import torch.nn as nn
import time

import warnings

warnings.filterwarnings("ignore")

csv_paths = [
    r'C:\Users\coanh\Desktop\Uni Work\Nutrient Classifier\Nutriend-Classifier\Labels_and_csvs\april\0_4.csv',
    r'C:\Users\coanh\Desktop\Uni Work\Nutrient Classifier\Nutriend-Classifier\Labels_and_csvs\march\0_3.csv',
    r'C:\Users\coanh\Desktop\Uni Work\Nutrient Classifier\Nutriend-Classifier\Labels_and_csvs\may\0_5.csv'
]
label_path = r'C:\Users\coanh\Desktop\Uni Work\Nutrient Classifier\Nutriend-Classifier\Labels_and_csvs\labels.yml'
image_paths = [
    r'C:\Users\coanh\Desktop\Uni Work\Nutrient Classifier\Nutriend-Classifier\DND-Diko-WWWR\WR2021\images',
    r'C:\Users\coanh\Desktop\Uni Work\Nutrient Classifier\Nutriend-Classifier\DND-Diko-WWWR\WW2020\images'
]

model_save_dir = r'C:\Users\coanh\Desktop\Uni Work\Nutrient Classifier\Nutriend-Classifier\Models'

batch_size = 32
num_workers = 7

epochs = 50
criterion = nn.CrossEntropyLoss()

momentum = [0.9]
learning_rate = [0.01]
weight_decay = [0]
gamma = [0.75]

epoch_step = 10
image_size = 224

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = 'res_net50'


def evaluate(val_batches, model):
    model.eval()
    total_correct = 0
    total_loss = 0
    total = 0
    for data in val_batches:
        image, label = data
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(image)
            loss = criterion(outputs, label)

            total_loss += loss.item() * image.size(0)
            total += image.size(0)
            _, prediction = outputs.max(1)

            total_correct += (label == prediction).sum()

        loss = total_loss / total
        accuracy = total_correct / total
        # print(f'Evaluate --- Epoch: {epoch}, Loss: {loss:6.8f}, Accuracy: {accuracy:6.8f}')

        return loss, accuracy


def train_model(model, val_batches, train_batches, es, g, lr, m, wd):
    model = model.double()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=m, weight_decay=wd)

    total = 0
    total_correct = 0
    total_loss = 0

    best_loss = 10000
    best_accuracy = -1
    best_epoch = 0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, es, g)

    for epoch in range(epochs):
        start = time.time()

        for data in tqdm(train_batches):
            image, label = data
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()

            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total += image.size(0)
            _, predictions = outputs.max(1)
            total_correct += (predictions == label).sum()

            total_loss += loss.item() * image.size(0)

        time_per_epoch = time.time() - start
        eval_loss, eval_accuracy = evaluate(val_batches, model)

        model.train()

        train_total_loss = total_loss / total
        train_accuracy = total_correct / total

        print(f'--- Epoch time: {time_per_epoch / 60.0:6.8f} minutes ---\n'
              f'Train Accuracy: {train_accuracy:6.8f} --- Train Loss: {train_total_loss:6.8f}\n'
              f'Eval Accuracy: {eval_accuracy:6.8f} --- Eval Loss: {eval_loss:6.8f}')

        if eval_accuracy >= best_accuracy:
            best_accuracy = eval_accuracy
            best_epoch = epoch
            torch.save(model, model_save_dir)
            best_loss = eval_loss if eval_loss <= best_loss else best_loss
        print(f'Best Accuracy: {best_accuracy:6.8f} --- Best Loss: {best_loss:6.8f}\n'
              f'Current Epoch: {epoch} --- Best Epoch: {best_epoch}')

        scheduler.step()


class PlantDataset(Dataset):
    def __init__(self, img_dirs, yml_label, csv_dirs, transform: A.Compose = None, train=True, mean=None, std=None):
        self.img_dirs = img_dirs
        self.yml_labels = yml_label
        self.images = []

        for csv_dir in csv_dirs:
            data = pd.read_csv(csv_dir)

            if train:
                self.images += data['train'].values.tolist()
            else:
                self.temp = data['val'].values.tolist()
                out = None
                for index, item in enumerate(self.temp):
                    try:
                        int(item)
                    except ValueError:
                        out = index
                        break
                self.images += self.temp[: out]

        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]

        try:
            out_image = np.array(Image.open(os.path.join(self.img_dirs[0], image_name)))
        except FileNotFoundError:
            out_image = np.array(Image.open(os.path.join(self.img_dirs[1], image_name)))

        if self.transform is not None:
            augmenter = self.transform(image=out_image)
            out_image = augmenter['image']

            if self.mean is not None and self.std is not None:
                mean_std_transformer = A.Compose(
                    transforms=[
                        A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0)
                    ],
                    p=1
                )

                mean_Std_augmenter = mean_std_transformer(image=out_image)
                out_image = mean_Std_augmenter['image']

        out_label = self.yml_labels[image_name]
        if out_label == 'unfertilized':
            out_label = 0
        elif out_label == '_PKCa':
            out_label = 1
        elif out_label == 'N_KCa':
            out_label = 2
        elif out_label == 'NP_Ca':
            out_label = 3
        elif out_label == 'NPK_':
            out_label = 4
        elif out_label == 'NPKCa':
            out_label = 5
        else:
            out_label = 6

        out_image = torch.from_numpy(out_image).permute(2, 0, 1)
        out_image = out_image.double()
        out_label = torch.tensor(out_label).long()

        return out_image, out_label


if __name__ == '__main__':
    with open(label_path, 'r') as f:
        labels = yaml.safe_load(f)

    transform = A.Compose(
        transforms=[
            A.Resize(224, 224)
        ]
    )

    test_set = PlantDataset(img_dirs=image_paths, yml_label=labels, csv_dirs=csv_paths, transform=transform)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    for i in tqdm(test_loader):
        print(i)
