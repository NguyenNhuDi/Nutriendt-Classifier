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
import argparse
import json
from torchvision import models
import sys

import warnings

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()


def lambda_transform(x: np.array, **kwargs) -> np.array:
    return x / 255


def find_mean_std(train_set):
    sum_ = torch.zeros(3)
    sq_sum = torch.zeros(3)
    num_images = 0

    print(f'---finding mean and std ---')

    for data in tqdm(train_set):
        image = data[0]
        batch = image.size(0)
        sum_ += torch.mean(image, dim=[0, 2, 3]) * batch
        sq_sum += torch.mean(image ** 2, dim=[0, 2, 3]) * batch
        num_images += batch

    mean = sum_ / num_images
    std = ((sq_sum / num_images) - mean ** 2) ** 0.5

    print(f'mean: {mean} --- std: {std}')
    message = f'mean: {mean} --- std: {std}'f'mean: {mean} --- std: {std}'
    return mean, std, message


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
        return loss, accuracy


def train_model(model, val_batches, train_batches, es, g, lr, m, wd, run_name, std_mean_vals,
                model_save_path, model_name, model_save_dir, epochs, out_name):
    model = model.double()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=m, weight_decay=wd)

    output_file = open(os.path.join(model_save_dir, f'{out_name}.txt'), 'w')

    output_file.write(f'{std_mean_vals}\n')
    output_file.write(f'momentum: {m} --- learning rate: {lr} --- weight decay: {wd} --- gamma: {g}\n')
    print((f'momentum: {m} --- learning rate: {lr} --- weight decay: {wd} --- gamma: {g}'))

    total = 0
    total_correct = 0
    total_loss = 0

    best_loss = 10000
    best_accuracy = -1
    best_epoch = 0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, es, g)

    for epoch in range(epochs):
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

        eval_loss, eval_accuracy = evaluate(val_batches, model)

        model.train()

        train_total_loss = total_loss / total
        train_accuracy = total_correct / total

        message = f'Train Accuracy: {train_accuracy:6.8f} --- Train Loss: {train_total_loss:6.8f}\n' \
                  f'Eval Accuracy: {eval_accuracy:6.8f} --- Eval Loss: {eval_loss:6.8f}\n'

        print(message, end='')
        output_file.write(message)

        if eval_accuracy >= best_accuracy:
            best_accuracy = eval_accuracy
            best_epoch = epoch
            torch.save(model, os.path.join(model_save_path, f'{model_name}_{run_name}.pt'))
            best_loss = eval_loss if eval_loss <= best_loss else best_loss
        
        message = f'Best Accuracy: {best_accuracy:6.8f} --- Best Loss: {best_loss:6.8f}\n' \
                  f'Current Epoch: {epoch} --- Best Epoch: {best_epoch}\n'

        print(message, end='')
        output_file.write(message)

        scheduler.step()
    output_file.close()
    torch.save(model, os.path.join(model_save_path, f'{model_name}_{run_name}_final.pt'))


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
                for i, item in enumerate(self.temp):
                    if type(item) is float:
                        out = i
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
            out_image = np.array(Image.open(os.path.join(self.img_dirs[0], f'{image_name}')))
        except FileNotFoundError:
            out_image = np.array(Image.open(os.path.join(self.img_dirs[1], f'{image_name}')))

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


class ModelChooser:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self):
        return self.__choose_model__()

    def __choose_model__(self):
        model = None

        if self.model_name == 'efficientnet_b6':
            model = models.efficientnet_b6(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(in_features=2304, out_features=256),
                nn.Linear(in_features=256, out_features=7)
            )
        elif self.model_name == 'efficientnet_v2_m':
            model = models.efficientnet_v2_m(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features=1280, out_features=1000),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.model_name == 'resnet152':
            model = models.resnet152(pretrained=True)

            model.fc = nn.Sequential(
                nn.Linear(in_features=2048, out_features=256),
                nn.Linear(in_features=256, out_features=7)
            )
        elif self.model_name == 'resnext101_32x8d':
            model = models.resnext101_32x8d(pretrained=True)

            model.fc = nn.Sequential(
                nn.Linear(in_features=2048, out_features=256),
                nn.Linear(in_features=256, out_features=7)
            )
        else:
            sys.exit(f'Model: {self.model_name} is not part of the registered models')

        return model


if __name__ == '__main__':
    print(f'Begin Code....')

    parser = argparse.ArgumentParser(
        prog='Model Trainer',
        description='This program will train a model',
        epilog='Vision Research Lab')
    parser.add_argument('-c', '--config', required=True,
                        help='The path to the config file.')
    args = parser.parse_args()

    with open(args.config) as f:
        args = json.load(f)

    label_path = args['label_path']
    image_size = args['image_size']
    image_paths = args['image_paths']
    csv_paths = args['csv_paths']
    batch_size = args['batch_size']
    num_workers = args['num_workers']

    with open(label_path, 'r') as f:
        labels = yaml.safe_load(f)

    transform = A.Compose(
        transforms=[
            A.Resize(image_size, image_size),

            A.Flip(p=0.5),
            A.Rotate(
                limit=(-90, 90),
                interpolation=1,
                border_mode=0,
                value=0,
                mask_value=0,
                always_apply=False,
                p=0.75,
            ),

            A.OneOf(
                transforms=[
                    A.Defocus(radius=[1, 1], alias_blur=(0.1, 0.3), p=0.2),
                    A.Sharpen(alpha=(0.01, 0.125), lightness=(1, 1), p=0.2),
                    A.RGBShift(r_shift_limit=[-5, 5], g_shift_limit=[-3, 3], b_shift_limit=[-5, 5], p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=[-0.015, 0.015], contrast_limit=[0, 0], p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=[0.0, 0.0], contrast_limit=[-0.015, 0.015], p=0.2),
                ],
                p=0.2,
            ),
            A.OneOf(
                transforms=[
                    A.Defocus(radius=[1, 1], alias_blur=(0.1, 0.3), p=0.2),
                    A.Sharpen(alpha=(0.01, 0.125), lightness=(1, 1), p=0.2),
                    A.RGBShift(r_shift_limit=[-5, 5], g_shift_limit=[-3, 3], b_shift_limit=[-5, 5], p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=[-0.015, 0.015], contrast_limit=[0, 0], p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=[0.0, 0.0], contrast_limit=[-0.015, 0.015], p=0.2),
                ],
                p=0.2,
            ),
            A.OneOf(
                transforms=[
                    A.Defocus(radius=[1, 1], alias_blur=(0.1, 0.3), p=0.2),
                    A.Sharpen(alpha=(0.01, 0.125), lightness=(1, 1), p=0.2),
                    A.RGBShift(r_shift_limit=[-5, 5], g_shift_limit=[-3, 3], b_shift_limit=[-5, 5], p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=[-0.015, 0.015], contrast_limit=[0, 0], p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=[0.0, 0.0], contrast_limit=[-0.015, 0.015], p=0.2),
                ],
                p=0.2,
            ),
            A.OneOf(
                transforms=[
                    A.Defocus(radius=[1, 1], alias_blur=(0.1, 0.3), p=0.2),
                    A.Sharpen(alpha=(0.01, 0.125), lightness=(1, 1), p=0.2),
                    A.RGBShift(r_shift_limit=[-5, 5], g_shift_limit=[-3, 3], b_shift_limit=[-5, 5], p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=[-0.015, 0.015], contrast_limit=[0, 0], p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=[0.0, 0.0], contrast_limit=[-0.015, 0.015], p=0.2),
                ],
                p=0.2,
            ),
            A.OneOf(
                transforms=[
                    A.Defocus(radius=[1, 1], alias_blur=(0.1, 0.3), p=0.2),
                    A.Sharpen(alpha=(0.01, 0.125), lightness=(1, 1), p=0.2),
                    A.RGBShift(r_shift_limit=[-5, 5], g_shift_limit=[-3, 3], b_shift_limit=[-5, 5], p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=[-0.015, 0.015], contrast_limit=[0, 0], p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=[0.0, 0.0], contrast_limit=[-0.015, 0.015], p=0.2),
                ],
                p=0.2,
            ),

            A.Lambda(image=lambda_transform)
        ],
        p=1.0,
    )

    init_train_set = PlantDataset(img_dirs=image_paths,
                                  yml_label=labels,
                                  csv_dirs=csv_paths,
                                  transform=transform,
                                  train=True)

    init_train_loader = DataLoader(init_train_set,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers)
    mean, std, mean_std_value = find_mean_std(init_train_loader)
    
    train_set = PlantDataset(img_dirs=image_paths,
                             yml_label=labels,
                             csv_dirs=csv_paths,
                             transform=transform,
                             train=True,
                             std=std,
                             mean=mean)

    val_set = PlantDataset(img_dirs=image_paths,
                           yml_label=labels,
                           csv_dirs=csv_paths,
                           transform=transform,
                           train=False,
                           std=std,
                           mean=mean)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)

    model_chooser = ModelChooser(args['model_name'])

    model = model_chooser()
    model.to(device)

    print(f'--- begin training ---')

    train_id = args['run_id']
    for i in range(len(train_id)):
        train_model(model=model,
                    val_batches=val_loader,
                    train_batches=train_loader,
                    es=args['epoch_step'],
                    g=args['gamma'][i],
                    wd=args['weight_decay'][i],
                    lr=args['learning_rate'][i],
                    m=args['momentum'][i],
                    run_name=train_id[i],
                    out_name=args['out_name'][i],
                    std_mean_vals=mean_std_value,
                    model_save_path=args['model_save_dir'],
                    model_name=args['model_name'],
                    model_save_dir=args['model_save_dir'],
                    epochs=args['epochs'])