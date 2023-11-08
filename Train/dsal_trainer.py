from DSAL import DSAL
from train_utils import find_mean_std, transform_image_label, train, lambda_transform
import os
import yaml
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
    _21_dir = args['21_dir']
    _20_dir = args['20_dir']
    csv_paths = args['csv_paths']
    batch_size = args['batch_size']
    num_workers = args['num_workers']
    epochs = args['epochs']
    run_id = args['run_id']
    out_name = args['out_name']
    momentum = args['momentum']
    learning_rate = args['learning_rate']
    weight_decay = args['weight_decay']
    gamma = args['gamma']
    model_name = args['model_name']
    model_save_dir = args['model_save_dir']
    epoch_step = args['epoch_step']



    with open(label_path, 'r') as f:
        labels = yaml.safe_load(f)

    print(f'---- Opening Images ----')

    all_imgs = {}

    for img_name in tqdm(labels):
        image_path = os.path.join(_21_dir, img_name) if img_name[3] == '1' else os.path.join(_20_dir, img_name)
        image = np.array(Image.open(image_path))
        all_imgs[img_name] = image



    out_json = {
        "20_test": "/home/nhu.nguyen2/Nutrient_Classifier/Nutriendt-Classifier/DND-Diko-WWWR/WW2020/test.txt",
        "21_test": "/home/nhu.nguyen2/Nutrient_Classifier/Nutriendt-Classifier/DND-Diko-WWWR/WR2021/test.txt",
        "save_dir": "",
        "models": [],
        "image_size": [],
        "mean": [],
        "std": [],
        "batch_size": [],
        "run_amount": 5
    }


    for i, output_file_name in enumerate(out_name):

        val_paths = []
        train_paths = []

        for csv_dir in csv_paths[i]:
            data = pd.read_csv(csv_dir)

            t = data['val'].values.tolist()

            out = None
            for j, item in enumerate(t):
                if type(item) is float:
                    out = j
                    break

            val_paths += t[: out]

            train_paths += data['train'].values.tolist()



        train_imgs = [(all_imgs[image_name], image_name) for image_name in train_paths]
        val_imgs = [(all_imgs[image_name], image_name) for image_name in val_paths]

        train_transform = A.Compose(
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


        val_transform = A.Compose(
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
                A.Lambda(image=lambda_transform)

            ],
            p=1.0
        )


        mean_std_dsal = DSAL(images=train_imgs,
                            read_and_transform_function=transform_image_label,
                            batch_size=batch_size,
                            epochs=1,
                            num_processes=num_workers,
                            yml=labels,
                            transform=train_transform)

        mean_std_dsal.start()

        mean, std = find_mean_std(mean_std_dsal)

        mean_std_dsal.join()

        
        out_json["mean"].append(mean)
        out_json["std"].append(std)
        out_json["image_size"].append(image_size)
        out_json["batch_size"].append(batch_size)
        out_json["models"].append(str(os.path.join(model_save_dir, f'{run_id[i]}_{model_name}_best.pth')))

        val_dsal = DSAL(images=val_imgs,
                        read_and_transform_function=transform_image_label,
                        batch_size=batch_size,
                        epochs=1,
                        num_processes=num_workers,
                        yml=labels,
                        transform=val_transform,
                        mean=mean,
                        std=std)
        
        val_dsal.start()

        val_batches = [val_dsal.get_item() for i in range(val_dsal.num_batches)]

        val_dsal.join()


        print(f'\n\n\n------ start training ------\n\n')

        criterion = nn.CrossEntropyLoss()

        model_chooser = ModelChooser(model_name=model_name)
        model = model_chooser()
        model.to(device)


        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate[i], momentum=momentum[i],
                                    weight_decay=weight_decay[i])
        
        message = f'\n{mean}, {std}\n--- gamma: {gamma[i]} --- momentum: {momentum[i]} --- learning rate: {learning_rate[i]} --- weight_decay: {weight_decay[i]}\n'
        print(message)

        train(model=model,
              criterion=criterion,
              optimizer=optimizer,
              val_batches=val_batches,
              train_images=train_imgs,
              train_transform=train_transform,
              mean=mean,
              std=std,
              labels=labels,
              out_text_name=f'{output_file_name}_.txt',
              model_save_dir=os.path.join(model_save_dir, f'{run_id[i]}_{model_name}_best.pth'),
              last_save_name=os.path.join(model_save_dir, f'{run_id[i]}_{model_name}_last.pth'),
              num_workers=num_workers,
              batch_size=batch_size,
              epochs=epochs,
              epoch_step=epoch_step,
              gamma=gamma[i],
              device=device,
              input_message=message)
    
    with open(f'{model_save_dir}/{model_name}.json', 'w') as json_file:
        json.dump(out_json, json_file, indent=4)
