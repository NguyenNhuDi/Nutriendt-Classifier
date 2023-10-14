import os
import yaml
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
import torch
from tqdm import tqdm

csv_path = r'C:\Users\coanh\Desktop\Uni Work\Nutrient Classifier\Nutriend-Classifier\Labels_and_csvs\april\0_4.csv'
label_path = r'C:\Users\coanh\Desktop\Uni Work\Nutrient Classifier\Nutriend-Classifier\Labels_and_csvs\labels.yml'
image_paths = [
    r'C:\Users\coanh\Desktop\Uni Work\Nutrient Classifier\Nutriend-Classifier\DND-Diko-WWWR\WR2021\images',
    r'C:\Users\coanh\Desktop\Uni Work\Nutrient Classifier\Nutriend-Classifier\DND-Diko-WWWR\WW2020\images'
]


class PlantDataset(Dataset):
    def __init__(self, img_dirs, yml_label, csv_dir, transform:A.compose=None, train=True, mean=None, std=None):
        self.img_dirs = img_dirs
        self.yml_labels = yml_label
        data = pd.read_csv(csv_dir)

        if train:
            self.images = data['train'].values.tolist()
        else:
            self.images = data['val'].values.tolist()
            out = None
            for index, item in enumerate(self.images):
                try:
                    int(item)
                except ValueError:
                    out = index
                    break
            self.images = self.images[: out]

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

    test_set = PlantDataset(img_dirs=image_paths, yml_label=labels, csv_dir=csv_path)

    test_set.__getitem__(2)
