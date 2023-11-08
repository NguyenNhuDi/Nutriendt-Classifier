import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import torch
import albumentations as A

def read_test_images(_20_test_dir, _21_test_dir):
    _20_test = open(_20_test_dir, 'r')
    _21_test = open(_21_test_dir, 'r')

    print(f'--- Opening test images ---')

    _20_name = [line[:-1] if line[-1] == '\n' else line for line in _20_test]
    _21_name = [line[:-1] if line[-1] == '\n' else line for line in _21_test]

    _20_out = [(np.array(Image.open(os.path.join(os.path.dirname(_20_test_dir), f'images/{image_name}'))), image_name) for image_name in tqdm(_20_name)]
    _21_out = [(np.array(Image.open(os.path.join(os.path.dirname(_21_test_dir), f'images/{image_name}'))), image_name) for image_name in tqdm(_21_name)]

    return _20_out, _21_out


def lambda_transform(x: torch.Tensor, **kwargs) -> torch.Tensor:
    return x / 255


def transform_images(transform, img_arr):
    out = []
    for data in img_arr:
        img, image_name = data
        augment = transform(image=img)
        out_img = augment['image']

        out_image = torch.from_numpy(out_img).permute(2,0, 1)
        out.append((out_image, image_name))

    return out

def batch_images(batch_size, transformed_imgs):
    out_imgs = []
    out_names = []

    temp_imgs = []
    temp_names = []

    batch_counter = 0
    for i in range(len(transformed_imgs)):
        image, image_name = transformed_imgs[i]

        temp_imgs.append(image)
        temp_names.append(image_name)

        batch_counter += 1

        if batch_counter == batch_size:
            temp_imgs = torch.stack(temp_imgs, dim=0)
            out_imgs.append(temp_imgs)
            out_names.append(temp_names)
            temp_imgs = []
            temp_names = []
            batch_counter = 0
    
    if len(temp_imgs) > 0:
            temp_imgs = torch.stack(temp_imgs, dim=0)
            out_imgs.append(temp_imgs)
            out_names.append(temp_names)


    return out_imgs, out_names


if __name__ == '__main__':
    _20_dir = r'/home/nhu.nguyen2/Nutrient_Classifier/Nutriendt-Classifier/DND-Diko-WWWR/WW2020/test.txt'
    _21_dir = r'/home/nhu.nguyen2/Nutrient_Classifier/Nutriendt-Classifier/DND-Diko-WWWR/WR2021/test.txt'

    _20_images, _21_images = read_test_images(_20_dir, _21_dir)

    transformer = A.Compose(
        transforms=[
            A.Resize(224, 224),
            A.Lambda(image=lambda_transform)
        ],
        p=1.0
    )

    _20_transformed = transform_images(transformer, _20_images)

    batched_images, batched_names = batch_images(10, _20_transformed)

    for i in batched_images:
        print(i.shape)