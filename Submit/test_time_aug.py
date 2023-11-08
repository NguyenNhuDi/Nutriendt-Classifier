from submit_util import read_test_images, lambda_transform, transform_images, batch_images
import json
import argparse
import torch
import albumentations as A
import numpy as np
from tqdm import tqdm
import os

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='Model Trainer',
        description='This program will train a model',
        epilog='Vision Research Lab')
    parser.add_argument('-c', '--config', required=True,
                        help='The path to the config file.')
    args = parser.parse_args()

    with open(args.config) as f:
        args = json.load(f)


    models = args['models']
    _20_test = args['20_test']
    _21_test = args['21_test']
    
    image_sizes = args['image_size']
    means =  args['mean']
    stds = args['std']
    save_dir = args['save_dir']
    batch_sizes = args['batch_size']
    run_amount = args['run_amount']

    _20_images, _21_images = read_test_images(_20_test, _21_test)

    all_predictions = {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for _ in range(run_amount):
        
        for i, mean in enumerate(means):
            transform = A.Compose(
                transforms=[
                    A.Resize(image_sizes[i], image_sizes[i]),
                    A.RandomRotate90(p=1.0),
                    A.Lambda(image=lambda_transform),
                    A.Normalize(mean=mean, std=stds[i], max_pixel_value=1.0)
                ],
                p=1.0
            )

            model = torch.load(models[i])
            model.to(device)

            transformed_images_and_name = transform_images(transform=transform, img_arr=_20_images)
            transformed_images_and_name += transform_images(transform=transform, img_arr=_21_images)

            batched_imgs, batched_name = batch_images(batch_size=batch_sizes[i], transformed_imgs=transformed_images_and_name)
            print(f'\n\n---iteration: {_} --- model: {i}---\n')

            for k in tqdm(range(len(batched_name))):
                image = batched_imgs[k].to(device)

                output = model(image)

                for j, image_name in enumerate(batched_name[k]):

                    if image_name not in all_predictions:
                        all_predictions[image_name] = output[j].cpu().detach().numpy()
                    else:
                        all_predictions[image_name] += output[j].cpu().detach().numpy()

        predictions_20 = []
        predictions_21 = []

        for key in all_predictions:
                curr_item = all_predictions[key]

                prediction = curr_item.argmax()

                if key[3] == '0':
                    predictions_20.append((key, prediction))
                else:
                    predictions_21.append((key, prediction))

        f = open(os.path.join(save_dir, 'predictions_WW2020.txt'), 'w')

        for i in predictions_20:
            f.write(f'{i[0]} {i[1]}\n')

        f.close()

        f = open(os.path.join(save_dir, 'predictions_WR2021.txt'), 'w')

        for i in predictions_21:
            f.write(f'{i[0]} {i[1]}\n')

        f.close()
