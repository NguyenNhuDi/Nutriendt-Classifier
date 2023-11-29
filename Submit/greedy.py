from typing import List, Tuple
import glob
from submit_util import *
import json
import argparse
import torch
import albumentations as A
import numpy as np
from tqdm import tqdm
import os

def get_sorted_performances(text_path:List[str])->List[Tuple[float, float, str]]:

    performances = []
    for path in text_path:
        txt_files = glob.glob(f'{path}/*.txt')

        for t_path in txt_files:
            
            if 'predictions' in t_path:
                continue
            curr_file = open(t_path, 'r')

            best = curr_file.readlines()[-3:-2][0].split(' ')

            loss = float(best[5][:-1])
            accuracy = float(best[8][:-1])


            performances.append((accuracy, loss, t_path))

    performances.sort(key=lambda x : x[1])

    return performances

def find_model_index(model_path:str, model_paths:List[str])->int:
    
    splited_model_path = model_path.split('/')
    model_number = splited_model_path[-1][0]
    model_name = splited_model_path[-2].split(' ')[0].lower()

    for i, path in enumerate(model_paths):
        if model_name in path and model_number in path:
            return i
    return -1


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
    text_path = args['text_path']
    choose_top_n = args['choose_top_n']

    performances = get_sorted_performances(text_path)

    _20_images, _21_images = read_test_images(_20_test, _21_test)
    
    all_predictions = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(choose_top_n):
        # Get the ith best performing model based on loss
        model_index = find_model_index(performances[i][-1], models)

        transform = A.Compose(
            transforms=[
                A.Resize(image_sizes[model_index], image_sizes[model_index]),
                A.RandomRotate90(p=1.0),
                A.Lambda(image=lambda_transform),
                A.Normalize(mean=means[model_index], std=stds[model_index], max_pixel_value=1.0)
            ],
            p=1.0
        )

        model = torch.load(models[model_index])
        model.to(device)

        
        transformed_images_and_name = transform_images(transform=transform, img_arr=_20_images)
        transformed_images_and_name += transform_images(transform=transform, img_arr=_21_images)

        batched_imgs, batched_name = batch_images(batch_size=batch_sizes[i], transformed_imgs=transformed_images_and_name)
        print(f'\n\n---iteration: {i} --- model: {model_index}---\n')

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