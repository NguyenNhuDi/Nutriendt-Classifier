import pandas as pd
import yaml
import os
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np

source_label_paths = [
    r'C:\Users\coanh\Desktop\Uni Work\Nutrient Classifier\Nutriend-Classifier\DND-Diko-WWWR\WW2020\labels_trainval.yml',
    r'C:\Users\coanh\Desktop\Uni Work\Nutrient Classifier\Nutriend-Classifier\DND-Diko-WWWR\WR2021\labels_trainval.yml'
]

output_path = r'C:\Users\coanh\Desktop\Uni Work\Nutrient Classifier\Nutriend-Classifier\Labels_and_csvs'
folds = 5
seed = 12345654321
val_percent = 0.2


def get_label(label):
    if label == 'unfertilized':
        return 0
    elif label == '_PKCa':
        return 1
    elif label == 'N_KCa':
        return 2
    elif label == 'NP_Ca':
        return 3
    elif label == 'NPK_':
        return 4
    elif label == 'NPKCa':
        return 5
    else:
        return 6


if __name__ == '__main__':
    out_labels = {}
    march_csvs = [
        {
            'train': [],
            'val': []
        }
        for i in range(folds)

    ]
    apr_csvs = [
        {
            'train': [],
            'val': []
        }

        for i in range(folds)
    ]
    may_csvs = [
        {
            'train': [],
            'val': []
        }
        for i in range(folds)

    ]
    classes = [[[] for i in range(7)] for j in range(3)]

    counter = 0
    for source_path in source_label_paths:
        with open(source_path, 'r') as f:
            curr_labels = yaml.safe_load(f)

        out_labels.update(curr_labels)

        for image_name in tqdm(curr_labels):
            counter += 1
            curr_class = curr_labels[image_name]
            curr_class = get_label(curr_class)

            month = int(image_name[5])

            classes[month % 3][curr_class].append(image_name)

    for month in range(3):
        kf = KFold(n_splits=folds, shuffle=True)

        for i in range(7):
            for j, (train_index, val_index) in enumerate(kf.split(classes[month][i])):

                for index in train_index:
                    if month == 0:
                        march_csvs[j]['train'].append(classes[month][i][index])
                    elif month == 1:
                        apr_csvs[j]['train'].append(classes[month][i][index])
                    else:
                        may_csvs[j]['train'].append(classes[month][i][index])

                for index in val_index:
                    if month == 0:
                        march_csvs[j]['val'].append(str(classes[month][i][index]).split('.')[0])
                    elif month == 1:
                        apr_csvs[j]['val'].append(str(classes[month][i][index]).split('.')[0])
                    else:
                        may_csvs[j]['val'].append(str(classes[month][i][index]).split('.')[0])

        for i in range(folds):
            if month == 0:
                while len(march_csvs[i]['train']) != len(march_csvs[i]['val']):
                    march_csvs[i]['val'].append(None)

                df = pd.DataFrame.from_dict(march_csvs[i])
                save_path = os.path.join(output_path, f'march')

                try:
                    os.makedirs(save_path)
                except FileExistsError:
                    pass

                df.to_csv(os.path.join(save_path, f'{i}_{3}.csv'), index=False)

            elif month == 1:
                while len(apr_csvs[i]['train']) != len(apr_csvs[i]['val']):
                    apr_csvs[i]['val'].append(None)

                df = pd.DataFrame.from_dict(apr_csvs[i])
                save_path = os.path.join(output_path, f'april')

                try:
                    os.makedirs(save_path)
                except FileExistsError:
                    pass

                df.to_csv(os.path.join(save_path, f'{i}_{4}.csv'), index=False)

            else:
                while len(may_csvs[i]['train']) != len(may_csvs[i]['val']):
                    may_csvs[i]['val'].append(None)

                df = pd.DataFrame.from_dict(may_csvs[i])
                save_path = os.path.join(output_path, f'may')

                try:
                    os.makedirs(save_path)
                except FileExistsError:
                    pass

                df.to_csv(os.path.join(save_path, f'{i}_{5}.csv'), index=False)

    with open(os.path.join(output_path, 'labels.yml'), 'w') as f:
        yaml.dump(out_labels, f)
