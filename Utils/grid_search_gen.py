from itertools import product
import json as json
import argparse
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

    lr = args['lr']
    m = args['m']
    wd = args['wd']
    g = args['g']

    num_files = args['num_of_files']
    save_file = args['save_dir']



    out_jsons = [{
    "csv_paths":[
            ["/home/nhu.nguyen2/Nutrient_Classifier/Nutriendt-Classifier/Labels_and_csvs/april/0_4.csv",
            "/home/nhu.nguyen2/Nutrient_Classifier/Nutriendt-Classifier/Labels_and_csvs/march/0_3.csv",
            "/home/nhu.nguyen2/Nutrient_Classifier/Nutriendt-Classifier/Labels_and_csvs/may/0_5.csv"
            ]
        ],

    "label_path": "/home/nhu.nguyen2/Nutrient_Classifier/Nutriendt-Classifier/Labels_and_csvs/labels.yml",
    
    
    "21_dir": "/home/nhu.nguyen2/Nutrient_Classifier/Nutriendt-Classifier/DND-Diko-WWWR/WR2021/images",
    "20_dir": "/home/nhu.nguyen2/Nutrient_Classifier/Nutriendt-Classifier/DND-Diko-WWWR/WW2020/images",
    
    
    "model_save_dir": args['save_dir'],

    "run_id": [],
    "out_name": [],
    "momentum": [],
    "learning_rate": [],
    "weight_decay": [],
    "gamma": [],
    "batch_size": args['batch_size'],
    "num_workers": args['num_workers'],
    "epochs": args['epochs'],
    "epoch_step": args['epoch_step'],
    "image_size": args['image_size'],
    "model_name": args['model_name']
    }

    for i in range(num_files)]



    all_combo = list(product(lr, m, wd, g))


    all_len = len(all_combo)

    index = 0
    counter = 0
    name_counter = 0

    for i in range(num_files):
        if i != num_files - 1:
            while counter < all_len // num_files:
                for item in all_combo[index]:
                    pType, value = item

                    if pType == 'lr':
                        out_jsons[i]['learning_rate'].append(value)
                    elif pType == 'm':
                        out_jsons[i]['momentum'].append(value)
                    elif pType == 'wd':
                        out_jsons[i]['weight_decay'].append(value)
                    else:
                        out_jsons[i]['gamma'].append(value)

                index += 1
                counter += 1
                out_jsons[i]['run_id'].append(2)
                out_jsons[i]['out_name'].append(name_counter)
                name_counter += 1
            counter = 0

        else:
            while index < all_len:
                for item in all_combo[index]:
                    pType, value = item

                    if pType == 'lr':
                        out_jsons[i]['learning_rate'].append(value)
                    elif pType == 'm':
                        out_jsons[i]['momentum'].append(value)
                    elif pType == 'wd':
                        out_jsons[i]['weight_decay'].append(value)
                    else:
                        out_jsons[i]['gamma'].append(value)

                index += 1
                out_jsons[i]['run_id'].append(2)
                out_jsons[i]['out_name'].append(name_counter)
                name_counter += 1
        
        json_string = json.dumps(out_jsons[i], indent=2)
        
        with open(os.path.join(save_file,f"{args['model_name']}_{i}.json"), 'w') as out:
            out.write(json_string)

