import os
import glob

if __name__ == '__main__':
    txt_path = '/home/nhu.nguyen2/Nutrient_Classifier/Nutriendt-Classifier/GridSearch/EN_b6'


    all_files = [os.path.basename(i) for i in glob.glob(f'{txt_path}/*.txt')]
    # print(all_files)


    best_performance = 0.0
    best_name = ''

    for i in all_files:
        try:
            curr_path = os.path.join(txt_path, i)
            f = open(curr_path, 'r')
            best = f.readlines()[-3:-2]
            if len(best) == 0:
                continue

            best = best[0]
            performance = float(best[-11:-1])

            if performance > best_performance:
                best_performance = performance
                best_name = i

            f.close()
            
        except FileNotFoundError:
            continue
            

    print(f'Performance: {best_performance} --- Name: {best_name}')