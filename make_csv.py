import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split


def make_csv_cub(input_path, csv_path):
    '''
    Make CUB 200 2011 csv file with corrected indices.
    '''
    info = []
    for subdir in os.scandir(input_path):
        label = int(subdir.name.split('.')[0]) - 1  # 直接在这里减1
        path_list = glob.glob(os.path.join(subdir.path, "*.jpg"))
        sub_info = [[item, label] for item in path_list]
        info.extend(sub_info)

    col = ['id', 'label']
    info_data = pd.DataFrame(columns=col, data=info)
    info_data.to_csv(csv_path, index=False)


def split_csv_cub(csv_path, train_csv_path, test_csv_path, split_ratio=0.8):
    '''
    Split CUB 200 2011 csv file using stratified sampling.
    '''
    info_data = pd.read_csv(csv_path)
    train_data, test_data = train_test_split(
        info_data, test_size=1 - split_ratio, stratify=info_data['label'])

    train_data.to_csv(train_csv_path, index=False)
    test_data.to_csv(test_csv_path, index=False)


if __name__ == "__main__":
    # Ensure directory exists
    if not os.path.exists('./csv_file'):
        os.makedirs('./csv_file')

    input_path = './datasets/CUB_200_2011/CUB_200_2011/images/'
    csv_path = './csv_file/cub_200_2011.csv'
    train_csv_path = './csv_file/cub_200_2011_train.csv'
    test_csv_path = './csv_file/cub_200_2011_test.csv'

    make_csv_cub(input_path, csv_path)

    # Split csv file using stratified sampling
    split_csv_cub(csv_path, train_csv_path, test_csv_path)
