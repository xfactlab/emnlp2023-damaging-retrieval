import argparse
from tqdm.auto import tqdm
import os
import pickle
from pathlib import Path


def save_by_instance(data, output_path):
    for instance in data:
        temp_dict = {}
        output_file_name = f"{instance['id']}"
        output_file = output_path + '/' + output_file_name + '.pickle'

        temp_dict['id'] = instance['id']
        temp_dict['embedding'] = instance['embedding']

        with open(output_file, 'wb') as f:
            pickle.dump(temp_dict, f)

def main():

    parser = argparse.ArgumentParser(description='Pickle -> Pickle by instance')

    parser.add_argument('--file_type', type=str,
                        default='train')
    parser.add_argument('--path_name', type=str,
                        default='/scratch/philhoon-relevance/decoder-classification/NQ-DEV-DPR/5-fold/1')

    args = parser.parse_args()

    output_path = args.path_name + '/' + f'{args.file_type}'
    os.makedirs(output_path, exist_ok = True)

    path = Path(args.path_name)
    file_filter = f'ctx100id_embedding_{args.file_type}_*.pickle'

    print(args.file_type)
    print(args.path_name)
    print(output_path)
    print(file_filter)

    file_lst = path.glob(file_filter)

    cnt = 0
    for file in tqdm(file_lst):
        with open(file, 'rb') as f:
            data = pickle.load(f)

        save_by_instance(data, output_path)
        cnt += 1
    print(cnt)

if __name__ == "__main__":
    main()

# python encoder_embedding_pickles_by_instance.py \
# --file_type train \
# --path_name /scratch/philhoon-relevance/decoder-classification/NQ-DEV-DPR/5-fold/1