import os
import os.path
import pathlib
from random import shuffle
from typing import Union, List, Tuple

import numpy as np
import torch
from pandas import read_csv
from scipy.sparse import load_npz
from torch_geometric.data import Dataset, Data

from preprocess import grd2element_graph


class MeshDataset(Dataset):
    def __init__(self, root, raw_file_root, num, label_func, transform=None, pre_transform=None):
        self.num = num 
        self.label_func = label_func
        raw_file_root = pathlib.Path(raw_file_root)
        self.raw_filename = set()
        self.raw_files = list()
        for file in raw_file_root.iterdir():
            if file.suffix in ['.npz', '.npy']:
                self.raw_filename.add(file.stem)
                self.raw_files.append(file.name)
        self.raw_filename = list(self.raw_filename)
        self.raw_filename.sort() 

        self.raw_file_dir = raw_file_root
        all_data = list(range(len(self.raw_filename)))
        shuffle(all_data)
        self.range = self.raw_filename[:self.num] 
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.raw_files

    @property
    def processed_file_names(self):
        return [f'{filename}.pt' for filename in self.range]

    def download(self):

        for filename in self.raw_filename:
            A_scr = os.path.join(self.raw_file_dir, f'{filename}.npz')
            A_scr = os.path.abspath(A_scr)
            X_scr = os.path.join(self.raw_file_dir, f'{filename}.npy')
            X_scr = os.path.abspath(X_scr)

            A_des = os.path.join(self.raw_dir, f'{filename}.npz')
            A_des = os.path.abspath(A_des)
            X_des = os.path.join(self.raw_dir, f'{filename}.npy')
            X_des = os.path.abspath(X_des)
            try:
                os.symlink(A_scr, A_des)
            except FileExistsError:
                pass
            try:
                os.symlink(X_scr, X_des)
            except FileExistsError:
                pass


            print(f'Copy Mesh{filename}')

    def process(self):


        for filename in self.range:
            A_file = os.path.join(self.raw_dir, f'{filename}.npz')
            X_file = os.path.join(self.raw_dir, f'{filename}.npy')
            A = load_npz(A_file).tocoo()
            X = np.load(X_file)
            X = torch.from_numpy(X).float()
            row_col = np.array([A.row, A.col])
            A = torch.tensor(row_col, dtype=torch.long)
            label = self.label_func(filename)
            label = torch.tensor(label)
            data = Data(x=X, edge_index=A, y=label)

            if self.pre_filter and not self.pre_filter(data):
                continue

            if self.pre_transform:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'{filename}.pt'))

            print(f'Processed Mesh {filename}.')

    def len(self) -> int:
        return len(self.range)

    def get(self, idx: int) -> Data:
        return torch.load(os.path.join(self.processed_dir,
                                       f'{self.range[idx]}.pt'))

def grd_label(filename) -> int:
    import re
    patten = r'Grid(\d*)'
    i = re.match(patten, filename).group(1)
    i = int(i)
    label_csv = read_csv('label.csv', header=None).values
    label = label_csv.transpose()[::2, :]
    label = np.argwhere(label <= i)[-1, 1]
    return label
