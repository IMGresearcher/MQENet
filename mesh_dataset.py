import os
import os.path
import pathlib
from random import shuffle
from typing import Union, List, Tuple

import numpy as np
import torch
from pandas import read_csv
from scipy.sparse import load_npz
from torch_geometric.data import Dataset, Data, InMemoryDataset

from preprocess import grd2element_graph


class InMemMeshDataset(InMemoryDataset):
    def __init__(self, root, raw_file_root, num, label_func, transform=None, pre_transform=None):
        """
        可以放入到内存中的网格数据集实现
        :param label_func: 输入网格文件名返回网格标签的函数
        :param root:数据集存储文件夹
        :param raw_file_root: 原始网格特征矩阵X和邻接矩阵A文件夹
        :param num:数据个数
        :param transform:在数据存储前对数据进行变换
        :param pre_transform:在数据存储到存盘前对数据进行变换
        """
        self.num = num  # 数据集样本个数
        self.label_func = label_func
        raw_file_root = pathlib.Path(raw_file_root)
        self.raw_filename = set()  # 所有的网格文件名
        self.raw_files = list()  # 所有的网格数据数据文件
        for file in raw_file_root.iterdir():
            if file.suffix in ['.npz', '.npy']:
                self.raw_filename.add(file.stem)
                self.raw_files.append(file.name)
        self.raw_filename = list(self.raw_filename)
        self.raw_filename.sort()  # 保证每次都是一个顺序

        self.raw_file_dir = raw_file_root
        # 打乱数据
        all_data = list(range(len(self.raw_filename)))
        shuffle(all_data)
        self.range = self.raw_filename[:self.num]  # 数据子集个数
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.raw_files

    @property
    def processed_file_names(self):
        return [f'data{self.num}.pt']

    def download(self):
        """
        从原始网格数据文件创建链接到数据集文件夹raw下
        :return:
        """
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
        """
        生成网格数据集,并保存在data{self.num}中
        :return:
        """

        data_list = []

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
            data_list.append(data)
            print(f'Processed Mesh {filename}.')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MeshDataset(Dataset):
    def __init__(self, root, raw_file_root, num, label_func, transform=None, pre_transform=None):
        """
        可以放入到内存中的网格数据集实现
        :param label_func: 输入网格文件名返回网格标签的函数
        :param root:数据集存储文件夹
        :param raw_file_root: 原始网格特征矩阵X和邻接矩阵A文件夹
        :param num:
        :param transform:
        :param pre_transform:
        """
        self.num = num  # 数据集样本个数
        self.label_func = label_func
        raw_file_root = pathlib.Path(raw_file_root)
        self.raw_filename = set()
        self.raw_files = list()
        for file in raw_file_root.iterdir():
            if file.suffix in ['.npz', '.npy']:
                self.raw_filename.add(file.stem)
                self.raw_files.append(file.name)
        self.raw_filename = list(self.raw_filename)
        self.raw_filename.sort()  # 保证每次都是一个顺序

        self.raw_file_dir = raw_file_root
        # 打乱数据
        all_data = list(range(len(self.raw_filename)))
        shuffle(all_data)
        self.range = self.raw_filename[:self.num]  # 数据子集个数
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.raw_files

    @property
    def processed_file_names(self):
        return [f'{filename}.pt' for filename in self.range]

    def download(self):
        """
        从原始网格数据文件创建链接到数据集文件夹raw下
        :return:
        """
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
        """
        生成网格数据集,并保存在data{self.num}中
        :return:
        """

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


class StreamMeshDataset(Dataset):
    def __init__(self, root, raw_file_root, num, label_func, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.num = num
        self.label_func = label_func
        self.raw_file_root = raw_file_root
        cnt = 0
        for _ in pathlib.Path(raw_file_root).iterdir():
            cnt += 1
        cnt //= 2
        self.num = min(self.num, cnt)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return self.num

    def get(self, idx):
        assert idx < self.num, 'idx不能大于数据个数'
        A_file = os.path.join(self.raw_file_root, f'Grid{idx}.npz')
        X_file = os.path.join(self.raw_file_root, f'Grid{idx}.npy')
        A = load_npz(A_file).tocoo()
        X = np.load(X_file)
        X = torch.from_numpy(X).float()
        row_col = np.array([A.row, A.col])
        A = torch.tensor(row_col, dtype=torch.long)
        label = self.label_func(f'Grid{idx}.grd')
        label = torch.tensor(label)
        data = Data(x=X, edge_index=A, y=label)
        return data


class GrdDataset(Dataset):
    """用于kaggle数据集的数据载入"""

    def __init__(self, root, num, label_func, xa_path, raw_file_root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.num = min(num, 10240)
        self.xa_path = pathlib.Path(xa_path)
        self.raw_file_root = pathlib.Path(raw_file_root)
        self.label_func = label_func
        self.data = dict()

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return self.num

    def get(self, idx):
        assert idx < self.num, 'idx不能大于数据个数'
        x_file = self.xa_path / f'Grid{idx}.npy'
        a_file = self.xa_path / f'Grid{idx}.npz'
        file = self.raw_file_root / f'Grid{idx}.grd'
        if x_file.exists() and a_file.exists():
            A = load_npz(a_file)
            X = np.load(x_file)
            X, A = self.data[idx]
        elif idx in self.data:
            pass
        else:
            X, A = grd2element_graph(str(file), '.', store=False)
            self.data[idx] = [X, A]
        A = A.tocoo()
        X = torch.from_numpy(X).float()
        row_col = np.array([A.row, A.col])
        A = torch.tensor(row_col, dtype=torch.long)
        label = self.label_func(f'{file.name}')
        label = torch.tensor(label)
        data = Data(x=X, edge_index=A, y=label)
        return data


def stl_label(filename) -> int:
    """
    传入网格文件名返回网格的标签
    :param filename: 文件名
    :return:int
    """
    filename = pathlib.Path(filename)
    filename = filename.stem
    if filename[8] == 'h':
        return 1
    elif filename[8] == 'p':
        return 0


def grd_label(filename) -> int:
    """
    :param filename:
    :return:
    """
    import re
    patten = r'Grid(\d*)'
    i = re.match(patten, filename).group(1)
    i = int(i)
    label_csv = read_csv('label.csv', header=None).values
    label = label_csv.transpose()[::2, :]
    label = np.argwhere(label <= i)[-1, 1]
    return label
