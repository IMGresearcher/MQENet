import os
import pathlib
import shutil
import warnings
from urllib import request
from urllib.request import Request

import torch
from torch.jit import TracerWarning

from .model import TraceModel


def generate_jit_model(model_config_file, output_path):
    """
    将指定模型生成Torchscript脚本
    pkl文件网络配置文件；
    pt文件模型文件
    :param model_config_file:
    :return:
    """
    print(f'Generating model from {model_config_file}...')
    model_config = torch.load(model_config_file)
    params = model_config['params']
    config = model_config['config']
    data = model_config['data']
    warnings.simplefilter('ignore', TracerWarning)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = TraceModel(**config)
    model = model.to(device)
    data = data.to(device)
    data.batch = torch.zeros((data.x.size(0)), dtype=torch.long, device=data.x.device)
    model.load_state_dict(params)
    model.eval()
    traced_model = torch.jit.trace(model, (data.x, data.edge_index, data.batch))
    model_name = f'Model{pathlib.Path(model_config_file).stem}.pt'
    traced_model.save(f'{output_path}/{model_name}')
    print(f'Torchscript model {model_name} generated successfully.')


def pack_model(model_dir, pt_file, version, model_store):
    """
    将model_dir下的模型进行打包,生成部署需要的mar文件,并将文件移动到model-store下
    model_dir需要的文件
    pkl：模型配置，权重，和输入文件
    pt：模型TorchScript文件
    index_to_name.json：输出映射文件
    :param version:
    :param model_dir:
    :param pt_file:
    :param model_store:
    :return:
    """
    model_name = pathlib.Path(pt_file).stem
    model_dir = pathlib.Path(model_dir)
    os.system(
        f'torch-model-archiver --model-name {model_name}_v{version} --serialized-file {str(pt_file)}'
        f' --handler GMeshNet/handler.py --extra-files GMeshNet/preprocess.py '
        f'--runtime python3 --version {version} -f --export-path {model_dir}')

    shutil.copy(f'{model_dir}/{model_name}_v{version}.mar', model_store)
    print(f'Successfully pack model {model_name}.')


def register_model(model_name, version):
    """
    部署模型
    :param version:
    :param model_name:
    :return:
    """
    url = Request(
        f'http://localhost:8081/models?url={model_name}_v{version}.mar&model_name={model_name}&initial_workers=1&synchronous=True',
        method='POST')
    try:
        response = request.urlopen(url)
        print(response.read().decode('utf-8'))
    except Exception as e:
        print(e)
        print(f'Fail to register model {model_name}.')
        raise e


def unregister_model(model_name, version):
    """
    取消部署模型
    :param version:
    :param model_name:
    :return:
    """
    url = Request(f'http://localhost:8081/models/{model_name}/{version}', method='DELETE')
    try:
        response = request.urlopen(url)
        print(response.read().decode('utf-8'))
    except Exception as e:
        print(e)
        print(f'Fail to unregister model {model_name}.')
        raise e


def show_model(model_name, version='all'):
    """
    获取模型信息
    :param model_name:
    :param version:
    :return:
    """
    url = Request(f'http://localhost:8081/models/{model_name}/{version}', method='GET')
    try:
        response = request.urlopen(url)
        print(response.read().decode('utf-8'))
    except Exception as e:
        print(e)
        print(f'Fail to show model {model_name}.')
        raise e


def scale_model(model_name, min_worker=None, max_worker=None):
    """
    调整模型的进程数
    :param max_worker:
    :param model_name:
    :param min_worker:
    return:
    """
    if max_worker is None:
        max_worker = min_worker
    if min_worker is None:
        min_worker = max_worker

    url = Request(
        f'http://localhost:8081/models/{model_name}?min_worker={min_worker}&max_worker={max_worker}&synchronous=True',
        method='PUT')
    try:
        response = request.urlopen(url)
        print(response.read().decode('utf-8'))
    except Exception as e:
        print(e)
        print(f'Fail to scale model {model_name}.')
        raise e


def list_all_models():
    """
    列表模型信息
    :param :
    :param :
    :return :
    """
    url = Request(f'http://localhost:8081/models', method='GET')
    try:
        response = request.urlopen(url)
        print(response.read().decode('utf-8'))
    except Exception as e:
        print(e)
        print(f'Fail to list models.')
        raise e


def check_server():
    """
    检查推理是否ok
    :return:
    """
    url = Request(f'http://localhost:8080/ping', method='GET')
    try:
        response = request.urlopen(url)
        print(response.read().decode('utf-8'))
    except Exception as e:
        print(e)
        print(f'TorchServe is down.')
        raise e
