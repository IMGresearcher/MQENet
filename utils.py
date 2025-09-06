import random
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm


def adjust_params(params):

    new_params = OrderedDict()
    for key, value in params.items():
        if key.startswith('module.'):
            new_params[key[7:]] = value
        else:
            new_params[key] = value

    return new_params


def best_model(model, optim, d, val_acc, epoch):

    d['best_val_acc'] = val_acc
    d['best_net'] = deepcopy(model.state_dict())
    d['best_optim'] = deepcopy(optim.state_dict())
    d['best_epoch'] = epoch


def coo2tensor(sparse_matrix):

    values = sparse_matrix.data
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = sparse_matrix.shape
    sparse_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_matrix


def split_dataset(dataset, train=.6, val=.2):

    import random
    num = len(dataset)
    index = list(range(num))
    random.shuffle(index, random.seed(0))
    return dataset[index[:int(train * num)]], dataset[index[int(train * num):int((train + val) * num)]], dataset[
        index[int((train + val) * num):]]


def set_seed(n):

    torch.manual_seed(n)
    random.seed(n)
    np.random.seed(n)


def train_model(model, optimizer, criterion, train_loader, val_loader, model_dict,  param_file, epochs,
                device="cuda:0", init_epoch=0, lr_scheduler=None, is_parallel=False, grad_clipping_value=None,
                patience=20, verbose=True):
    best_val_acc = best_train_acc = early_stopping_cnt = 0.
    for epoch in tqdm(range(epochs), total=epochs, leave=False):
        epoch += init_epoch
        train_cnt = 0  
        train_acc_sum = 0 
        model.train()
        train_loss = 0
        save_model(model_dict, param_file, model, optimizer)
        for data in tqdm(train_loader, leave=False, total=len(train_loader)):
            if not is_parallel:
                data = data.to(device)
                y_hat = data.y
            else:
                y_hat = [item[1].cpu().detach().numpy() for item in data]
                #y_hat = y_hat.numpy()
                y_hat = torch.tensor(y_hat)

            optimizer.zero_grad()
            result = model(data)

            loss = criterion(result, y_hat)
            train_loss += loss.item()

            torch.cuda.empty_cache()
            loss.backward()  
            if grad_clipping_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=grad_clipping_value)
            optimizer.step()
            train_acc = torch.eq(result.max(1)[1], y_hat).float().mean()


            train_acc_sum += train_acc.item() * len(y_hat)
            train_cnt += len(y_hat)
            torch.cuda.empty_cache()

        print("Train_loss", train_loss / train_cnt, epoch)
        print("Train_acc", train_acc_sum / train_cnt, epoch)
        if train_acc_sum / train_cnt >= best_train_acc:
            best_train_acc = model_dict['best_train_acc'] = train_acc_sum / train_cnt


        model.eval()
        val_loss = 0
        val_sum_acc = 0
        val_cnt = 0
        with torch.no_grad():
            for data in tqdm(val_loader, total=len(val_loader)):
                if not is_parallel:
                    data = data.to(device)
                    y_hat = data.y
                else:
                    y_hat = [item.y for item in data]
                    y_hat = torch.tensor(np.array(y_hat))
                result = model(data)
                loss = criterion(result, y_hat)
                val_loss += loss.item()
                test_acc = torch.eq(result.max(1)[1], y_hat).float().mean()
                val_sum_acc += test_acc.item() * y_hat.shape[0]
                val_cnt += y_hat.shape[0]

        torch.cuda.empty_cache()

        if val_sum_acc / val_cnt >= best_val_acc:
            best_model(model, optimizer, model_dict, val_sum_acc / val_cnt, epoch)
            best_val_acc = val_sum_acc / val_cnt
            early_stopping_cnt = 0
        else:
            # early_stopping
            if patience:
                early_stopping_cnt += 1
                if early_stopping_cnt > patience:
                    break

        if lr_scheduler:
            lr_scheduler.step(best_train_acc)
            lr = [group['lr'] for group in optimizer.param_groups]
            lr = lr[0]
            #writer.add_scalar('Learning Rate', lr, epoch)

        if verbose:
            print(
                f"Epoch {epoch:03d}:  TrainAcc {train_acc_sum / train_cnt:.4} ValAcc {val_sum_acc / val_cnt:.4}"
                f" BestTrainAcc {best_train_acc:.4} BestValAcc {best_val_acc:.4}\n")


def test_model(model, test_data, device="cuda:0", parallel=False):

    model.eval()
    test_sum_acc = 0
    test_cnt = 0
    with torch.no_grad():
        for data in tqdm(test_data, total=len(test_data)):
            if parallel:
                y_hat = [item.y for item in data]
                y_hat = torch.tensor(np.array(y_hat))
            else:
                data = data.to(device)
                y_hat = data.y
            logit = model(data)
            test_acc = torch.eq(logit.max(1)[1], y_hat).float().mean()
            test_sum_acc += test_acc.item() * y_hat.shape[0]
            test_cnt += y_hat.shape[0]

    torch.cuda.empty_cache()

    print("test accuracy: ",test_sum_acc / test_cnt)

    return test_sum_acc / test_cnt


def save_model(model_dict, param_file, model, optimizer):

    model_dict.update({'net': deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict())})
    torch.save(model_dict, param_file)


def print_results(model_dict):

    print(f'Best Train Acc: {model_dict["best_train_acc"]:.4%}')
    print(f'Best Val Acc: {model_dict["best_val_acc"]:.4%}')
    print(f'Best Test Acc: {model_dict["best_test_acc"]:.4%}')
    print(f'Final Model Acc: {model_dict["net_acc"]:.4%}')


def add_weight_decay(model, weight_decay):

    weight = []
    bias = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight.append(param)
        else:
            bias.append(param)

    params = [{'params': weight, 'weight_decay': weight_decay}, {'params': bias}]
    return params


def adjust_lr(optimizer, lr):

    for params in optimizer.param_groups:
        params['lr'] = lr
