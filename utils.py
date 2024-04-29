import json
import random
import torch
import os
import sys
import logging
import functools
import numpy as np
from termcolor import colored
from torch import optim as optim


def print_commandline(commandline):
    commandline_arg_dic = vars(commandline)
    # save to json file
    # json = json.dumps(commandline_arg_dic, skipkeys=True, indent=4)
    print(json.dumps(commandline_arg_dic, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # 是否使用使用确定性算法
    torch.backends.cudnn.benchmark = True  # 是否使用加速算法，在输入数据不变时使用
    # torch.backends.cudnn.enabled = False


@functools.lru_cache()
def create_logger(output_dir, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def return_optimizer(args, model):
    opt_lower = args.optimizer.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(model.parameters(), momentum=args.optimizer_momentum, nesterov=True,
                              lr=args.learning_rate, weight_decay=args.weight_decay)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(model.parameters(), eps=args.optimizer_eps, betas=args.optimizer_betas,
                                lr=args.learning_rate, weight_decay=args.weight_decay)
    elif opt_lower == 'rmsprot':
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        import numpy as np

        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')

        if mode == 'min':
            self.is_better = lambda a, best: a < best - best * min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + best * min_delta

def commandline_to_json(commandline, logger=None):
    commandline_arg_dic = vars(commandline)

    output_dir = commandline_arg_dic['output']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # remove these entries from the commandline log file
    # if 'ensemble_paths' in commandline_arg_dic:
    #     del commandline_arg_dic['ensemble_paths']

    # save to json file
    json_info = json.dumps(commandline_arg_dic, skipkeys=True, indent=4)
    if logger is not None:
        logger.info("Path of json file:{}".format(os.path.join(output_dir, "commandline.json")))
    else:
        print("Path of json file:{}".format(os.path.join(output_dir, "commandline.json")))
    f = open(os.path.join(output_dir, "commandline.json"), "w")
    f.write(json_info)
    f.close()

def save_best_model(args, epoch, model, max_acc, optimizer, lr_scheduler, logger, save_all=False, save_threshold=60):
    if max_acc >= 0:  # max_acc >= save_threshold:
        if save_all:
            save_state = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'max_acc': max_acc,
                          'epoch': epoch,
                          'args': args}
            save_path = os.path.join(args.output, f'ckpt_epoch_{epoch}.pth')
            logger.info(f"{save_path} saving......")
            torch.save(save_state, save_path)
            logger.info(f"{save_path} saved !!!")
        save_path = os.path.join(args.output, f'{max_acc:.2f}_best_model_epoch_{epoch}.pth')
        logger.info(f"Best acc: {max_acc} - {save_path} saving......")
        torch.save(model.state_dict(), save_path)
        # if args.save_complete_model:
        #     torch.save(model, save_path.replace('pth', 'pkl'))
        logger.info(f"Best acc: {max_acc} - {save_path} saved !!!")

        return save_path
    else:
        return None

def save_checkpoint(args, epoch, model, max_acc, optimizer, lr_scheduler, logger, fold=None):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_acc': max_acc,
                  'epoch': epoch,
                  'args': args}
    if fold is None:
        save_path = os.path.join(args.output, f'ckpt_epoch_{epoch}.pth')
    else:
        save_path = os.path.join(args.output, f'ckpt_fold_{fold}_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm
