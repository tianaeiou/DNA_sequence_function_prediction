import wandb
import time
import os
from config import args
from utils import print_commandline, set_seed, create_logger

DESCRIPTION = '{}_{}_'.format(args.model_name, args.data_type) + time.strftime(
    '%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
LOGGER = create_logger(output_dir='.', name=f"{args.model_name}")

# create folder by description
args.output = os.path.join(args.output, DESCRIPTION)
if not os.path.exists(args.output):
    os.mkdir(args.output)

# print args
print_commandline(args)

# set seed
set_seed(args.seed)

# set wandb
wandb.init(config=args, project="DNA_sequence_function_prediction", allow_val_change=True,
           dir=args.output, name=DESCRIPTION)

# start training
print('=-------------------training-------------------------=')
from trainer import DanQ_train

DanQ_train(args, LOGGER)