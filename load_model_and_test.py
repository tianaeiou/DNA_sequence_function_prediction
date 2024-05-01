import torch
from models import DanQ
from trainer import test_epoch
from datasets import get_testDataLoader
from config import args
from utils import create_logger
from metrics import accuracy_score

import torch.nn as nn

model_dir = "/Users/person/codeScope/BEHI/600A/group project/output/DanQ_whole_2024-04-30_10-14-03/0.98_best_model_epoch_0.pth"

# load model
model = DanQ()
model.load_state_dict(torch.load(model_dir))
# model.eval()

# load testing data
test_dataloader = get_testDataLoader(args)
criterion = nn.BCEWithLogitsLoss()
logger = create_logger(output_dir='.', name=f"{args.model_name}")
metrics = [accuracy_score()]
loss, acc, metrics_result = test_epoch(args, test_dataloader, model, criterion, logger, metrics, 1)

# test performance
