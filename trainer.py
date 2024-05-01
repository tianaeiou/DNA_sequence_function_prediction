import torch
import os
import time
import datetime
import wandb

import torch.nn as nn

from datasets import get_DataLoader
from models import get_model
from utils import EarlyStopping, save_checkpoint, save_best_model, commandline_to_json, AverageMeter, \
    get_grad_norm
from metrics import accuracy_score


def get_acc(pred, target):
    return float(sum(sum(pred == target)) / (target.shape[0] * target.shape[1]))


def DanQ_train(args, logger):
    # step 1: get DataLoader
    train_data_loader, test_data_loader = get_DataLoader(args)
    # step 2: get Model
    model = get_model(args)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model.to(device)
    logger.info(str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Creating model:{args.model_name} with number of params: {n_parameters}")

    # stage 3: build criterion
    criterion = nn.BCEWithLogitsLoss()
    # stage 4: build optimizer
    import torch.optim as optim
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)  # return_optimizer(args, model)
    # stage 5: build lr_scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,
                                                              verbose=1)  # return_scheduler(args, optimizer, len(train_data_loader))

    early_stopping = EarlyStopping('max', min_delta=1e-9, patience=args.patience)

    # train loop
    max_acc = 0.0
    max_acc_path = None
    start_time = time.time()
    metrics = [accuracy_score()]

    logger.info(" ------------------------------------- START TRAINING ------------------------------------- ")

    for epoch in range(args.start_epoch, args.num_epochs):
        train_loss, grad_norm, lr, train_acc, metrics = train_epoch(args, model, optimizer, train_data_loader,
                                                                    criterion,
                                                                    lr_scheduler, logger, epoch, metrics)

        test_loss, test_acc, metrics = test_epoch(args, test_data_loader, model, criterion, logger, metrics, epoch,
                                                  phase='test',
                                                  log_conf=False)
        # logger.info(f"train_loss: {train_loss}, train_acc:{train_acc}; valid_loss: {test_loss}. valid_acc: {test_acc}")

        if epoch + 1 % args.save_freq == 0:
            save_checkpoint(args, epoch, model, max(max_acc, test_acc), optimizer, lr_scheduler, logger)

        if test_acc > max_acc:
            max_acc = max(max_acc, test_acc)
            if max_acc_path is not None and args.only_save_best:
                logger.info('Find new Acc:{}, delete {}'.format(max_acc, max_acc_path))
                os.remove(max_acc_path)
            max_acc_path = save_best_model(args, epoch, model, max_acc, optimizer, lr_scheduler, logger, save_all=False,
                                           save_threshold=args.save_threshold)
        logger.info(
            f'Max Acc: {max_acc * 100.0:.2f}%, Current Valid Acc: {test_acc * 100.0:.2f}%, Current Train Acc: {train_acc * 100.0:.2f}%')
        if early_stopping.step(test_acc):
            print('Early Stopped!!!!')
            break

        wandb.log(
            {'test_acc': test_acc, 'test_loss': test_loss, 'max_acc': max_acc, 'train_loss': train_loss,
             'train_acc': train_acc, 'grad_norm': grad_norm, 'lr': lr, 'epoch': epoch})
    last_epoch_model_path = save_best_model(args, epoch, model, max_acc, optimizer, lr_scheduler, logger, save_all=False,
                                   save_threshold=args.save_threshold)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    commandline_to_json(args, logger)


def train_epoch(args, model, optimizer, data_loader, criterion, lr_scheduler, logger, epoch, metrics, phase='Train'):
    for metric in metrics:
        metric.reset()
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    acc_meter = AverageMeter()
    start = time.time()
    end = time.time()
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    print('device here is', device)
    for idx, (train_batch_x, train_batch_y) in enumerate(data_loader):
        batch_len = train_batch_y.shape[0]
        # 2. forward
        outputs = model(train_batch_x)
        # 3. calculate loss
        loss = criterion(outputs, train_batch_y)
        # 4. Back Propagation
        optimizer.zero_grad()
        loss.backward()
        # 5. update param
        if args.clip_grad:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        else:
            grad_norm = get_grad_norm(model.parameters())

        optimizer.step()
        # 6. update lr
        lr_scheduler.step(loss)

        # 7. update metric
        loss_meter.update(loss.item(), batch_len)
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        preds = (outputs >= 0.5).float()
        acc = get_acc(preds.detach().cpu().numpy(),
                      train_batch_y.detach().cpu().numpy())  # sklearn.metrics.accuracy_score(train_batch_y.detach().cpu().numpy(), preds.detach().cpu().numpy())
        acc_meter.update(acc)
        end = time.time()
        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{args.num_epochs}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB\t'
                f'Acc {acc_meter.val:.2f} ({acc_meter.avg:.2f})\t')
        for metric in metrics:
            metric(outputs, train_batch_y, None, phase)
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return loss_meter.avg, norm_meter.avg, lr, acc_meter.avg, metrics


def test_epoch(args, data_loader, model, criterion, logger, metrics, epoch, phase=None, log_conf=False):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()

        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        end = time.time()
        for idx, (batch_x, batch_y) in enumerate(data_loader):
            # 1.  prepare data
            batch_len = batch_y.shape[0]
            # 2. forward
            outputs = model(batch_x)
            preds = (outputs >= 0.5).float()
            # 3. calculate loss
            loss = criterion(outputs, batch_y)
            # 7. update metric
            loss_meter.update(loss.item(), batch_len)
            batch_time.update(time.time() - end)
            acc = get_acc(preds.detach().cpu().numpy(),
                          batch_y.detach().cpu().numpy())  # sklearn.metrics.accuracy_score(batch_y.detach().cpu().numpy(), preds.detach().cpu().numpy())
            acc_meter.update(acc)
            for metric in metrics:
                metric(outputs, batch_y, None, phase)
            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_freq == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')
    return loss_meter.avg, acc_meter.avg, metrics
