import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from tensorboardX import SummaryWriter

import pdb

import datetime

from util import dataset, transform, config
from util.util import AverageMeter, poly_learning_rate, calc_mae, check_makedirs

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
from torch.utils.data import DataLoader, SubsetRandomSampler
import math
import faiss
import h5py

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/cod_mgl50.yaml', help='config file')
    parser.add_argument('opts', help='see config/cod_mgl50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def check(args):
    assert args.classes == 1
    assert args.zoom_factor in [1, 2, 4, 8]
    if args.arch == 'mgl':
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    else:
        raise Exception('architecture not supported yet'.format(args.arch))


def main():
    args = get_parser()
    check(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    save_folder = args.save_folder + '/tmp'
    gray_folder = os.path.join(save_folder, 'gray')
    edge_folder = os.path.join(save_folder, 'edge')
    check_makedirs(save_folder)
    check_makedirs(gray_folder)
    check_makedirs(edge_folder)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(manualSeed)
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args, gray_folder, edge_folder))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args, gray_folder, edge_folder)


def main_worker(gpu, ngpus_per_node, argss, gray_folder, edge_folder):
    global args
    args = argss
    if args.sync_bn:
        if args.multiprocessing_distributed:
            BatchNorm = apex.parallel.SyncBatchNorm
        else:
            from lib.sync_bn.modules import BatchNorm2d
            BatchNorm = BatchNorm2d
    else:
        BatchNorm = nn.BatchNorm2d
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    if args.arch == 'mgl':
        from model.mglnet import MGLNet
        model = MGLNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion, BatchNorm=BatchNorm, pretrained=False, args=args)

        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4, model.region_conv, model.edge_cat]
        modules_new = [model.mutualnet0] #, model.mutualnet1]
        # model.edge_cat, model.mutualnet0.edge_proj0, model.mutualnet0.edge_conv, model.mutualnet0.region_conv1, model.mutualnet0.region_conv2, model.mutualnet0.r2e, model.mutualnet0.e2r]

        frozen_layers = [] #[model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        for l in frozen_layers:
            for p in l.parameters():
                p.requires_grad = False
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr ))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
    args.index_split = 5
    optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        if args.use_apex:
            model, optimizer = apex.amp.initialize(model.cuda(), optimizer, opt_level=args.opt_level, keep_batchnorm_fp32=args.keep_batchnorm_fp32, loss_scale=args.loss_scale)
            model = apex.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])

    else:
        model = torch.nn.DataParallel(model.cuda())


    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if main_process():
                logger.info("=> loaded weight '{}', epoch {}".format(args.weight, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.Resize((args.train_h, args.train_w)),
        #transform.RandScale([args.scale_min, args.scale_max]),
        #transform.RandomEqualizeHist(),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.RandomVerticalFlip(),
        #transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    train_data = dataset.SemData(split='train', data_root=args.data_root, data_list=args.train_list, transform=train_transform)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    if args.evaluate:
        val_transform = transform.Compose([
            transform.Resize((args.train_h, args.train_w)),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
        val_data = dataset.SemData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)


    date_str = str(datetime.datetime.now().date())
    check_makedirs(args.save_path + '/' + date_str)
    best_mae = 255.
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train = train(train_loader, model, optimizer, epoch, train_data.data_list)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)

        # pdb.set_trace()
        if args.evaluate:
            r_mae, e_mae = validate(val_loader, model, gray_folder, edge_folder, val_data.data_list)
            if main_process():
                writer.add_scalar('r_mae', r_mae)
                writer.add_scalar('e_mae', e_mae)
            curr_mae = r_mae # + e_mae
    
            if curr_mae < best_mae and main_process():
                best_mae = curr_mae
                filename = args.save_path + '/' + date_str + '/train_best.pth'

                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                    if main_process():
                        logger.info('Saving checkpoint to: ' +  filename)
                    torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
                except IOError:
                    logger.info('error')
                filename = args.save_path + '/' + date_str + '/train_best_' + str(epoch_log) + '.pth'
                torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/' + date_str  + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

            '''
            if epoch_log / args.save_freq > 2:
                deletename = args.save_path + '/' + date_str + '/train_epoch_' + str(epoch_log - args.save_freq * 2) + '.pth'
                try:
                    if os.path.exists(deletename):
                        os.remove(deletename)
                except IOError:
                    logger.info('error')
            '''

def train(train_loader, model, optimizer, epoch, data_list):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    #aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    visual = False
    if visual:
        date_str = str(datetime.datetime.now().date())
        save_folder = args.save_folder + '/' + date_str
        check_makedirs(save_folder)

        fg_folder = os.path.join(save_folder, 'fg')
        bg_folder = os.path.join(save_folder, 'bg')
        check_makedirs(fg_folder)
        check_makedirs(bg_folder)

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, target, edge) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.zoom_factor != 8:
            h = int((target.size()[1] - 1) / 8 * args.zoom_factor + 1)
            w = int((target.size()[2] - 1) / 8 * args.zoom_factor + 1)
            # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
            target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='bilinear', align_corners=True).squeeze(1).long()
            edge = F.interpolate(edge.unsqueeze(1).float(), size=(h, w), mode='bilinear', align_corners=True).squeeze(1).long()


        target = torch.where(target > 127, torch.full_like(target, 255), torch.full_like(target, 0))
        edge = torch.where(edge > 127, torch.full_like(edge, 255), torch.full_like(edge, 0))

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        edge = edge.cuda(non_blocking=True)

        target = target.unsqueeze(1).float() / 255.
        edge = edge.unsqueeze(1).float() / 255.

        region, edge, main_loss = model(input, target, epoch, edge)
        #output, main_loss, fg_assign, bg_assign, colors, h, w = model(input, target, epoch, edge)

        if not args.multiprocessing_distributed:
            main_loss = torch.mean(main_loss)
        loss = main_loss

        optimizer.zero_grad()
        if args.use_apex and args.multiprocessing_distributed:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        # output = torch.sigmoid(output)
        n = input.size(0)
        if args.multiprocessing_distributed:
            main_loss, loss = main_loss.detach() * n, loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(main_loss), dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            main_loss, loss = main_loss / n, loss / n

        main_loss_meter.update(main_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        for index in range(0, args.index_split): # backbone
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          loss_meter=loss_meter))
        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
    if main_process():
        logger.info('Train result at epoch [{}/{}]'.format(epoch+1, args.epochs))

    torch.cuda.empty_cache()

    return main_loss_meter.avg


def validate(val_loader, model, gray_folder, edge_folder, data_list):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    r_mae, e_mae = AverageMeter(), AverageMeter()

    sync_idx = 0

    model.eval()
    for i, (input, target1, target2) in enumerate(val_loader):
        input = input.cuda(non_blocking=True)
        with torch.no_grad():
            pred1, pred2 = model(input)
        pred1, pred2 = torch.sigmoid(pred1.squeeze(1)), torch.sigmoid(pred2.squeeze(1))
 
        if args.zoom_factor != 8:
            pred1 = F.interpolate(pred1, size=target.size()[1:], mode='bilinear', align_corners=True)
            pred2 = F.interpolate(pred2, size=target.size()[1:], mode='bilinear', align_corners=True)

        pred1, pred2 = pred1.detach().cpu().numpy(), pred2.detach().cpu().numpy()
        target1, target2 = target1.numpy(), target2.numpy()

        for j in range(len(pred1)):
            pred1_j = np.uint8(pred1[j]*255)
            pred2_j = np.uint8(pred2[j]*255)

            '''
            img_name =  'during_training.png'
            cv2.imwrite(os.path.join(gray_folder, img_name), pred1_j)
            cv2.imwrite(os.path.join(edge_folder, img_name), pred2_j)

            pred1_j = cv2.imread(os.path.join(gray_folder, img_name), cv2.IMREAD_GRAYSCALE)
            pred2_j = cv2.imread(os.path.join(edge_folder, img_name), cv2.IMREAD_GRAYSCALE)                        
            '''

            if pred1_j is not None:
                r_mae.update(calc_mae(pred1_j, target1[j]))

            if pred2_j is not None:
                e_mae.update(calc_mae(pred2_j, target2[j]))

            sync_idx += 1

    if main_process():
        logger.info('val result: region_mae / edge_mae {:.7f}/{:.7f}'.format(r_mae.avg, e_mae.avg))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return r_mae.avg, e_mae.avg


if __name__ == '__main__':
    main()

