import os
import time
import logging
import argparse
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from util import dataset, transform, config
from util.util import AverageMeter, calc_mae, check_makedirs
import datetime

cv2.ocl.setUseOpenCL(False)


def get_parser(cfg_path):
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default=cfg_path, help='config file')
    parser.add_argument('opts', help='see ' + cfg_path + 'for all options', default=None, nargs=argparse.REMAINDER)
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


def check(args):
    assert args.classes == 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert args.split in ['train', 'val', 'test']
    if args.arch == 'mgl':
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    else:
        raise Exception('architecture not supported yet'.format(args.arch))


def main():
    global args, logger
    args = get_parser('config/cod_mgl50.yaml')
    check(args)
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    date_str = str(datetime.datetime.now().date())
    save_folder = args.save_folder + '/' + date_str
    check_makedirs(save_folder)
    cod_folder = os.path.join(save_folder, 'cod')
    coee_folder = os.path.join(save_folder, 'coee')

    test_transform = transform.Compose([
             transform.Resize((args.test_h, args.test_w)),
             transform.ToTensor(),
              transform.Normalize(mean=mean, std=std)])

    test_data = dataset.SemData(split=args.split, data_root=args.data_root, data_list=args.test_list, transform=test_transform)
    index_start = args.index_start
    if args.index_step == 0:
        index_end = len(test_data.data_list)
    else:
        index_end = min(index_start + args.index_step, len(test_data.data_list))
    test_data.data_list = test_data.data_list[index_start:index_end]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    if not args.has_prediction:
        if args.arch == 'mgl':
            from model.mglnet import MGLNet
            model = MGLNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False, args=args)
        #logger.info(model)
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        if os.path.isfile(args.model_path):
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path, map_location='cuda:0')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info("=> loaded checkpoint '{}', epoch {}".format(args.model_path, checkpoint['epoch']))

        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
        test(test_loader, test_data.data_list, model, cod_folder, coee_folder)
    if args.split != 'test':
        calc_acc(test_data.data_list, cod_folder, coee_folder)


def test(test_loader, data_list, model, cod_folder, coee_folder):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    check_makedirs(cod_folder)
    check_makedirs(coee_folder)
    for i, (input, _, _) in enumerate(test_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            cod_pred, coee_pred = model(input)
        cod_pred, coee_pred = torch.sigmoid(cod_pred), torch.sigmoid(coee_pred) 
        
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        cod = np.uint8(cod_pred.squeeze().detach().cpu().numpy()*255)
        coee = np.uint8(coee_pred.squeeze().detach().cpu().numpy()*255)

        image_path, _, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        cod_path = os.path.join(cod_folder, image_name + '.png')
        coee_path = os.path.join(coee_folder, image_name + '.png')
        cv2.imwrite(cod_path, cod)
        cv2.imwrite(coee_path, coee)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def calc_acc(data_list, pred_folder, edge_folder):
    r_mae = AverageMeter()
    e_mae = AverageMeter()

    for i, (image_path, target1_path, target2_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred1 = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
        pred2 = cv2.imread(os.path.join(edge_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)

        target1 = cv2.imread(target1_path, cv2.IMREAD_GRAYSCALE)
        target2 = cv2.imread(target2_path, cv2.IMREAD_GRAYSCALE)

        if pred1.shape[1] != target1.shape[1] or pred1.shape[0] != target1.shape[0]:
            pred1 = cv2.resize(pred1, (target1.shape[1], target1.shape[0]))
            pred2 = cv2.resize(pred2, (target2.shape[1], target2.shape[0]))

        r_mae.update(calc_mae(pred1, target1))
        e_mae.update(calc_mae(pred2, target2))

        logger.info('Evaluating {0}/{1} on image {2}, mae {3:.4f}.'.format(i + 1, len(data_list), image_name+'.png', r_mae.avg))

    logger.info('Test result: r_mae / e_mae: {0:.3f}/{1:.3f}'.format(r_mae.avg, e_mae.avg))

if __name__ == '__main__':
    main()
