import argparse
import collections
import os
from datetime import datetime
import shutil

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, tuple_collater, TransformTwice, \
    Resizer, AspectRatioBasedSampler, TwoStreamBatchSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval
from retinanet import losses

from utils import ramps
from utils.logger import setup_logger

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.extend([ param_group['lr'] ])
    return lr

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # print ema_param.name
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def get_current_consistency_weight(epoch_num, consistency_weight, consistency_rampup):
    return consistency_weight * ramps.sigmoid_rampup(epoch_num, consistency_rampup)

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Root path to COCO-style data directory. The directory should contain both image and json data.')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=55)

    parser.add_argument('--load_pth', help='Path to a pre-trained RetinaNet weights')

    parser.add_argument('--ema_decay', help='Updating ratio of the teacher model', type=float, default=0.99)
    parser.add_argument('--consistency_weight', help='Max value of unsupervised loss weight', type=float, default=100.0)
    parser.add_argument('--consistency_rampup', help='Max step for rampup', type=int, default=20.0)

    parser.add_argument('--num_labeled_data', help='number of labeled data in the training set.', type=int)
    parser.add_argument('--num_data', help='number of unlabeled data in the training set.', type=int)

    parser.add_argument('--experiment_name', help='experiment name')

    parser = parser.parse_args(args)

    print("========> Experiment: {}".format(parser.experiment_name))

    checkpoint_dir = os.path.join("./checkpoints", parser.experiment_name)
    logger_dir = os.path.join("./logs", parser.experiment_name)
    if parser.experiment_name != 'debug':
        for dir_ in [checkpoint_dir, logger_dir]:
            if not os.path.isdir(dir_):
                os.makedirs(dir_)
    # Save the codes for current run if needed
    #    if not os.path.isdir(os.path.join(logger_dir, 'code-bk')):
    #       shutil.copytree('./', os.path.join(logger_dir, 'code-bk'),
    #                       ignore=shutil.ignore_patterns('checkpoints', 'viz', 'LICENSE', 'logs', '__pycache__',
    #                                                     '*pyc', '*ipynb', '*json', '*out', '*pth'))

    logger = setup_logger(parser.experiment_name, logger_dir)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='dataset1010_512_lu',
                                    transform=TransformTwice(transforms.Compose([Augmenter(), Normalizer(), Resizer()])))
        dataset_val   = CocoDataset(parser.coco_path, set_name='dataset1010_512_val',
                                    transform=transforms.Compose([Normalizer(), Resizer()]))
        dataset_test  = CocoDataset(parser.coco_path, set_name='dataset1010_512_test',
                                    transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Augmenter(), Normalizer(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

#    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    labeled_idxs = list(range(parser.num_labeled_data))
    unlabeled_idxs = list(range(parser.num_labeled_data, parser.num_data))
    labeled_batch, unlabeled_batch = 2, 2
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, labeled_batch, unlabeled_batch)
    dataloader_train = DataLoader(dataset_train, num_workers=8, collate_fn=tuple_collater, batch_sampler=batch_sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
    if dataset_test is not None:
        sampler_val = AspectRatioBasedSampler(dataset_test, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_test, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Uncomment the following two lines if you meet urlopen error
    # import ssl
    # ssl._create_default_https_context = ssl._create_unverified_context

    # Create the model
    load_from_url = False
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=load_from_url)
        ema_retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=False)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=load_from_url)
        ema_retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=False)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=load_from_url)
        ema_retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=False)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=load_from_url)
        ema_retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=False)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=load_from_url)
        ema_retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=False)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    retinanet.load_state_dict(torch.load(parser.load_pth).state_dict(), strict=False)
    ema_retinanet.load_state_dict(torch.load(parser.load_pth).state_dict(), strict=False)

    for param in ema_retinanet.parameters():
        param.detach_()
    ema_retinanet.eval()

    supervisedLoss     = losses.FocalLoss()    # the loss for supervised classification and regression
    globalLoss         = losses.GlobalClassificationLoss(pos_weight=None, with_logit=True) # global classification loss
    globalContrastLoss = losses.GlobalContrastiveLoss()     # global prototype loss
    softFocalLoss      = losses.ConsistencyLoss()  # soft focal loss

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()
            ema_retinanet = ema_retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    tick = datetime.now()
    logger.info("\nBegin training at: {}".format(tick.strftime("%Y-%m-%d %H:%M:%S")))

    global_step = 0

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        for iter_num, (data, ema_data) in enumerate(dataloader_train):
            #try:
            optimizer.zero_grad()

            classification, regression, anchors, annotations, global_output, cls_features, prototypes = retinanet(
                [data['img'].cuda().float(), data['annot'], global_step])
            with torch.no_grad():
                ema_classification, ema_res, ema_global_output = ema_retinanet(
                    ema_data['img'][labeled_batch:].cuda().float(), ema=True)

            classification_loss, regression_loss = supervisedLoss(classification[:labeled_batch],
                                                                  regression[:labeled_batch],
                                                                  anchors, annotations[:labeled_batch])

            classification_consistency_loss, regression_consistency_loss = softFocalLoss(
                classification[labeled_batch:], regression[labeled_batch:],
                anchors, ema_classification, ema_res, data['annot'][labeled_batch:])

            global_classification_loss = globalLoss(global_output, data['annot'])
            intra_loss, inter_loss = globalContrastLoss(cls_features, prototypes, data['annot'], global_step)

            consistency_weight = get_current_consistency_weight(epoch_num, parser.consistency_weight,
                                                                parser.consistency_rampup)

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            classification_consistency_loss = classification_consistency_loss.mean()
            regression_consistency_loss = regression_consistency_loss.mean()

            loss = classification_loss + regression_loss + classification_consistency_loss + \
                   regression_consistency_loss + global_classification_loss + intra_loss + inter_loss

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()

            # update ema retinanet
            update_ema_variables(retinanet, ema_retinanet, parser.ema_decay, global_step)
            global_step = global_step + 1

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            if iter_num % 10 == 0:
                logger.info(
                    'Epoch: {} | Iteration: {} | Cls loss: {:1.5f} | Reg loss: {:1.5f} | '
                    'G cls loss: {:1.5f} | Intra loss: {:1.5f} | Inter loss: {:1.5f} | '
                    'Cls Cons loss: {:1.5f} | Reg Cons loss: {:1.5f} | Running loss: {:1.5f} | lr: {}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss),
                        float(global_classification_loss),
                        intra_loss, inter_loss,
                        float(classification_consistency_loss),
                        float(regression_consistency_loss),
                        np.mean(loss_hist), get_learning_rate(optimizer)[0]))

            del classification_loss
            del regression_loss
            del classification_consistency_loss
            del regression_consistency_loss
            del intra_loss
            del inter_loss
            #except Exception as e:
            #    print(e)
            #    continue

        if (epoch_num % 5 == 0) and (epoch_num >=20):

            if parser.dataset == 'coco':
                metric_names = [
                    'AP', 'AP40', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10',
                    'AR100', 'ARs', 'ARm', 'ARl', 'p_AP', 'p_AP40'
                ]
                print('Evaluating val dataset')

                val_res, val_g_aucs, val_l_aucs = coco_eval.evaluate_coco(dataset_val, retinanet,
                                                                          experiment_name=parser.experiment_name)
                logger.info(
                    'VAL | Epoch: {}\n'.format(epoch_num) + '========= G AUC =========\n{}\n'.format(
                        sum(val_g_aucs) / len(val_g_aucs)) + '========= L AUC =========\n{}\n'.format(
                        sum(val_l_aucs) / len(val_l_aucs)) + ' '.join(
                        ['========= {} =========\n{}\n'.format(metric, val_res[metric]) for metric in
                         metric_names])
                )

                print('Evaluating test data')
                test_res, test_g_aucs, test_l_aucs = coco_eval.evaluate_coco(dataset_test, retinanet,
                                                                             experiment_name=parser.experiment_name)
                logger.info(
                    'TEST | Epoch: {}\n'.format(epoch_num) + '=== G AUC ===\n{}\n'.format(
                        sum(test_g_aucs) / len(test_g_aucs)) + '=== L AUC ===\n{}\n'.format(
                        sum(test_l_aucs) / len(test_l_aucs)) + ' '.join(
                        ['=== {} ===\n{}\n'.format(metric, test_res[metric]) for metric in
                         metric_names])
                )

            elif parser.dataset == 'csv' and parser.csv_val is not None:

                print('Evaluating dataset')

                mAP = csv_eval.evaluate(dataset_val, retinanet)

        # scheduler.step(np.mean(epoch_loss))
        if (epoch_num == 20):
            for g in optimizer.param_groups:
                if g['lr'] > 1e-6:
                    g['lr'] = 1e-6
        if (epoch_num == 45):
            for g in optimizer.param_groups:
                if g['lr'] > 1e-7:
                    g['lr'] = 1e-7

        torch.save(retinanet.module, checkpoint_dir+'/{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()

    torch.save(retinanet, checkpoint_dir+'/model_final.pt')

    tock = datetime.now()
    logger.info("\nEnd training at: {}".format(tock.strftime("%Y-%m-%d %H:%M:%S")))


if __name__ == '__main__':
    main()
