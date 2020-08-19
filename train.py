import os
import argparse
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchcontrib.optim import SWA

from .src.dataset import Resizer, Normalizer, train_transform, collater_train, collater_test, VOCDetection
from .src.model import EfficientDet
from .src.model import resume, Anchors
from .src.loss import FocalLoss
from .src.eval import evaluate

from tensorboardX import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm

MODEL_MAP = {
    'efficientdet-d0': 'efficientnet-b0',
    'efficientdet-d1': 'efficientnet-b1',
    'efficientdet-d2': 'efficientnet-b2',
    'efficientdet-d3': 'efficientnet-b3',
    'efficientdet-d4': 'efficientnet-b4',
    'efficientdet-d5': 'efficientnet-b5',
    'efficientdet-d6': 'efficientnet-b6',
    'efficientdet-d7': 'efficientnet-b6',
}

EFFICIENTDET = {
    'efficientdet-d0': {'input_size': 512,
                        'backbone': 'B0',
                        'W_bifpn': 64,
                        'D_bifpn': 2,
                        'D_class': 3},
    'efficientdet-d1': {'input_size': 640,
                        'backbone': 'B1',
                        'W_bifpn': 88,
                        'D_bifpn': 3,
                        'D_class': 3},
    'efficientdet-d2': {'input_size': 768,
                        'backbone': 'B2',
                        'W_bifpn': 112,
                        'D_bifpn': 4,
                        'D_class': 3},
    'efficientdet-d3': {'input_size': 896,
                        'backbone': 'B3',
                        'W_bifpn': 160,
                        'D_bifpn': 5,
                        'D_class': 4},
    'efficientdet-d4': {'input_size': 1024,
                        'backbone': 'B4',
                        'W_bifpn': 224,
                        'D_bifpn': 6,
                        'D_class': 4},
    'efficientdet-d5': {'input_size': 1280,
                        'backbone': 'B5',
                        'W_bifpn': 288,
                        'D_bifpn': 7,
                        'D_class': 4},
    'efficientdet-d6': {'input_size': 1408,
                        'backbone': 'B6',
                        'W_bifpn': 384,
                        'D_bifpn': 8,
                        'D_class': 5},
    'efficientdet-d7': {'input_size': 1536,
                        'backbone': 'B6',
                        'W_bifpn': 384,
                        'D_bifpn': 8,
                        'D_class': 5},
}

def get_args():
    parser = argparse.ArgumentParser("EfficientDet: Scalable and Efficient Object Detection implementation")
    parser.add_argument('--train_dataset_root', default='train_set', help='train Dataset root directory path')
    parser.add_argument('--test_dataset_root', default='test_set', help='test Dataset root directory path')
    parser.add_argument("--network", type=str, default='efficientdet-d0', help="network model")
    parser.add_argument("--model_id", type=str, default='3', help="network model id")
    parser.add_argument("--model_version", type=str, default='3', help="model version")
    parser.add_argument("--batch_size", type=int, default=8, help="The number of images per batch")
    parser.add_argument("--num_worker", type=int, default=4, help="The number of worker for dataloader")
    parser.add_argument("--gpu", type=int, default=1, help="The gpu device id")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--smoothing_factor', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.00004)
    parser.add_argument('--glip_threshold', type=float, default=0.25)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--test_interval", type=int, default=2, help="Number of epoches between testing phases")
    parser.add_argument("--eval_interval", type=int, default=20, help="Number of epoches between eval phases")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--log_path", type=str, default="tensorboard/signatrix_efficientdet_coco")
    parser.add_argument("--checkpoint_root_dir", type=str, default="checkpoint_dir", help="checkpoint directory path")
    parser.add_argument("--resume", type=str, default=None, help="resume checkpoint path")
    parser.add_argument("--score_threshold", type=float, default=0.5, help="score threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")
    parser.add_argument("--from_pretrain", type=bool, default=False, help="transfer learning or not")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        # num_gpus = torch.cuda.device_count()
        device = 'cuda'
        torch.cuda.manual_seed(123)
        num_gpus = 1
    else:
        num_gpus = 1
        device = 'cpu'
        torch.manual_seed(123)

    training_params = {
        "batch_size": opt.batch_size * num_gpus,
        "shuffle": True,
        "drop_last": True,
        "collate_fn": collater_train,
        "num_workers": opt.num_worker,
        "pin_memory": True
    }

    test_params = {
        "batch_size": opt.batch_size * num_gpus,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": collater_test,
        "num_workers": opt.num_worker,
        "pin_memory": True
    }

    train_dataset = VOCDetection(
        train=True,
        root=opt.train_dataset_root,
        transform=train_transform(width=EFFICIENTDET[opt.network]['input_size'], height=EFFICIENTDET[opt.network]['input_size'], lamda_norm=False)
    )

    test_dataset = VOCDetection(
        train=False,
        root=opt.test_dataset_root,
        transform=transforms.Compose([Normalizer(lamda_norm=False, grey_p=0.0), Resizer(EFFICIENTDET[opt.network]['input_size'])])
    )

    test_dataset_grey = VOCDetection(
        train=False,
        root=opt.test_dataset_root,
        transform=transforms.Compose([Normalizer(lamda_norm=False, grey_p=1.0), Resizer(EFFICIENTDET[opt.network]['input_size'])])
    )

    train_generator = DataLoader(train_dataset, **training_params)
    test_generator = DataLoader(test_dataset, **test_params)
    test_grey_generator = DataLoader(test_dataset_grey, **test_params)

    network_id = int(''.join(filter(str.isdigit, opt.network)))
    loss_func = FocalLoss(alpha=opt.alpha, gamma=opt.gamma, smoothing_factor=opt.smoothing_factor)
    model = EfficientDet(
        MODEL_MAP[opt.network],
        image_size=[EFFICIENTDET[opt.network]['input_size'], EFFICIENTDET[opt.network]['input_size']],
        num_classes=train_dataset.num_classes(),
        compound_coef=network_id,
        num_anchors=9,
        advprop=True,
        from_pretrain=opt.from_pretrain
    )
    anchors_finder = Anchors()

    model.to(device)

    if opt.resume is not None:
        _ = resume(model, device, opt.resume)

    model = nn.DataParallel(model)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)

    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.checkpoint_root_dir):
        os.makedirs(opt.checkpoint_root_dir)

    writer = SummaryWriter(opt.log_path)

    base_optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, amsgrad=True)
    # optimizer = base_optimizer
    optimizer = SWA(base_optimizer, swa_start=10, swa_freq=5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.99
    )

    if opt.resume is not None:
        model.eval()

        loss_regression_ls = []
        loss_classification_ls = []
        with torch.no_grad():
            for iter, data in enumerate(tqdm(test_generator)):
                if torch.cuda.is_available():
                    anchors = anchors_finder(data['image'].cuda().float())
                    classification, regression = model(data['image'].cuda().float())
                    cls_loss, reg_loss = loss_func(classification, regression, anchors, data['annots'].cuda())
                else:
                    anchors = anchors_finder(data['image'].float())
                    classification, regression = model(data['image'].float())
                    cls_loss, reg_loss = loss_func(classification, regression, anchors, data['annots'])

                cls_loss = cls_loss.sum()
                reg_loss = reg_loss.sum()

                loss_classification_ls.append(float(cls_loss))
                loss_regression_ls.append(float(reg_loss))

        cls_loss = np.sum(loss_classification_ls) / test_dataset.__len__()
        reg_loss = np.sum(loss_regression_ls) / test_dataset.__len__()
        loss = (reg_loss + cls_loss) / 2

        writer.add_scalars('Total_loss', {'test': loss}, 0)
        writer.add_scalars('Regression_loss', {'test': reg_loss}, 0)
        writer.add_scalars('Classfication_loss (focal loss)', {'test': cls_loss}, 0)

        print(
            'Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                0, opt.num_epochs, cls_loss, reg_loss, np.mean(loss)
            )
        )

        mAP_1, _ = evaluate(test_generator, model, iou_threshold=0.5, score_threshold=0.5)
        mAP_5, _ = evaluate(test_generator, model, iou_threshold=0.75, score_threshold=0.1)

        writer.add_scalars('mAP', {'score threshold 0.5; iou threshold {}'.format(0.5): mAP_1, }, 0)
        writer.add_scalars('mAP', {'score threshold 0.1 ; iou threshold {}'.format(0.75): mAP_5, }, 0)

        mAP_1_grey, _ = evaluate(test_grey_generator, model, iou_threshold=0.5, score_threshold=0.5)
        mAP_5_grey, _ = evaluate(test_grey_generator, model, iou_threshold=0.75, score_threshold=0.1)

        writer.add_scalars('mAP', {'grey: True; score threshold 0.5; iou threshold {}'.format(0.5): mAP_1_grey, }, 0)
        writer.add_scalars('mAP', {'grey: True; score threshold 0.1 ; iou threshold {}'.format(0.75): mAP_5_grey, }, 0)

    model.train()

    num_iter_per_epoch = len(train_generator)
    train_iter = 0
    best_eval_loss = 10.0
    for epoch in range(opt.num_epochs):
        epoch_loss = []
        bn_update_data_list = []
        progress_bar = tqdm(train_generator)
        for iter, data in enumerate(progress_bar):
            scheduler.step(epoch + iter / train_generator.__len__())
            optimizer.zero_grad()

            if torch.cuda.is_available():
                if iter == 0:
                    bn_update_data_list.append(data['image'].float())
                anchors = anchors_finder(data['image'].cuda().float())
                classification, regression = model(data['image'].cuda().float())
                cls_loss, reg_loss = loss_func(classification, regression, anchors, data['annots'].cuda())
            else:
                if iter == 0:
                    bn_update_data_list.append(data['image'].float())
                anchors = anchors_finder(data['image'].float())
                classification, regression = model(data['image'].float())
                cls_loss, reg_loss = loss_func(classification, regression, anchors, data['annots'])

            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()
            loss = (reg_loss + cls_loss) / 2

            if loss == 0:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.glip_threshold)
            optimizer.step()
            epoch_loss.append(float(loss))
            total_loss = np.mean(epoch_loss)
            train_iter += 1

            progress_bar.set_description(
                'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. loss: {:.5f} Total loss: {:.5f}'.format(
                    epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss, reg_loss, loss, total_loss
                )
            )
            writer.add_scalars('Total_loss', {'train':total_loss}, train_iter)
            writer.add_scalars('Regression_loss', {'train':reg_loss}, train_iter)
            writer.add_scalars('Classfication_loss (focal loss)',{'train':cls_loss}, train_iter)

        if (epoch + 1) % opt.test_interval == 0 and epoch + 1 >= 0:

            loss_regression_ls = []
            loss_classification_ls = []
            optimizer.swap_swa_sgd()
            optimizer.bn_update(bn_update_data_list, model)
            model.eval()

            with torch.no_grad():
                for iter, data in enumerate(tqdm(test_generator)):

                    if torch.cuda.is_available():
                        anchors = anchors_finder(data['image'].cuda().float())
                        classification, regression = model(data['image'].cuda().float())
                        cls_loss, reg_loss = loss_func(classification, regression, anchors, data['annots'].cuda())

                    else:
                        anchors = anchors_finder(data['image'].float())
                        classification, regression = model(data['image'].float())
                        cls_loss, reg_loss = loss_func(classification, regression, anchors, data['annots'])

                    cls_loss = cls_loss.sum()
                    reg_loss = reg_loss.sum()

                    loss_classification_ls.append(float(cls_loss))
                    loss_regression_ls.append(float(reg_loss))

            cls_loss = np.sum(loss_classification_ls) / test_dataset.__len__()
            reg_loss = np.sum(loss_regression_ls) / test_dataset.__len__()
            loss = (reg_loss + cls_loss) / 2

            writer.add_scalars('Total_loss', {'test': loss}, train_iter)
            writer.add_scalars('Regression_loss', {'test': reg_loss}, train_iter)
            writer.add_scalars('Classfication_loss (focal loss)', {'test': cls_loss}, train_iter)

            print(
                'Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                    epoch + 1, opt.num_epochs, cls_loss, reg_loss, np.mean(loss)
                )
            )

            if 0 < loss < best_eval_loss and not (epoch + 1) % opt.eval_interval == 0:
                best_eval_loss = loss

                mAP_1, _ = evaluate(test_generator, model, iou_threshold=0.5, score_threshold=0.5)
                mAP_5, _ = evaluate(test_generator, model, iou_threshold=0.75, score_threshold=0.1)

                writer.add_scalars('mAP', {'score threshold 0.5; iou threshold {}'.format(0.5): mAP_1, }, train_iter)
                writer.add_scalars('mAP', {'score threshold 0.1 ; iou threshold {}'.format(0.75): mAP_5, }, train_iter)

                mAP_1_grey, _ = evaluate(test_grey_generator, model, iou_threshold=0.5, score_threshold=0.5)
                mAP_5_grey, _ = evaluate(test_grey_generator, model, iou_threshold=0.75, score_threshold=0.1)

                writer.add_scalars('mAP', {'grey: True; score threshold 0.5; iou threshold {}'.format(0.5): mAP_1_grey, }, train_iter)
                writer.add_scalars('mAP', {'grey: True; score threshold 0.1 ; iou threshold {}'.format(0.75): mAP_5_grey, }, train_iter)

                if torch.cuda.device_count() > 1:
                    checkpoint_dict = {'epoch': epoch + 1, 'state_dict': model.module.state_dict()}
                    torch.save(
                        checkpoint_dict,
                        os.path.join(
                            opt.checkpoint_root_dir,
                            '{}_checpoint_epoch_{}_loss_{}.pth'.format('model_v1', epoch + 1, loss)
                        )
                    )
                else:
                    checkpoint_dict = {'epoch': epoch + 1, 'state_dict': model.state_dict()}
                    torch.save(
                        checkpoint_dict,
                        os.path.join(
                            opt.checkpoint_root_dir,
                            '{}_checpoint_epoch_{}_loss_{}.pth'.format('model_v1', epoch + 1, loss)
                        )
                    )

            if (epoch + 1) % opt.eval_interval == 0 and epoch + 1 >= 0:
                mAP_1, _ = evaluate(test_generator, model, iou_threshold=0.5, score_threshold=0.5)
                mAP_5, _ = evaluate(test_generator, model, iou_threshold=0.75, score_threshold=0.1)

                writer.add_scalars('mAP', {'score threshold 0.5; iou threshold {}'.format(0.5): mAP_1, }, train_iter)
                writer.add_scalars('mAP', {'score threshold 0.1 ; iou threshold {}'.format(0.75): mAP_5, }, train_iter)

                mAP_1_grey, _ = evaluate(test_grey_generator, model, iou_threshold=0.5, score_threshold=0.5)
                mAP_5_grey, _ = evaluate(test_grey_generator, model, iou_threshold=0.75, score_threshold=0.1)

                writer.add_scalars('mAP', {'grey: True; score threshold 0.5; iou threshold {}'.format(0.5): mAP_1_grey, }, train_iter)
                writer.add_scalars('mAP', {'grey: True; score threshold 0.1 ; iou threshold {}'.format(0.75): mAP_5_grey, }, train_iter)

                if torch.cuda.device_count() > 1:
                    checkpoint_dict = {'epoch': epoch + 1, 'state_dict': model.module.state_dict()}
                    torch.save(
                        checkpoint_dict,
                        os.path.join(
                            opt.checkpoint_root_dir,
                            '{}_checpoint_epoch_{}_loss_{}.pth'.format('model_v1', epoch + 1, loss)
                        )
                    )
                else:
                    checkpoint_dict = {'epoch': epoch + 1, 'state_dict': model.state_dict()}
                    torch.save(
                        checkpoint_dict,
                        os.path.join(
                            opt.checkpoint_root_dir,
                            '{}_checpoint_epoch_{}_loss_{}.pth'.format('model_v1', epoch + 1, loss)
                        )
                    )

            optimizer.swap_swa_sgd()

            model.train()

        scheduler.step()

    writer.close()

if __name__ == "__main__":
    # model_list = {'2':8}
    opt = get_args()
    # for model_id, batch_size in model_list.items():
    opt.network = 'efficientdet-d{}'.format(opt.model_id)
    opt.log_path = 'tensorboard/d{}_v{}'.format(opt.model_id, opt.model_version)
    opt.checkpoint_root_dir = 'checkpoint_dir/d{}_v{}'.format(opt.model_id, opt.model_version)
    train(opt)
