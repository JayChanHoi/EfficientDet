import os
import cv2
import sys
import random

import torch
import numpy as np
from torch.utils.data import Dataset

import albumentations as albu

from .utils import lamda_norm_tran

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

CLASSES = [
    'chair'
]

# note: if you used our download scripts, this should be right
class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(zip(CLASSES, range(len(CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1

            if not self.keep_difficult and difficult:
                continue

            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []

            for i, pt in enumerate(pts):
                cur_pt = float(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)

            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class VOCDetection(Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, transform=None, target_transform=VOCAnnotationTransform(), train=False):
        self.train = train
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.annotation_path = {}
        self.img_path = {}
        self.ids = list()

        for file_name in os.listdir(root):
            id = file_name.split('.')[0]
            if id not in self.ids:
                self.ids.append(id)

            if file_name.find('.jpg') >= 0:
                self.img_path[id] = file_name

            if file_name.find('.xml') >= 0:
                self.annotation_path[id] = file_name

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = ET.parse(os.path.join(self.root, self.annotation_path[img_id])).getroot()
        img = cv2.imread(os.path.join(self.root, self.img_path[img_id]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img.shape
        target = self.target_transform(target, width, height)
        target = np.array(target, dtype=np.float)
        bbox = target[:, :4]
        labels = target[:, 4]
        sample = {'image': img, 'bboxes':list(bbox), 'label':list(labels)}

        if self.transform is not None:
            if self.train:
                sample = self.transform(True, **sample)
            else:
                sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.ids)

    def num_classes(self):
        return len(CLASSES)

    def label_to_name(self, label):
        return CLASSES[label]

    def load_img_path(self, index):
        img_id = self.ids[index]

        return self.img_path[img_id]

    def load_annotations(self, index):
        img_id = self.ids[index]
        annotation = ET.parse(os.path.join(self.root, self.annotation_path[img_id])).getroot()
        gt = self.target_transform(annotation, 1, 1)
        gt = np.array(gt)
        return gt

def collater_train(data):
    imgs = [s['image'] for s in data]
    annots_np = [np.concatenate([np.asarray(s['bboxes']).reshape(-1, 4), np.asarray(s['label']).reshape(-1, 1)], axis=1) for s in data]
    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots_np)

    if max_num_annots > 0:

        annots_padded = torch.ones((len(annots_np), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots_np):
                if annot.shape[0] > 0:
                    annots_padded[idx, :annot.shape[0], :] = torch.from_numpy(annot)
    else:
        annots_padded = torch.ones((len(annots_np), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'image': imgs, 'annots': annots_padded}

def collater_test(data):
    imgs = [s['image'] for s in data]
    annots_np = [np.concatenate([np.asarray(s['bboxes']).reshape(-1, 4), np.asarray(s['label']).reshape(-1, 1)], axis=1) for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots_np)

    if max_num_annots > 0:

        annots_padded = torch.ones((len(annots_np), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots_np):
                if annot.shape[0] > 0:
                    annots_padded[idx, :annot.shape[0], :] = torch.from_numpy(annot)
    else:
        annots_padded = torch.ones((len(annots_np), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'image': imgs, 'annots': annots_padded, 'scale': scales}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, common_size=512):
        self.common_size = common_size

    def __call__(self, sample):
        image, bbox, label = sample['image'], np.asarray(sample['bboxes']), np.asarray(sample['label'])
        height, width, _ = image.shape
        if height > width:
            scale = self.common_size / height
            resized_height = self.common_size
            resized_width = int(width * scale)
        else:
            scale = self.common_size / width
            resized_height = int(height * scale)
            resized_width = self.common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((self.common_size, self.common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        bbox[:, :4] *= scale

        return {'image': new_image, 'bboxes': bbox, 'label':label, 'scale': scale}

def train_transform(width=512, height=512, min_area=0.0, min_visibility=0.0, lamda_norm=False):
        list_transforms = []
        augment = albu.Compose([
            albu.OneOf(
                [
                    albu.RandomSizedBBoxSafeCrop(p=1.0, height=height, width=width),
                    albu.HorizontalFlip(p=1.0),
                    albu.VerticalFlip(p=1.0),
                    albu.RandomRotate90(p=1.0),
                    albu.NoOp(p=1.0)
                ]
            ),
            albu.OneOf(
                [
                    albu.RandomBrightnessContrast(p=1.0),
                    albu.RandomGamma(p=1.0),
                    albu.NoOp(p=1.0)
                ]
            ),
            albu.OneOf(
                [
                    albu.MotionBlur(p=1.0),
                    albu.RandomFog(p=1.0),
                    albu.RandomRain(p=1.0),
                    albu.CLAHE(p=1.0),
                    albu.ToGray(p=1.0),
                    albu.NoOp(p=1.0)
                ]
            )
        ])
        list_transforms.extend([augment])

        if lamda_norm:
            list_transforms.extend([albu.Lambda(image=lamda_norm_tran)])
        else:
            list_transforms.extend([albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255., p=1.0)])
        list_transforms.extend([albu.Resize(height=height, width=width, p=1.0)])

        return albu.Compose(list_transforms, bbox_params=albu.BboxParams(format='pascal_voc', min_area=min_area,
                                                                         min_visibility=min_visibility,
                                                                         label_fields=['label']))

class Normalizer(object):

    def __init__(self, lamda_norm=False, grey_p=0.0):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])
        self.lamda_norm = lamda_norm
        self.grey_p = grey_p

    def __call__(self, sample):
        if random.random() <= self.grey_p:
            image, bbox, label = cv2.cvtColor(sample['image'], cv2.COLOR_RGB2GRAY), np.asarray(sample['bboxes']), np.asarray(sample['label'])
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image, bbox, label = sample['image'], np.asarray(sample['bboxes']), np.asarray(sample['label'])
        image = image.astype(np.float32) / 255.

        if self.lamda_norm:
            return {'image': (image.astype(np.float32)) * 2.0 - 1.0, 'bboxes': bbox, 'label':label}
        else:
            return {'image': ((image.astype(np.float32) - self.mean) / self.std), 'bboxes': bbox, 'label':label}
