import argparse
import torch
import torch.nn as nn

from .src.config import COCO_CLASSES, colors
from .src.utils import resume
from .src.model import EfficientDet
import cv2
import numpy as np

CLASSES = [
    'hook'
]

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
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
    parser.add_argument("--score_threshold", type=float, default=0.01)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--resume", type=str, help='path of trained checkpoint')
    parser.add_argument("--model_id", type=int, default="id for the backbone network")
    parser.add_argument("--input", type=str, default="input video path")
    parser.add_argument("--output", type=str, default="output video path")

    args = parser.parse_args()
    return args

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def test(opt):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = 'cuda'
        torch.cuda.manual_seed(123)
    else:
        num_gpus = 1
        device = 'cpu'
        torch.manual_seed(123)

    network_id = int(opt.model_id)
    model = EfficientDet(
        MODEL_MAP['efficientdet-d{}'.format(opt.model_id)],
        image_size=[EFFICIENTDET['efficientdet-d{}'.format(opt.model_id)]['input_size'], EFFICIENTDET['efficientdet-d{}'.format(opt.model_id)]['input_size']],
        num_classes=1,
        compound_coef=network_id,
        num_anchors=9,
        advprop=True
    )

    if torch.cuda.is_available():
        model.cuda()

    _ = resume(model, device, opt.resume)

    model.eval()
    image_size = EFFICIENTDET['efficientdet-d{}'.format(opt.model_id)]['input_size']

    cap = cv2.VideoCapture(opt.input)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(opt.output, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while cap.isOpened():
        flag, image = cap.read()
        output_image = np.copy(image)
        if flag:
            image = adjust_gamma(image, gamma = 1.0)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        else:
            break

        height, width = image.shape[:2]
        image = image.astype(np.float32) / 255.
        image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
        image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        if height > width:
            scale = image_size / height
            resized_height = image_size
            resized_width = int(width * scale)
        else:
            scale = image_size / width
            resized_height = int(height * scale)
            resized_width = image_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((image_size, image_size, 3))
        new_image[0:resized_height, 0:resized_width] = image
        new_image = torch.from_numpy(new_image)

        if torch.cuda.is_available():
            new_image = new_image.cuda()

        with torch.no_grad():
            if torch.cuda.is_available():
                scores, labels, boxes = model(
                    new_image.permute(2, 0, 1).cuda().float().unsqueeze(dim=0),
                    train=False,
                    score_threshold=opt.score_threshold,
                    iou_threshold=opt.iou_threshold
                )
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                boxes = boxes.cpu().numpy()
            else:
                scores, labels, boxes = model(
                    new_image.permute(2, 0, 1).float().unsqueeze(dim=0),
                    train=False,
                    score_threshold=opt.score_threshold,
                    iou_threshold=opt.iou_threshold
                )
            boxes /= scale
            print(scores)

        #
        if boxes.shape[0] == 0:
            out.write(output_image)
        else:
            # for box_id in range(boxes.shape[0]):
            for box_id in range(boxes.shape[0]):
                pred_prob = float(scores[box_id])
                if pred_prob < opt.score_threshold:
                    continue
                else:
                    pred_label = int(labels[box_id])
                    xmin, ymin, xmax, ymax = boxes[box_id, :]
                    color = colors[pred_label]
                    cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                    text_size = \
                    cv2.getTextSize(CLASSES[pred_label] + ' : %.2f' % pred_prob, cv2.FONT_HERSHEY_PLAIN, 1, 1)[
                        0]
                    cv2.putText(
                        output_image,
                        CLASSES[pred_label],
                        (int(xmin), int(ymin + text_size[1] + 4)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255),
                        1
                    )
            out.write(output_image)

    cap.release()
    out.release()


if __name__ == "__main__":
    opt = get_args()
    test(opt)
