# EfficientDet
This is a repo contain a pytorch verison of EfficientDet, a STOA one stage model architecture for object detection.

It contain the training module and video test module.

## remark:
1. currently, only support VOC format dataset.
2. included Storchastic weight average.
3. included the auto mAP evaluation module. If practitioners want to have a tailormake nms parameters,
   you can just direct change it in the train.py.
4. support a family of efficientNet backbone. But not yet confirm the performance of those larger than B4 as they require
   large GPu memory for training.