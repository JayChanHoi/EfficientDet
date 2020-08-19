# EfficientDet
This is a repo contain a pytorch verison of EfficientDet, a STOA one stage model architecture for object detection.

It contain the training module and video test module.

## Remark:
1. Currently, only support VOC format dataset.
2. Included Storchastic weight average.
3. Included the auto mAP evaluation module. If practitioners want to have a tailormake nms parameters,
   you can just direct change it in the train.py.
4. Support a family of efficientNet backbone. But not yet confirm the performance of those larger than B4 as they require
   large GPu memory for training.
5. Included data augmentation in the dataset module. Practitioner can make some adjustment for it if needed.
   Current one is optimized by my own use case. 
6. Here we use the backbone network from efficentNet-pytorch as the detection network backbone. 
7. Can use pretrain weights of the backbone network for transfer learning or train from sratch.