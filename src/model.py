import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet as EffNet
from functools import partial

from .utils import *

def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std
        if torch.cuda.is_available():
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    def forward(self, boxes, deltas):

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes

class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes

class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image):

        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        anchors = torch.from_numpy(all_anchors.astype(np.float32))
        if torch.cuda.is_available():
            anchors = anchors.cuda()
        return anchors

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class ExtendConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, image_size):
        super(ExtendConvBlock, self).__init__()
        self._expand_conv = Conv2dStaticSamePadding(
            in_channels=input_channel,
            out_channels=int(input_channel * 6),
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False,
            image_size=image_size
        )
        self._bn_0 = torch.nn.BatchNorm2d(
            input_channel * 6,
            eps=0.001,
            momentum=0.010000000000000009,
            affine=True,
            track_running_stats=True
        )
        self._depthwise_conv = Conv2dStaticSamePadding(
            in_channels=int(input_channel * 6),
            out_channels=int(input_channel * 6),
            kernel_size=(5, 5),
            stride=[2, 2],
            groups=int(input_channel * 6),
            bias=False,
            image_size=image_size
        )
        self._bn_1 = torch.nn.BatchNorm2d(
            input_channel * 6,
            eps=0.001,
            momentum=0.010000000000000009,
            affine=True,
            track_running_stats=True
        )
        self._se_reduce = Conv2dStaticSamePadding(
            in_channels=int(input_channel * 6),
            out_channels=int(input_channel / 4),
            kernel_size=(1, 1),
            stride=[1, 1],
            image_size=image_size
        )
        self._se_expand = Conv2dStaticSamePadding(
            in_channels=int(input_channel / 4),
            out_channels=int(input_channel * 6),
            kernel_size=(1, 1),
            stride=(1, 1),
            image_size=image_size
        )
        self._project_conv = Conv2dStaticSamePadding(
            in_channels=int(input_channel * 6),
            out_channels=output_channel,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False,
            image_size=image_size
        )
        self._bn2 = torch.nn.BatchNorm2d(
            output_channel,
            eps=0.001,
            momentum=0.010000000000000009,
            affine=True,
            track_running_stats=True
        )

        self._swish = MemoryEfficientSwish()

        self.block = nn.Sequential(
            self._expand_conv,
            self._bn_0,
            self._depthwise_conv,
            self._bn_1,
            self._se_reduce,
            self._se_expand,
            self._project_conv,
            self._bn2,
            self._swish
        )

    def forward(self, x):
        x = self.block(x)

        return x

class ConvBlock(nn.Module):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, groups=num_channels),
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=num_channels, momentum=0.9997, eps=4e-5),
            nn.ReLU()
        )

    def forward(self, input):
        return self.conv(input)

class BiFPN(nn.Module):
    def __init__(self, num_channels, epsilon=1e-4):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # Conv layers
        self.conv6_up = ConvBlock(num_channels)
        self.conv5_up = ConvBlock(num_channels)
        self.conv4_up = ConvBlock(num_channels)
        self.conv3_up = ConvBlock(num_channels)
        self.conv4_down = ConvBlock(num_channels)
        self.conv5_down = ConvBlock(num_channels)
        self.conv6_down = ConvBlock(num_channels)
        self.conv7_down = ConvBlock(num_channels)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = nn.MaxPool2d(kernel_size=2)
        self.p5_downsample = nn.MaxPool2d(kernel_size=2)
        self.p6_downsample = nn.MaxPool2d(kernel_size=2)
        self.p7_downsample = nn.MaxPool2d(kernel_size=2)

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2))
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2))
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2))
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2))
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3))
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3))
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3))
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2))
        self.p7_w2_relu = nn.ReLU()

    def forward(self, input):
        """
            P7_0 -------------------------- P7_2 -------->

            P6_0 ---------- P6_1 ---------- P6_2 -------->

            P5_0 ---------- P5_1 ---------- P5_2 -------->

            P4_0 ---------- P4_1 ---------- P4_2 -------->

            P3_0 -------------------------- P3_2 -------->
        """

        # P3_0, P4_0, P5_0, P6_0 and P7_0
        p3_in, p4_in, p5_in, p6_in, p7_in = input

        # P7_0 to P7_2
        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in))
        # Weights for P5_0 and P6_0 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up))
        # Weights for P4_0 and P5_0 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up))

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out))
        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out))
        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out))
        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out))

        return p3_out, p4_out, p5_out, p6_out, p7_out

class ClassBoxNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers):
        super(ClassBoxNet, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        class_layers = []
        box_layers = []
        for _ in range(num_layers):
            class_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            box_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))

            class_layers.append(nn.ReLU(True))
            box_layers.append(nn.ReLU(True))

        self.class_layers = nn.Sequential(*class_layers)
        self.class_header = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.class_act = nn.Sigmoid()

        self.box_layers = nn.Sequential(*box_layers)
        self.box_header = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        class_inputs = self.class_layers(inputs)
        class_inputs = self.class_header(class_inputs)
        class_inputs = self.class_act(class_inputs)
        class_inputs = class_inputs.permute(0, 2, 3, 1)
        class_output = class_inputs.contiguous().view(
            class_inputs.shape[0],
            class_inputs.shape[1],
            class_inputs.shape[2],
            self.num_anchors,
            self.num_classes
        )
        class_output = class_output.contiguous().view(class_output.shape[0], -1, self.num_classes)

        box_inputs = self.box_layers(inputs)
        box_inputs = self.box_header(box_inputs)
        box_output = box_inputs.permute(0, 2, 3, 1)
        box_output = box_output.contiguous().view(box_output.shape[0], -1, 4)

        return class_output, box_output

class EfficientNet(nn.Module):
    def __init__(self, network, advprop=False, from_pretrain=False):
        super(EfficientNet, self).__init__()
        if from_pretrain:
            self.model = EffNet.from_pretrained(network, advprop=advprop)
        else:
            self.model = EffNet.from_name(network)
        del self.model._conv_head
        del self.model._bn1
        del self.model._avg_pooling
        del self.model._dropout
        del self.model._fc

    def find_feature_maps_channel(self):
        test_input = torch.rand([1, 3, 512, 910])
        feature_maps = self.extract_feature_list(test_input)

        del test_input
        return [feature_map.shape[1] for feature_map in feature_maps]

    def extract_feature_list(self, x):
        x = self.model._swish(self.model._bn0(self.model._conv_stem(x)))
        feature_maps = []
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            feature_maps.append(x)

        td_feature_maps = {}
        for feature_map in feature_maps:
            if feature_map.shape[2] not in td_feature_maps.keys():
                td_feature_maps[feature_map.shape[2]] = [feature_map]
            else:
                td_feature_maps[feature_map.shape[2]].append(feature_map)

        refine_feature_maps = [value[-1] for value in td_feature_maps.values()]

        return refine_feature_maps[2], refine_feature_maps[3], refine_feature_maps[4]

class EfficientDet(nn.Module):
    def __init__(self, network, image_size, num_anchors=9, num_classes=20, compound_coef=0, advprop=False, from_pretrain=False):
        super(EfficientDet, self).__init__()
        self.compound_coef = compound_coef
        self.backbone_net = EfficientNet(network, advprop, from_pretrain)

        self.num_channels = [64, 88, 112, 160, 224, 288, 384, 384][self.compound_coef]
        network_channels = self.backbone_net.find_feature_maps_channel()

        self.conv3 = nn.Sequential(
            nn.Conv2d(network_channels[0], self.num_channels, kernel_size=1, stride=1, padding=0),
            # nn.Dropout2d(p=0.1)
            # nn.BatchNorm2d(self.num_channels, track_running_stats=True, affine=True),
            # nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(network_channels[1], self.num_channels, kernel_size=1, stride=1, padding=0),
            # nn.Dropout2d(p=0.1)
            # nn.BatchNorm2d(self.num_channels, track_running_stats=True, affine=True),
            # nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(network_channels[2], self.num_channels, kernel_size=1, stride=1, padding=0),
            # nn.Dropout2d(p=0.1)

            # nn.BatchNorm2d(self.num_channels, track_running_stats=True, affine=True),
            # nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(int(network_channels[2] * 1.5), self.num_channels, kernel_size=1, stride=1, padding=0),
            # nn.Dropout2d(p=0.1)
            # nn.BatchNorm2d(self.num_channels, track_running_stats=True, affine=True),
            # nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(int(network_channels[2] * 1.5 * 1.5), self.num_channels, kernel_size=1, stride=1, padding=0),
            # nn.Dropout2d(p=0.1)
            # nn.BatchNorm2d(self.num_channels, track_running_stats=True, affine=True),
            # nn.ReLU()
        )

        self.extend_conv_6 = ExtendConvBlock(
            input_channel=network_channels[2],
            output_channel=int(network_channels[2] * 1.5),
            image_size=[int(image_size[0] / (2 ** 6)), int(image_size[1] / (2 ** 6))]
        )
        self.extend_conv_7 = ExtendConvBlock(
            input_channel=int(network_channels[2] * 1.5),
            output_channel=int(network_channels[2] * 1.5 * 1.5),
            image_size=[int(image_size[0] / (2 ** 7)), int(image_size[1] / (2 ** 7))]
        )

        self.bifpn = nn.Sequential(*[BiFPN(self.num_channels) for _ in range(min(2 + self.compound_coef, 8))])

        self.num_classes = num_classes

        self.class_box_network = ClassBoxNet(
            in_channels=self.num_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            num_layers=3 + self.compound_coef // 3
        )

        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.class_box_network.class_header.weight.data.fill_(0)
        self.class_box_network.class_header.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.class_box_network.box_header.weight.data.fill_(0)
        self.class_box_network.box_header.bias.data.fill_(0)


    def forward(self, inputs, train=True, score_threshold=0.5, iou_threshold=0.5):
        img_batch = inputs

        c = self.backbone_net.extract_feature_list(img_batch)
        c6 = self.extend_conv_6(c[2])
        c7 = self.extend_conv_7(c6)
        p3 = self.conv3(c[0])
        p4 = self.conv4(c[1])
        p5 = self.conv5(c[2])
        p6 = self.conv6(c6)
        p7 = self.conv7(c7)

        features = [p3, p4, p5, p6, p7]
        features = self.bifpn(features)

        box_features = []
        class_features = []
        for feature in features:
            class_feature, box_feature = self.class_box_network(feature)
            class_features.append(class_feature)
            box_features.append(box_feature)

        regression = torch.cat(box_features, dim=1)
        classification = torch.cat(class_features, dim=1)

        if train:
            return classification, regression
        else:
            anchors = self.anchors(img_batch)
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
            scores = torch.max(classification, dim=2, keepdim=True)[0]
            scores_over_thresh = (scores >= score_threshold)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], iou_threshold)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]
