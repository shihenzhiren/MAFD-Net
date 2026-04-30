from .resnet import resnet38, resnet110, resnet116, resnet14x2, resnet38x2, resnet110x2
from .resnet import resnet8x4, resnet14x4, resnet32x4, resnet38x4
from .vgg import vgg8_bn, vgg13_bn
from .mobilenetv2 import mobile_half, mobile_half_double
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2, ShuffleV2_1_5

from .resnet_imagenet import resnet18, resnet34, resnet50, wide_resnet50_2, resnext50_32x4d
from .resnet_imagenet import wide_resnet10_2, wide_resnet18_2, wide_resnet34_2
from .mobilenetv2_imagenet import mobilenet_v2
from .shuffleNetv2_imagenet import shufflenet_v2_x1_0

# 导入原始EfficientNet
from .efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7

# 导入修改后的EfficientNet-T系列
from .efficientnet_T import (
    efficientnet_b0_t, efficientnet_b1_t, efficientnet_b2_t,
    efficientnet_b3_t, efficientnet_b4_t, efficientnet_b5_t,
    efficientnet_b6_t, efficientnet_b7_t
)

model_dict = {
    # 原始模型
    'resnet38': resnet38,
    'resnet110': resnet110,
    'resnet116': resnet116,
    'resnet14x2': resnet14x2,
    'resnet38x2': resnet38x2,
    'resnet110x2': resnet110x2,
    'resnet8x4': resnet8x4,
    'resnet14x4': resnet14x4,
    'resnet32x4': resnet32x4,
    'resnet38x4': resnet38x4,
    'vgg8': vgg8_bn,
    'vgg13': vgg13_bn,
    'MobileNetV2': mobile_half,
    'MobileNetV2_1_0': mobile_half_double,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'ShuffleV2_1_5': ShuffleV2_1_5,

    # ImageNet模型
    'ResNet18': resnet18,
    'ResNet34': resnet34,
    'ResNet50': resnet50,
    'resnext50_32x4d': resnext50_32x4d,
    'ResNet10x2': wide_resnet10_2,
    'ResNet18x2': wide_resnet18_2,
    'ResNet34x2': wide_resnet34_2,
    'wrn_50_2': wide_resnet50_2,
    'MobileNetV2_Imagenet': mobilenet_v2,
    'ShuffleV2_Imagenet': shufflenet_v2_x1_0,

    # 原始EfficientNet系列
    'EfficientNetB0': efficientnet_b0,
    'EfficientNetB1': efficientnet_b1,
    'EfficientNetB2': efficientnet_b2,
    'EfficientNetB3': efficientnet_b3,
    'EfficientNetB4': efficientnet_b4,
    'EfficientNetB5': efficientnet_b5,
    'EfficientNetB6': efficientnet_b6,
    'EfficientNetB7': efficientnet_b7,

    # 新增带自注意力的EfficientNet-T系列
    'EfficientNetB0_t': efficientnet_b0_t,
    'EfficientNetB1_t': efficientnet_b1_t,
    'EfficientNetB2_t': efficientnet_b2_t,
    'EfficientNetB3_t': efficientnet_b3_t,
    'EfficientNetB4_t': efficientnet_b4_t,
    'EfficientNetB5_t': efficientnet_b5_t,
    'EfficientNetB6_t': efficientnet_b6_t,
    'EfficientNetB7_t': efficientnet_b7_t
}
