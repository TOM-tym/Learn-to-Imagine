"""Taken & slightly modified from:
* https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True, alternative_BN=False):
        super(BasicBlock, self).__init__()
        BN = AlternativeBN if alternative_BN else nn.BatchNorm2d
        self.alternative_BN = alternative_BN
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x, if_mixed=False):
        identity = x

        out = self.conv1(x)

        if self.alternative_BN:
            out = self.bn1(out, if_mixed=if_mixed)
        else:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.alternative_BN:
            out = self.bn2(out, if_mixed=if_mixed)
        else:
            out = self.bn2(out)

        if self.downsample is not None:
            if self.alternative_BN:
                identity = self.downsample(x, if_mixed=if_mixed)
            else:
                identity = self.downsample(x)

        out += identity

        if self.last_relu:
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.last_relu:
            out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block,
            layers,
            zero_init_residual=True,
            nf=64,
            last_relu=True,
            initial_kernel=3,
            **kwargs
    ):
        super(ResNet, self).__init__()

        self.last_relu = last_relu
        self.inplanes = nf
        # self.conv1 = nn.Conv2d(3, nf, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, nf, kernel_size=initial_kernel, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nf)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage_1 = self._make_layer(block, 1 * nf, layers[0])
        self.stage_2 = self._make_layer(block, 2 * nf, layers[1], stride=2)
        self.stage_3 = self._make_layer(block, 4 * nf, layers[2], stride=2)
        self.stage_4 = self._make_layer(block, 8 * nf, layers[3], stride=2, last=True, alternative_BN=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_dim = 8 * nf * block.expansion
        print("Features dimension is {}.".format(self.out_dim))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    if isinstance(m.bn2, nn.BatchNorm2d):
                        nn.init.constant_(m.bn2.weight, 0)
                    elif isinstance(m.bn2, nn.Module):
                        for mm in m.bn2.modules():
                            if isinstance(mm, nn.BatchNorm2d):
                                nn.init.constant_(mm.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last=False, alternative_BN=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = MyDownSampleBlock(inplanes=self.inplanes, planes=planes, expansion=block.expansion,
                                           stride=stride, alternative_BN=alternative_BN)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, alternative_BN=alternative_BN))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            if i == blocks - 1 or last:
                layers.append(block(self.inplanes, planes, last_relu=False, alternative_BN=alternative_BN))
            else:
                layers.append(block(self.inplanes, planes, last_relu=self.last_relu, alternative_BN=alternative_BN))

        return MySequential(*layers)

    @property
    def last_block(self):
        return self.stage_4

    @property
    def last_conv(self):
        return self.stage_4[-1].conv2

    def forward(self, x, pre_pass=False, if_mixed=False):
        if not pre_pass:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x_1 = self.stage_1(x)
            x_2 = self.stage_2(self.end_relu(x_1))
            x_3 = self.stage_3(self.end_relu(x_2))
        else:
            x_1 = x
            x_2 = x
            x_3 = x
        x_4 = self.stage_4(self.end_relu(x_3), if_mixed=if_mixed)

        raw_features = self.end_features(x_4)
        features = self.end_features(F.relu(x_4, inplace=False))

        return {
            "raw_features": raw_features,
            "features": features,
            "attention": [x_1, x_2, x_3, x_4],
            'stage2_feature_map': x_3
        }

    def end_features(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def end_relu(self, x):
        if hasattr(self, "last_relu") and self.last_relu:
            return F.relu(x)
        return x

    def freeze(self, trainable=False, model="all"):
        if model == "all":
            model = self
        elif model == "stage4":
            model = self.stage_4
        elif model == "stage1":
            model = self.stage_1
        elif model == "stage2":
            model = self.stage_2
        elif model == "stage3":
            model = self.stage_3
        elif model == "conv1":
            model = self.conv1
        elif model == "bn1":
            model = self.bn1
        else:
            assert False, model
        if not isinstance(model, nn.Module):
            return self
        for param in model.parameters():
            param.requires_grad = trainable
        if not trainable:
            model.eval()
        else:
            model.train()
        return self

    def init_fake_BN(self):
        stage4 = self.stage_4
        for i in stage4.modules():
            if isinstance(i, AlternativeBN):
                i.bn2.load_state_dict(i.bn1.state_dict())

class AlternativeBN(nn.Module):
    def __init__(self, bn_config):
        super(AlternativeBN, self).__init__()
        self.bn1 = nn.BatchNorm2d(bn_config)
        self.bn2 = nn.BatchNorm2d(bn_config)

    def forward(self, x, if_mixed=False):
        return self.bn2(x) if if_mixed else self.bn1(x)


class MySequential(nn.Sequential):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input, **args):
        for module in self._modules.values():
            input = module(input, **args)
        return input


class MyDownSampleBlock(nn.Module):
    def __init__(self, inplanes, planes, expansion, stride, alternative_BN=False):
        super(MyDownSampleBlock, self).__init__()
        BN = AlternativeBN if alternative_BN else nn.BatchNorm2d
        self.alternative_BN = alternative_BN
        self.conv1x1 = conv1x1(inplanes, planes * expansion, stride)
        self.bn = BN(planes * expansion)

    def forward(self, x, if_mixed=False):
        x = self.conv1x1(x)
        if self.alternative_BN:
            x = self.bn(x, if_mixed=if_mixed)
        else:
            x = self.bn(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        print("Loading pretrained network")
        state_dict = model_zoo.load_url(model_urls['resnet18'])
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
        model.load_state_dict(state_dict)
    return model


def resnet32(**kwargs):
    model = ResNet(BasicBlock, [5, 4, 3, 2], **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        print("Loading pretrained network")
        state_dict = model_zoo.load_url(model_urls['resnet101'])
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
        model.load_state_dict(state_dict)
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
