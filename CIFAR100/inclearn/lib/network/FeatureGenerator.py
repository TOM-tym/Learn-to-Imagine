import torch
import torch.nn as nn
import torch.nn.functional as F
from inclearn.convnet.my_resnet import ResidualBlock, Stage


class FeatureGenerator(nn.Module):
    def __init__(self, input_channels, fusion_num=2, latent_dim=64, num_blocks=1):
        super().__init__()
        self.fusion_num = fusion_num
        self.input_channels = input_channels
        self._downsampling_type = 'stride'
        self.last_relu = False
        self.stage2 = self._make_layer(ResidualBlock, self.fusion_num * input_channels, increase_dim=False,
                                       n=num_blocks)
        self.conv3 = nn.Conv2d(self.fusion_num * input_channels, input_channels, kernel_size=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, labeled_features, unlabeled_features):
        mixed_feat = []
        for idx_l in range(len(labeled_features)):
            for idx_u in range(len(unlabeled_features)):
                l_mix_u = torch.cat((labeled_features[idx_l], unlabeled_features[idx_u]), dim=0)
                mixed_feat.append(l_mix_u)
        mixed_feat = torch.stack(mixed_feat)
        _, out = self.stage2(mixed_feat)
        out = self.conv3(out)
        return out

    def freeze(self, trainable=False, model="all"):
        if model == "all":
            model = self
        else:
            assert False, model

        if not isinstance(model, nn.Module):
            return self

        for param in model.parameters():
            param.requires_grad = trainable

    def _make_layer(self, Block, planes, increase_dim=False, n=None):
        layers = []

        if increase_dim:
            layers.append(
                Block(
                    planes,
                    increase_dim=True,
                    last_relu=False,
                    downsampling=self._downsampling_type
                )
            )
            planes = 2 * planes

        for i in range(n):
            layers.append(Block(planes, last_relu=True, downsampling=self._downsampling_type))

        return Stage(layers, block_relu=self.last_relu)

