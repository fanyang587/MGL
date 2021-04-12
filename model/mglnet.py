import torch
from torch import nn
import torch.nn.functional as F
import model.resnet as models
import numpy as np
from model.basicnet import MutualNet, ConcatNet


class MGLNet(nn.Module):
    def __init__(self, layers=50, dropout=0.1, classes=1, zoom_factor=8, criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True, args=None):
        super(MGLNet, self).__init__()
        assert layers in [50, 101, 152]
        assert classes == 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.args = args
        models.BatchNorm = BatchNorm
        self.gamma = 1.0

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.dim = 512

        self.pred = nn.Sequential(
            nn.Conv2d(2048, self.dim, kernel_size=3, padding=1, bias=False),
            BatchNorm(self.dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(self.dim, classes, kernel_size=1)
        )

        self.region_conv = self.pred[0:4] # 2048 -> 512
        self.edge_cat = ConcatNet(BatchNorm) # concat low-level feature map to predict edge

        # cascade mutual net
        self.mutualnet0 = MutualNet(BatchNorm, dim=self.dim, num_clusters=args.num_clusters, dropout=dropout)
        if args.stage == 1:
            self.mutualnets = nn.ModuleList([self.mutualnet0])
        elif args.stage == 2:
            self.mutualnet1 = MutualNet(BatchNorm, dim=self.dim, num_clusters=args.num_clusters, dropout=dropout)
            self.mutualnets = nn.ModuleList([self.mutualnet0, self.mutualnet1])

    def forward(self, x, y=None, iter_num=0, y2=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        ##step1. backbone layer
        x_0 = self.layer0(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        ##step2. concat edge feature by side-output feature
        coee_x = self.edge_cat(x_1, x_2, x_3, x_4) # edge pixel-level feature
        cod_x = self.region_conv(x_4) # 2048 -> 512

        main_loss = 0.
        for net in self.mutualnets:
            n_coee_x, coee, n_cod_x, cod = net(coee_x, cod_x)
            coee_x = coee_x + n_coee_x
            cod_x = cod_x + n_cod_x

            if self.zoom_factor != 1:
                coee = F.interpolate(coee, size=(h, w),  mode='bilinear', align_corners=True)
                cod = F.interpolate(cod, size=(h, w), mode='bilinear', align_corners=True)
                if self.training:
                    main_loss += self.gamma * self.criterion(coee, y2) # supervise edge
                    main_loss += self.criterion(cod, y) # supervise region

        if self.training:
            return cod, coee, main_loss
        else:
            return cod, coee

