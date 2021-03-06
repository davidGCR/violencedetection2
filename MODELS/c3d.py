from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm
from torch import nn as nn
from torchvision.ops.roi_pool import RoIPool
import torch

class C3D(nn.Module):
    """C3D backbone.

    Args:
        pretrained (str | None): Name of pretrained model.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: 'pytorch'.
        conv_cfg (dict | None): Config dict for convolution layer.
            If set to None, it uses ``dict(type='Conv3d')`` to construct
            layers. Default: None.
        norm_cfg (dict | None): Config for norm layers. required keys are
            ``type``, Default: None.
        act_cfg (dict | None): Config dict for activation layer. If set to
            None, it uses ``dict(type='ReLU')`` to construct layers.
            Default: None.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation of fc layers. Default: 0.01.
    """

    def __init__(self,
                 pretrained=None,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 dropout_ratio=0.5,
                 init_std=0.005):
        super().__init__()
        if conv_cfg is None:
            conv_cfg = dict(type='Conv3d')
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        self.pretrained = pretrained
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        c3d_conv_param = dict(
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv1a = ConvModule(3, 64, **c3d_conv_param)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = ConvModule(64, 128, **c3d_conv_param)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = ConvModule(128, 256, **c3d_conv_param)
        self.conv3b = ConvModule(256, 256, **c3d_conv_param)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = ConvModule(256, 512, **c3d_conv_param)
        self.conv4b = ConvModule(512, 512, **c3d_conv_param)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = ConvModule(512, 512, **c3d_conv_param)
        self.conv5b = ConvModule(512, 512, **c3d_conv_param)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.roi_pool = RoIPool(3, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_ratio)

        self.fc8 = nn.Linear(4096, 2)
        self.init_weights()

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=self.init_std)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)
        if self.pretrained is not None:
            load_checkpoint(self, self.pretrained, strict=False)

        # if isinstance(self.pretrained, str):
        #     # logger = get_root_logger()
        #     # logger.info(f'load model from: {self.pretrained}')
        #     load_checkpoint(self, self.pretrained, strict=False)

        # elif self.pretrained is None:
        #     for m in self.modules():
        #         if isinstance(m, nn.Conv3d):
        #             kaiming_init(m)
        #         elif isinstance(m, nn.Linear):
        #             normal_init(m, std=self.init_std)
        #         elif isinstance(m, _BatchNorm):
        #             constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, X):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
                the size of x is (num_batches, 3, 16, 112, 112).

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """

        (x, vid_name, dynamicImages, bboxes) = X

        # print('forward C3D function, X size=', x.size())

        # batch_size, timesteps, C, H, W = x.size()
        x = x.permute(0,2,1,3,4)

        # print('forward C3D function, X permuted size=', x.size())

        x = self.conv1a(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)


        x = self.conv5a(x) #torch.Size([8, 512, 2, 7, 7])
        x = self.conv5b(x)

        # print('conv5b out=', x.size())
        x = self.pool5(x)

        # print('pool5 out=', x.size())

        x = x.flatten(start_dim=1)

        # print('flatten out=', x.size())

        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))

        x = self.fc8(x)

        return x

class C3D_roi_pool(nn.Module):
    """C3D backbone.

    Args:
        pretrained (str | None): Name of pretrained model.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: 'pytorch'.
        conv_cfg (dict | None): Config dict for convolution layer.
            If set to None, it uses ``dict(type='Conv3d')`` to construct
            layers. Default: None.
        norm_cfg (dict | None): Config for norm layers. required keys are
            ``type``, Default: None.
        act_cfg (dict | None): Config dict for activation layer. If set to
            None, it uses ``dict(type='ReLU')`` to construct layers.
            Default: None.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation of fc layers. Default: 0.01.
    """

    def __init__(self,
                 pretrained=None,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 dropout_ratio=0.5,
                 init_std=0.005):
        super().__init__()
        if conv_cfg is None:
            conv_cfg = dict(type='Conv3d')
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        self.pretrained = pretrained
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        c3d_conv_param = dict(
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv1a = ConvModule(3, 64, **c3d_conv_param)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = ConvModule(64, 128, **c3d_conv_param)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = ConvModule(128, 256, **c3d_conv_param)
        self.conv3b = ConvModule(256, 256, **c3d_conv_param)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = ConvModule(256, 512, **c3d_conv_param)
        self.conv4b = ConvModule(512, 512, **c3d_conv_param)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = ConvModule(512, 512, **c3d_conv_param)
        self.conv5b = ConvModule(512, 512, **c3d_conv_param)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        # self.fc6 = nn.Linear(2048, 1024)
        # self.fc7 = nn.Linear(1024, 1024)

        self.roi_pool = RoIPool(7, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_ratio)

        self.fc8 = nn.Linear(4096, 2)
        # self.fc8 = nn.Linear(1024, 2)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            # logger = get_root_logger()
            # logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=self.init_std)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, X):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
                the size of x is (num_batches, 3, 16, 112, 112).

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """

        (x, vid_name, dynamicImages, bboxes) = X

        # print('forward C3D function, X size=', x.size())

        # batch_size, timesteps, C, H, W = x.size()
        x = x.permute(0,2,1,3,4)

        # print('forward C3D function, X permuted size=', x.size())

        x = self.conv1a(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)

        # print('conv5a in=', x.size())
        x = self.conv5a(x) #torch.Size([8, 512, 2, 7, 7])

        batch_size, C, D, H, W = x.size()
        x = x.view(batch_size * D, C, H, W)

        # print('x before rp=', x.size())

        bbx = [torch.cat((bboxes, bboxes), 0)]
        # print('[bboxes]=', bbx)

        x = self.roi_pool(x, bbx)
        # print('x after rp=', x.size())
        # x = x.view(batch_size, C, D, 3, 3)
        x = x.view(batch_size, C, D, H, W)

        # print('x after rp view=', x.size())
        # print('conv5a out=', x.size())
        x = self.conv5b(x)
        # print('conv5b out=', x.size())
        x = self.pool5(x)

        # print('pool5 out=', x.size())

        x = x.flatten(start_dim=1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))

        x = self.fc8(x)

        return x

class C3D_bn(nn.Module):
    """C3D backbone.

    Args:
        pretrained (str | None): Name of pretrained model.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: 'pytorch'.
        conv_cfg (dict | None): Config dict for convolution layer.
            If set to None, it uses ``dict(type='Conv3d')`` to construct
            layers. Default: None.
        norm_cfg (dict | None): Config for norm layers. required keys are
            ``type``, Default: None.
        act_cfg (dict | None): Config dict for activation layer. If set to
            None, it uses ``dict(type='ReLU')`` to construct layers.
            Default: None.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation of fc layers. Default: 0.01.
    """

    def __init__(self,
                 pretrained=None,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 dropout_ratio=0.5,
                 init_std=0.005):
        super().__init__()
        if conv_cfg is None:
            conv_cfg = dict(type='Conv3d')
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        self.pretrained = pretrained
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        c3d_conv_param = dict(
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv1a = ConvModule(3, 64, **c3d_conv_param)
        self.conv1_bn = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = ConvModule(64, 128, **c3d_conv_param)
        self.conv2_bn = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = ConvModule(128, 256, **c3d_conv_param)
        self.conv3a_bn = nn.BatchNorm3d(256)
        self.conv3b = ConvModule(256, 256, **c3d_conv_param)
        self.conv3b_bn = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = ConvModule(256, 512, **c3d_conv_param)
        self.conv4a_bn = nn.BatchNorm3d(512)
        self.conv4b = ConvModule(512, 512, **c3d_conv_param)
        self.conv4b_bn = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = ConvModule(512, 512, **c3d_conv_param)
        self.conv5a_bn = nn.BatchNorm3d(512)
        self.conv5b = ConvModule(512, 512, **c3d_conv_param)
        self.conv5b_bn = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_ratio)

        self.fc8 = nn.Linear(4096, 2)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            # logger = get_root_logger()
            # logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=self.init_std)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, X):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
                the size of x is (num_batches, 3, 16, 112, 112).

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """

        (x, vid_name, dynamicImages, bboxes) = X

        # print('forward C3D function, X size=', x.size())

        # batch_size, timesteps, C, H, W = x.size()
        x = x.permute(0,2,1,3,4)

        # print('forward C3D function, X permuted size=', x.size())

        x = self.conv1a(x)
        x = self.pool1(self.conv1_bn(x))

        x = self.conv2a(x)
        x = self.pool2(self.conv2_bn(x))

        x = self.conv3a(x)
        x = self.conv3b(self.conv3a_bn(x))
        x = self.pool3(self.conv3b_bn(x))

        x = self.conv4a(x)
        x = self.conv4b(self.conv4a_bn(x))
        x = self.pool4(self.conv4b_bn(x))

        x = self.conv5a(x)
        x = self.conv5b(self.conv5a_bn(x))
        x = self.pool5(self.conv5b_bn(x))

        x = x.flatten(start_dim=1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))

        x = self.fc8(x)

        return x

    # def train(self, mode=True):
    #     super(C3D, self).train(mode)

if __name__ == '__main__':
    model = C3D(pretrained='https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_pretrain_20201016-dcc47ddc.pth')
    model.init_weights()
    print(model)
