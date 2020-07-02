import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class PixelShuffleBlock(nn.Module):
    def forward(self, x):
        return F.pixel_shuffle(x, 2)
      

def CNNBlock(in_channels, out_channels,
                 kernel_size=3, layers=1, stride=1,
                 follow_with_bn=True, activation_fn=lambda: nn.ReLU(True), affine=True):

        assert layers > 0 and kernel_size%2 and stride>0
        current_channels = in_channels
        _modules = []
        for layer in range(layers):
            _modules.append(nn.Conv2d(current_channels, out_channels, kernel_size, stride=stride if layer==0 else 1, padding=int(kernel_size/2), bias=not follow_with_bn))
            current_channels = out_channels
            if follow_with_bn:
                _modules.append(nn.BatchNorm2d(current_channels, affine=affine))
            if activation_fn is not None:
                _modules.append(activation_fn())
        return nn.Sequential(*_modules)

def SubpixelUpsampler(in_channels, out_channels, kernel_size=3, activation_fn=lambda: torch.nn.ReLU(inplace=False), follow_with_bn=True):
    _modules = [
        CNNBlock(in_channels, out_channels * 4, kernel_size=kernel_size, follow_with_bn=follow_with_bn),
        PixelShuffleBlock(),
        activation_fn(),
    ]
    return nn.Sequential(*_modules)

class UpSampleBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels,passthrough_channels, stride=1):
        super(UpSampleBlock, self).__init__()
        self.upsampler = SubpixelUpsampler(in_channels=in_channels,out_channels=out_channels)
        self.follow_up = Block(out_channels+passthrough_channels,out_channels)

    def forward(self, x, passthrough):
        out = self.upsampler(x)
        out = torch.cat((out,passthrough), 1)
        return self.follow_up(out)



class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class SaliencyModel(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(SaliencyModel, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block=block, planes=64, num_blocks=num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block=block, planes=128, num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block=block, planes=256, num_blocks=num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block=block, planes=512, num_blocks=num_blocks[3], stride=2)
        
        self.uplayer4 = UpSampleBlock(in_channels=512,out_channels=256,passthrough_channels=256)
        self.uplayer3 = UpSampleBlock(in_channels=256,out_channels=128,passthrough_channels=128)
        self.uplayer2 = UpSampleBlock(in_channels=128,out_channels=64,passthrough_channels=64)
        
        self.embedding = nn.Embedding(num_classes,512)
        # self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear = nn.Linear(512*7*7, num_classes) #to alexnet
        # self.linear = nn.Linear(512, num_classes)
        self.saliency_chans = nn.Conv2d(64,2,kernel_size=1,bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    

    
    def forward(self, x, labels):
        batch_size, timesteps, C, H, W = x.size()
        xx = x.view(batch_size * timesteps, C, H, W)
        # print('=> forward input:', x.size())
        out = self.conv1(xx)
        # print('=> out conv1:', out.size())
        out = self.bn1(out)
        # print('=> out bn1:', out.size())
        out = F.relu(out)
        # # out = F.relu(self.bn1(self.conv1(x)))
        # print('=> out relu:', out.size())
        
        scale1 = self.layer1(out)
        imgNums1, c1, h1, w1 = scale1.size()  #torch.Size([16, 64, 224, 224])
        scale1_cat = torch.flatten(scale1, 1)
        scale1_cat = scale1_cat.view(batch_size, timesteps, c1 * h1 * w1)
        scale1_cat = scale1_cat.max(dim=1).values
        scale1_cat = scale1_cat.view(batch_size, c1, h1, w1) #torch.Size([8, 64, 224, 224])
        # print('scale1Final=', scale1_cat.size())

        scale2 = self.layer2(scale1)
        imgNums2, c2, h2, w2 = scale2.size()  # torch.Size([16, 128, 112, 112])
        scale2_cat = torch.flatten(scale2, 1)
        scale2_cat = scale2_cat.view(batch_size, timesteps, c2 * h2 * w2)
        scale2_cat = scale2_cat.max(dim=1).values
        scale2_cat = scale2_cat.view(batch_size, c2, h2, w2) #torch.Size([8, 128, 112, 112])
        # print('scale2Final=', scale2_cat.size())
        
        scale3 = self.layer3(scale2)
        imgNums3, c3, h3, w3 = scale3.size()  # torch.Size([16, 256, 56, 56])
        scale3_cat = torch.flatten(scale3, 1)
        scale3_cat = scale3_cat.view(batch_size, timesteps, c3 * h3 * w3)
        scale3_cat = scale3_cat.max(dim=1).values
        scale3_cat = scale3_cat.view(batch_size, c3, h3, w3) # torch.Size([8, 256, 56, 56])
        # print('scale3Final=', scale3_cat.size())
        
        scale4 = self.layer4(scale3)
        imgNums4, c4, h4, w4 = scale4.size() # torch.Size([16, 512, 28, 28])
        scale4_cat = torch.flatten(scale4, 1)
        scale4_cat = scale4_cat.view(batch_size, timesteps, c4 * h4 * w4)
        scale4_cat = scale4_cat.max(dim=1).values
        scale4_cat = scale4_cat.view(batch_size, c4, h4, w4) #torch.Size([8, 512, 28, 28])

        # scale4 = scale4_cat
        # print('scale4Final=', scale4_cat.size())
        
        # feature filter
        em = torch.squeeze(self.embedding(labels.view(-1, 1)), 1)
        act = torch.sum(scale4*em.view(-1, 512, 1, 1), 1, keepdim=True)
        th = torch.sigmoid(act)
        scale4 = scale4 * th
        # print('=> out scale4:', scale4.size())

        # scale3 = scale3_cat
        # scale2 = scale2_cat
        # scale1 = scale1_cat
        
        upsample3 = self.uplayer4(scale4,scale3)
        upsample2 = self.uplayer3(upsample3,scale2)
        upsample1 = self.uplayer2(upsample2,scale1)
        # print('upsample1 input:', upsample1.size())

        saliency_chans = self.saliency_chans(upsample1)
        
        
        out = F.avg_pool2d(scale4, 4)
        # print('=> out avg_pool2d:', out.size())
        out = out.view(out.size(0), -1)
        # print('=> out view:', out.size())
        out = self.linear(out)
        # print('out linear input:', out.size())
       
        a = torch.abs(saliency_chans[:,0,:,:])
        b = torch.abs(saliency_chans[:, 1,:,:])
        mask = torch.unsqueeze(a / (a + b), dim=1)
        
        print(mask.size())
        
        return mask, out


def build_saliency_model(num_classes):
    return SaliencyModel(Block, [2,2,2,2],num_classes=num_classes)
