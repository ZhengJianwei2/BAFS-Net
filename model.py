import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import resnet

def weight_init(module):
    for n, m in module.named_children():
        if 'backbone' in n:
            continue
        print('Initialize:' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if "nonlocal_block" in n:
                continue
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Upsample, nn.AdaptiveAvgPool2d, nn.Sigmoid, nn.MaxPool2d, nn.Softmax)):
            pass
        else:
            m.initialize()

class WeightedBlock(nn.Module):
    """Weighted Block
    """
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        input_conv = self.input_conv(x)
        return input_conv * self.weight(input_conv)

    def initialize(self):
        weight_init(self)

class MappingModule(nn.Module):
    def __init__(self, out_c):
        super(MappingModule, self).__init__()

        nums = [256, 512, 1024, 2048]
        self.cv1_3 = nn.Sequential(
            nn.Conv2d(nums[0], out_c, kernel_size=3, stride=1,
                      padding=1, dilation=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.cv1_1 = nn.Sequential(
            nn.Conv2d(nums[0], out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
        self.cv2_3 = nn.Sequential(
            nn.Conv2d(nums[1], out_c, kernel_size=3, stride=1,
                      padding=3, dilation=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.cv2_1 = nn.Sequential(
            nn.Conv2d(nums[1], out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
        self.cv3_3 = nn.Sequential(
            nn.Conv2d(nums[2], out_c, kernel_size=3, stride=1,
                      padding=5, dilation=5),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.cv3_1 = nn.Sequential(
            nn.Conv2d(nums[2], out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
        self.cv4_3 = nn.Sequential(
            nn.Conv2d(nums[3], out_c, kernel_size=3, stride=1,
                      padding=5, dilation=5),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.cv4_1 = nn.Sequential(
            nn.Conv2d(nums[3], out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, out2, out3, out4, out5):
        o2_1 = self.cv1_3(out2)
        o2_2 = self.cv1_1(out2)
        o2 = o2_1 + o2_2
        
        o3_1 = self.cv2_3(out3)
        o3_2 = self.cv2_1(out3)
        o3 = o3_1 + o3_2
        
        o4_1 = self.cv3_3(out4)
        o4_2 = self.cv3_1(out4)
        o4 = o4_1 + o4_2
        
        o5_1 = self.cv4_3(out5)
        o5_2 = self.cv4_1(out5)
        o5 = o5_1 + o5_2
        
        return o2, o3, o4, o5

    def initialize(self):
        weight_init(self)

class EdgeGuidance(nn.Module):
    def __init__(self):
        super(EdgeGuidance, self).__init__()
        self.conv0 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

    def forward(self, input1, input2=[0,0,0,0]):
        out0 = F.relu(self.bn0(self.conv0(input1[0]+input2[0])), inplace=True)
        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out1 = F.relu(self.bn1(self.conv1(input1[1]+input2[1]+out0)), inplace=True)
        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        out2 = F.relu(self.bn2(self.conv2(input1[2]+input2[2]+out1)), inplace=True)
        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(input1[3]+input2[3]+out2)), inplace=True)
        return out3
    
    def initialize(self):
        weight_init(self)

class DecoderBlock(nn.Module):
    def __init__(self, in_c):
        super(DecoderBlock, self).__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(in_c*2, in_c, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )
        self.edge_out = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, res, de, att, edge=None):
        shape = res.shape[2:]
        att1 = F.interpolate(
            att, size=shape, mode='bilinear', align_corners=True)
        b1 = att1 * res
        b1 = res + b1
        b1_out = self.cv1(b1)

        att2 = F.interpolate(
            att, size=de.shape[2:], mode='bilinear', align_corners=True)
        b2 = att2 * de
        b2 = de + b2
        b2 = self.cv2(b2)

        b2_out = F.interpolate(b2, size=shape, mode='bilinear', align_corners=True)

        out = torch.cat((b1_out , b2_out), dim=1)
        out = self.cv3(out)
        e = F.interpolate(edge, size=out.shape[2:],mode='bilinear',align_corners=True)
        out = out + e
        out = self.edge_out(out)

        return out

    def initialize(self):
        weight_init(self)


class BiAttentionalBlock(nn.Module):
    def __init__(self, plane, norm_layer=nn.BatchNorm2d):
        super(BiAttentionalBlock, self).__init__()
        self.conv1 = nn.Linear(plane, plane)
        self.conv2 = nn.Linear(plane, plane)
        self.conv = nn.Sequential(nn.Conv2d(plane, plane, 3, stride=1, padding=1, bias=False),
                                  norm_layer(plane),
                                  nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Temporarily hidden due to confidentiality
        
        pass

    def initialize(self):
        weight_init(self)



class CascadeDecoder(nn.Module):
    def __init__(self):
        super(CascadeDecoder, self).__init__()
        self.in_c = 128

        self.dec1 = DecoderBlock(self.in_c)
        self.dec2 = DecoderBlock(self.in_c)
        self.dec3 = DecoderBlock(self.in_c)

        self.s0 = nn.Conv2d(self.in_c, 1, kernel_size=3, stride=1, padding=1)
        self.s1 = nn.Conv2d(self.in_c, 1, kernel_size=3, stride=1, padding=1)
        self.s2 = nn.Conv2d(self.in_c, 1, kernel_size=3, stride=1, padding=1)
        self.s3 = nn.Conv2d(self.in_c, 1, kernel_size=3, padding=1)
        
        self.w3 = WeightedBlock(128, 128)

        self.nl1 = BiAttentionalBlock(128)
        self.nl2 = BiAttentionalBlock(128)
        self.nl3 = BiAttentionalBlock(128)

    def forward(self, x, edge=None):
        r2, r3, r4, r5 = x
        r2 = self.nl1(r2)
        
        r2_ = F.interpolate(r2, size=r3.shape[2:], mode='bilinear')
        r3 = self.nl2(r3 + r2_)

        r3_ = F.interpolate(r3, size=r4.shape[2:], mode='bilinear')
        r4 = self.nl3(r4 + r3_)

        dec1 = self.dec1(r4, r5, r5, edge)

        dec2 = self.dec2(r3, dec1, r5, edge)

        dec3 = self.dec3(r2, dec2, r5, edge)

        s1 = self.s1(dec1)
        s2 = self.s2(dec2)
        s3 = self.s3(self.w3(dec3))

        return s1, s2, s3

    def initialize(self):
        weight_init(self)




class Net(nn.Module):

    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.backbone = resnet(cfg)

        self.mapc1 = MappingModule(128)
        self.edge_guidance = EdgeGuidance()
        self.cascade_decoder = CascadeDecoder()

        self.e_conv = nn.Conv2d(128,1,3,padding=1)
        self.eb = nn.Conv2d(128,128,3,padding=1)

        self.initialize()

    def forward(self, x):
        #    64 256 512 1024 2048
        #    /2  /4  /8  /16  /32
        shape = x.shape[2:]
        res1, res2, res3, res4, res5 = self.backbone(x)

        res2a, res3a, res4a, res5a = self.mapc1(res2, res3, res4, res5)
        
        res2b, res3b, res4b, res5b = self.mapc1(res2, res3, res4, res5)

        edge_branch = self.edge_guidance([res5a, res4a, res3a, res2a])

        out1,out2,out3 = self.cascade_decoder([res2b, res3b, res4b, res5b], self.eb(edge_branch))

        edge = self.e_conv(edge_branch)
        o1 = F.interpolate(out1, size=shape, mode='bilinear', align_corners=True)
        o2 = F.interpolate(out2, size=shape, mode='bilinear', align_corners=True)
        o3 = F.interpolate(out3, size=shape, mode='bilinear', align_corners=True)
        e = F.interpolate(edge, size=shape, mode='bilinear', align_corners=True)
        return  o1, o2, o3, e

    def initialize(self):
        if self.cfg.snapshot != 'none':
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
