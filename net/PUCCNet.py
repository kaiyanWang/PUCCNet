import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################################################
# DownSample and UpSample
class DownSample(nn.Module):
    def __init__(self, in_channels):
        super(DownSample, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, 4, 2,1),
            nn.PReLU()
        )

    def forward(self, x):
        out = self.conv1(x)
        return out

class UpSample(nn.Module):
    def __init__(self, in_channels):
        super(UpSample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=4, stride=2, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        out = self.deconv(x)
        return out

##########################################################################
# channel shuffle
class ChannelShuffle(nn.Module):
    def __init__(self,groups):
        super(ChannelShuffle, self).__init__()
        self.group=groups
    def forward(self,x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.group
        # num_channels = groups * channels_per_group

        # grouping, 通道分组
        # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
        x = x.view(batchsize, self.group, channels_per_group, height, width)

        # channel shuffle, 通道洗牌
        x = torch.transpose(x, 1, 2).contiguous()
        # x.shape=(batchsize, channels_per_group, groups, height, width)
        # flatten
        x = x.view(batchsize, -1, height, width)

        return x

##########################################################################
# Pixel Attention
class PA(nn.Module):
    def __init__(self, channel):
        super(PA, self).__init__()
        self.inter=channel // 8
        if self.inter==0:
            self.inter = channel // 4
        self.pa = nn.Sequential(
                nn.Conv2d(channel, self.inter, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return y

##########################################################################
# LCA
class LCA(nn.Module):
    def __init__(self,in_channel,num=1):
        super(LCA, self).__init__()
        self.num=num
        self.in_channel=in_channel

        self.avg=nn.AdaptiveAvgPool2d(num)
        self.pixunshuffle=nn.PixelUnshuffle(self.num)
        self.channelshuffle1=ChannelShuffle(self.num*self.num)
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=in_channel*self.num*self.num,out_channels=self.num*self.num,groups=self.num*self.num,kernel_size=1,stride=1,padding=0),
            nn.PReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=self.num*self.num, out_channels=in_channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self,x):
        avg=self.avg(x)
        ps=self.pixunshuffle(avg)
        cs1=self.channelshuffle1(ps)
        conv1=self.conv1(cs1)
        return self.conv2(conv1)


##########################################################################
# SSB
class SSB(nn.Module):
    def __init__(self,in_channels):
        super(SSB, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.down = nn.Conv2d(in_channels, in_channels * 2, 3, 2, 1)
        self.conv=nn.Conv2d(in_channels,in_channels,3,1,1)
        self.uppa = PA(in_channels//2)
        self.downpa = PA(in_channels*2)
        self.pa = PA(in_channels)


    def forward(self,x):
        up = self.up(x)
        down = self.down(x)
        conv = self.conv(x)

        uppa = self.uppa(up)
        downpa = self.downpa(down)
        pa = self.pa(conv)

        uppa = F.interpolate(uppa,scale_factor=0.5,mode="bilinear",recompute_scale_factor=True,align_corners=True)
        downpa = F.interpolate(downpa,scale_factor=2,mode="bilinear",recompute_scale_factor=True,align_corners=True)
        return (pa+uppa+downpa)/3

##########################################################################
# AFF: Fusion
class AFF(nn.Module):
    def __init__(self,in_channel,out_channel,num=2):
        super(AFF, self).__init__()
        self.conv1_1=nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3,1,1),
            nn.PReLU()
        )
        self.conv1_2=nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3,1,1),
            nn.PReLU()
        )
        self.lca1 = LCA(in_channel)
        self.lca2 = LCA(in_channel)
        self.lca3 = LCA(in_channel)


    def forward(self,x,y):
        x = self.conv1_1(x)
        y = self.conv1_2(y)
        lca1 = self.lca1(x)
        lca2 = self.lca2(y)
        lca3 = self.lca3(x + y)
        lca = lca1 * lca2
        result = (lca + lca3) / 2 * x + (1-(lca + lca3) / 2) * y
        return result

##########################################################################
class Dilated(nn.Module):
    def  __init__(self,in_c,out_c,size=3,stride=1,d=1):
        super(Dilated, self).__init__()
        padding=((size-1)//2)*d
        self.conv=nn.Conv2d(in_c,out_c,kernel_size=size,stride=stride,padding=padding,bias=False,dilation=d)
    def forward(self,input):
        return self.conv(input)

##########################################################################
class DetailLayer(nn.Module):
    def __init__(self,in_c,out_c):
        super(DetailLayer, self).__init__()
        n=out_c//5
        n_1=out_c-4*n
        self.conv_1x1=nn.Conv2d(in_c,n,1,1)
        self.d1 = Dilated(n,n_1,3,1,1)
        self.d2 = Dilated(n, n, 3, 1, 2)
        self.d4 = Dilated(n, n, 3, 1, 4)
        self.d8 = Dilated(n, n, 3, 1, 8)
        self.d16 = Dilated(n, n, 3, 1, 16)
        self.bn=nn.Sequential(
            nn.BatchNorm2d(out_c)
        )
    def forward(self,x):
        output1=self.conv_1x1(x)
        d1=self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1+d4
        add3 = add2+d8
        add4 = add3+d16

        combine=torch.cat([d1,add1,add2,add3,add4],dim=1)
        return self.bn(combine+x)



##########################################################################
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1,padding=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

##########################################################################
class DLGRB(nn.Module):
    def __init__(self, inchannel, num=1):
        super(DLGRB, self).__init__()

        self.lca = LCA(inchannel, num)
        self.rdb = nn.Sequential(
            ResidualBlock(inchannel),
            ResidualBlock(inchannel)
        )
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm2d(inchannel)

    def forward(self,x):
        ca = self.lca(x) * x
        rdb = self.rdb(ca)
        out = self.bn(rdb)
        return self.prelu(out) + x

##########################################################################
class DRB(nn.Module):
    def __init__(self, in_c,num=2):
        super(DRB, self).__init__()
        self.lca = LCA(in_c,num)
        self.ssb = SSB(in_c)
        self.detail=DetailLayer(in_c,in_c)
        self.conv=nn.Sequential(
            nn.Conv2d(in_c,in_c,3,1,1),
            nn.BatchNorm2d(in_c),
            nn.PReLU()
        )
    def forward(self, x):
        lca = self.lca(x)
        x_1 = lca * x
        detail = self.detail(x_1)
        ssb = self.ssb(detail+x)
        detail_1 = ssb*(detail+x)
        conv = self.conv(detail_1)
        return conv+x


##########################################################################
class FNet(nn.Module):
    def __init__(self,in_c=8):
        super(FNet, self).__init__()

        self.drb1 = DRB(in_c,4)
        self.down1=DownSample(in_c)

        self.drb2 = DRB(in_c*2, 4)
        self.down2 = DownSample(in_c*2)

        self.drb3 = DRB(in_c*4, 2)
        self.down3 = DownSample(in_c*4)

        self.drb4 = DRB(in_c * 8, 2)
        self.down4 = DownSample(in_c * 8)

        self.coder=nn.Sequential(
            DLGRB(in_c * 16),
            DLGRB(in_c * 16),
            DLGRB(in_c * 16),
            DLGRB(in_c * 16),
            DLGRB(in_c * 16),
            DLGRB(in_c * 16),
            DLGRB(in_c * 16),
            DLGRB(in_c * 16)
        )

        self.up4=UpSample(in_c * 16)
        self.aff4 = AFF(in_c * 8,in_c * 8)
        self.drb4x = DRB(in_c * 8,2)

        self.up3 = UpSample(in_c * 8)
        self.aff3 = AFF(in_c * 4, in_c * 4)
        self.drb3x = DRB(in_c * 4,2)

        self.up2 = UpSample(in_c * 4)
        self.aff2 = AFF(in_c * 2, in_c * 2)
        self.drb2x = DRB(in_c * 2,4)

        self.up1 = UpSample(in_c * 2)
        self.drb1x = DRB(in_c,4)

    def forward(self,x):
        drb1 = self.drb1(x)
        down1=self.down1(drb1)

        drb2=self.drb2(down1)
        down2=self.down2(drb2)

        drb3 = self.drb3(down2)
        down3 = self.down3(drb3)

        drb4 = self.drb4(down3)
        down4 = self.down4(drb4)

        coder = self.coder(down4)

        up4 = self.up4(coder)
        aff4 = self.aff4(up4,down3)
        drb4x = self.drb4x(aff4)

        up3 = self.up3(drb4x)
        aff3=self.aff3(up3,down2)
        drb3x = self.drb3x(aff3)

        up2 = self.up2(drb3x)
        aff2 = self.aff2(up2,down1)
        drb2x = self.drb2x(aff2)

        up1 = self.up1(drb2x)
        drb1x = self.drb1x(up1)

        return drb1x


##########################################################################
class PUCCNet(nn.Module):
    def __init__(self):
        super(PUCCNet, self).__init__()
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(3,8,1),
            nn.PReLU()
        )
        self.cfrt = FNet()
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(8,3,1),
            nn.PReLU()
        )

    def forward(self,x):
        conv1x1_1 = self.conv1x1_1(x)
        result = self.cfrt(conv1x1_1)
        conv1x1_2 = self.conv1x1_2(result)
        return conv1x1_2


if __name__ == '__main__':
    x = torch.randn((1, 3, 512, 512)).cuda()
    net = PUCCNet().cuda()

    from thop import profile, clever_format

    flops, params = profile(net, (x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)