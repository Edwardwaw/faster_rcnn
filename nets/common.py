from torch import nn


## base block --------------------------------------------------------------------------------------

class CR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CBR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CGR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CGR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.gn = nn.GroupNorm(32, out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        return x






########################### self.neck=FPN -------------------------------------------------------------------


class FPN(nn.Module):
    def __init__(self,in_channels,out_channel,bias=True):
        super(FPN, self).__init__()
        assert len(in_channels)==4, "only support fpn input with c2,c3,c4,c5"
        self.c2_inner=nn.Conv2d(in_channels[0],out_channel,1,bias=bias)
        self.c2_layer=nn.Conv2d(out_channel,out_channel,3,padding=1,bias=bias)

        self.c3_inner=nn.Conv2d(in_channels[1],out_channel,1,bias=bias)
        self.c3_layer=nn.Conv2d(out_channel,out_channel,3,padding=1,bias=bias)

        self.c4_inner=nn.Conv2d(in_channels[2],out_channel,1,bias=bias)
        self.c4_layer=nn.Conv2d(out_channel,out_channel,3,padding=1,bias=bias)

        self.c5_inner=nn.Conv2d(in_channels[3],out_channel,1,bias=bias)
        self.c5_layer=nn.Conv2d(out_channel,out_channel,3,padding=1,bias=bias)

        self.max_pooling=nn.MaxPool2d(2,2)

    def forward(self,xs):
        assert len(xs)==4, "be sure len(xs)==4"
        c2,c3,c4,c5=xs
        f2,f3,f4,f5=self.c2_inner(c2),self.c3_inner(c3),self.c4_inner(c4),self.c5_inner(c5)
        p5=self.c5_layer(f5)
        p4=self.c4_layer(nn.UpsamplingNearest2d(size=(f4.shape[-2:]))(f5)+f4)
        p3=self.c3_layer(nn.UpsamplingNearest2d(size=(f3.shape[-2:]))(f4)+f3)
        p2=self.c2_layer(nn.UpsamplingNearest2d(size=(f2.shape[-2:]))(f3)+f2)
        p6=self.max_pooling(p5)
        return [p2,p3,p4,p5,p6]












