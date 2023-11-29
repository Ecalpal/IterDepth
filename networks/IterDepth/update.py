import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super(DepthHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        # return self.sig(self.conv2(self.relu(self.conv1(x))))
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128+256):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicEncoder(nn.Module):
    def __init__(self, in_e=128):
        super(BasicEncoder, self).__init__()
        self.convc1 = nn.Conv2d(in_e, in_e*2, 1, padding=0)
        self.convc2 = nn.Conv2d(in_e*2, int(in_e*1.5), 3, padding=1)
        # self.convf1 = nn.Conv2d(1, 128, 7, padding=3)
        # self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convf1 = torch.nn.Sequential(
            nn.ReflectionPad2d(3),
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=0, bias=True),
            # torch.nn.LeakyReLU(inplace=True)
            )
        self.convf2 = torch.nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=0, bias=True),
            # torch.nn.LeakyReLU(inplace=True)
            )

        self.conv = nn.Conv2d(32+int(in_e*1.5), in_e-1, 3, padding=1)

    def forward(self, depth, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        dep = F.relu(self.convf1(depth))
        dep = F.relu(self.convf2(dep))

        cor_dep = torch.cat([cor, dep], dim=1)
        out = F.relu(self.conv(cor_dep))
        return torch.cat([out, depth], dim=1)

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = DepthHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

class BasicUpdateBlock(nn.Module):
    def __init__(self, in_e=128, in_c=256):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicEncoder(in_e=in_e)
        self.gru = SepConvGRU(hidden_dim=in_e, input_dim=in_e+in_c)
        self.depth_head = DepthHead(in_e, hidden_dim=in_e)

        self.mask = nn.Sequential(
            nn.Conv2d(in_e, in_e*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_e*2, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, depth, upsample=True):
        depth_features = self.encoder(depth, corr)
        inp = torch.cat([inp, depth_features], dim=1)

        net = self.gru(net, inp)
        delta_depth = self.depth_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_depth



