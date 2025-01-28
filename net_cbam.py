import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU
from torch.nn import Softmax
EPSILON = 1e-10

def INF(B,H,W):
     tensor = torch.tensor(float("inf")).repeat(H)
     return -diag2(tensor).unsqueeze(0).repeat(B*W,1,1)

def diag2(x):
    diag_matrix = x.unsqueeze(0) * torch.eye(len(x))
    diag_matrix = torch.nan_to_num(diag_matrix, 0)
    return diag_matrix

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_relu=False, use_norm= False):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = kernel_size//2)
        # self.dropout = nn.Dropout2d(p=0.5)
        self.use_relu = use_relu
        self.use_norm = use_norm
        self.bn = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        out = self.conv2d(x)
        if self.use_norm:
            x =self.bn(out)
        if self.use_relu:
            x = F.relu(out, inplace=False)

        return out

class resdnet_Block(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=1, kernel_size_neck=3, stride=1, width=4):
        super(resdnet_Block, self).__init__()
        self.width = width

        self.conv1 = ConvLayer(in_channels, self.width * 4, kernel_size, stride, use_relu=True, use_norm=False)
        
        convs1 = []
        for i in range(8):
            convs1.append(ConvLayer(self.width * 4, self.width, kernel_size_neck, stride, use_relu=True, use_norm= False))
        self.convs1 = nn.ModuleList(convs1)

        self.conv2 =  ConvLayer(self.width * 4, out_channels, kernel_size, stride, use_relu=True, use_norm = False)
        
        self.conv3 = ConvLayer(in_channels, out_channels, kernel_size, stride, use_relu=True,  use_norm = False)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv3( x)
        x = self.conv1(x)
        
        [x1, x2, x3, x4] = torch.split(x, self.width, 1)
        y1 = self.convs1[0](x)
        y2 = self.convs1[1](torch.cat((x2, x3, x4, y1), dim=1))
        y3 = self.convs1[2](torch.cat((x3, x4, y1, y2), dim=1))
        y4 = self.convs1[3](torch.cat((x4, y1 ,y2 ,y3), dim=1))
        x1 = self.convs1[4](torch.cat((y1 ,y2 ,y3, y4), dim=1))
        x2 = self.convs1[5](torch.cat((y2 ,y3, y4, x1), dim=1))
        x3 = self.convs1[6](torch.cat((y3, y4, x1, x2), dim=1))
        x4 = self.convs1[7](torch.cat((y4 ,x1, x2,x3), dim=1))
        
        out2 =  self.conv2(torch.cat([x1, x2, x3, x4],dim=1))

        return self.relu(out1+out2)
    
class CrissCrossSpatialAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossSpatialAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.inf = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.inf(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)

        return self.gamma*(out_H + out_W) + x
    
    
class RCCASpatialModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCCASpatialModule, self).__init__()
        inter_channels = in_channels // 4
        self.downsample1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=4)
        self.upsample1 = nn.ConvTranspose2d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=5, stride=4)
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.cca = CrissCrossSpatialAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, recurrence=1):
        downsampled = self.downsample1(x)
        output = self.conva(downsampled)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)
        output = self.upsample1(output, output_size=x.size())
        output = self.bottleneck(torch.cat([x, output], 1))
        return output
    
class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        gamma = torch.tensor(0.5)
        output = self.conv(concat)
        output = F.sigmoid(output) * x
        return gamma * output 

class CAM(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(CAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in/ float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )

        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        max_pool = torch.max(x.flatten(2), 2)[0].unsqueeze(2).unsqueeze(3)
        avg_pool = torch.mean(x.flatten(2), 2).unsqueeze(2).unsqueeze(3)

        
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)
        

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1,1,kernel[0], kernel[1])
        return  out * x
    
# DenseFuse network
class ResCCNet_cbam_fuse(nn.Module):
    def __init__(self, in_channel=1, out_channel=1):
        super(ResCCNet_cbam_fuse, self).__init__()
        resblock = resdnet_Block
               
        width = [4, 8, 16]
        
        channels = [16, 32, 64]
        decoder_channel = [16, 32, 64, 112]
        
        kernel_size_1 = 1
        kernel_size_2 = 3
        stride = 1

        # encoder
        self.encoder_conv = nn.Sequential(nn.Conv2d(in_channel, channels[0], kernel_size=3, padding=1), ReLU(inplace=True))
        self.RDB1 = resblock(channels[0], channels[0], kernel_size_1, kernel_size_2, stride, width[0])
        self.RDB2 = resblock(channels[1], channels[1], kernel_size_1, kernel_size_2, stride, width[1])
        self.RDB3 = resblock(channels[2], channels[2], kernel_size_1, kernel_size_2, stride, width[2])

        #Spatial Attention
        self.sam = SAM(bias=False)
        #self.sam = RCCASpatialModule(decoder_channel[3], decoder_channel[3])

        #Channel Attention
        self.cam = CAM(decoder_channel[3], 2)

        # decoder
        self.conv1 = ConvLayer(decoder_channel[3], decoder_channel[2], kernel_size_2, stride,use_relu=True)
        self.conv2 = ConvLayer(decoder_channel[2], decoder_channel[1], kernel_size_2, stride,use_relu=True)
        self.conv3 = ConvLayer(decoder_channel[1], decoder_channel[0], kernel_size_2, stride,use_relu=True)
        self.conv4 = ConvLayer(decoder_channel[0], out_channel, kernel_size_2, stride,use_relu=True)


    def encoder(self, input):
        x0 = self.encoder_conv(input)
        x1 = self.RDB1(x0)
        x2 = self.RDB2(torch.cat((x0, x1), dim=1))
        x3 = self.RDB3(torch.cat((x0, x1, x2), dim=1))

        return torch.cat([x1, x2, x3], dim=1)
    
    def cbam(self, input):
        output = self.cam(input)
        output = self.sam(output)
        return output + input

    
    def decoder(self, input):
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv3(input)
        out = self.conv4(input)
        return out
    
    def channel_fusion(self, tensor1, tensor2):
        # calculate channel attention
        attention_map1 = self.cam(tensor1)
        attention_map2 = self.cam(tensor2)
        # get weight map
        attention_p1_w1 = attention_map1 / (attention_map1 + attention_map2 + EPSILON)
        attention_p2_w2 = attention_map2 / (attention_map1 + attention_map2 + EPSILON)

        tensor_f = attention_p1_w1 * tensor1 + attention_p2_w2 * tensor2
        return tensor_f
    
    def spatial_fusion(self, tensor1, tensor2):
        # calculate spatial attention
        spatial1 = self.sam(tensor1)
        spatial2 = self.sam(tensor2)
        # get weight map
        spatial_w1 = spatial1 / (spatial1 + spatial2 + EPSILON)
        spatial_w2 = spatial2 / (spatial1 + spatial2 + EPSILON)

        tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2
        return tensor_f
    
    def fusion(self, tensor1, tensor2 ):
        f_channel = self.channel_fusion(tensor1, tensor2)
        f_spatial = self.spatial_fusion(tensor1, tensor2)
        tensor_f = (f_channel + f_spatial) / 2
        return tensor_f 
    
    def forward(self, x, y):
            future_x = torch.jit.fork(self.encoder, x)
            future_y = torch.jit.fork(self.encoder, y)
            encoder_x = torch.jit.wait(future_x)
            encoder_y = torch.jit.wait(future_y)

            fusion_x = self.fusion(encoder_x, encoder_y)
            return self.decoder(fusion_x)

