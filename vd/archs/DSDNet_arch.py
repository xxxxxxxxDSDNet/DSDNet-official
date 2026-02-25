import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .archs_utils import *
from .VSSM_arch import VSSM as CMB

#SADM includes  DynamicReflectionAttention and CDGM
@ARCH_REGISTRY.register()
class DSDNet(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, Depth, kernel_size=3, reduction=4, bias=False, heads=[2, 2, 4, 8]):
        super(DSDNet, self).__init__()
        act = nn.PReLU()
        [head1, head2, head3, head4] = heads

        self.raw2YCbCr = RGB2YCbCr()

        self.shallow_feat1 = nn.Sequential(conv(7, n_feat, kernel_size=3, bias=False))

        self.shallow_feat2 = nn.Sequential(conv(3, n_feat, kernel_size=3, bias=False))
        
        self.DSDC_1 = CGB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.encoder_level1 = CMB(in_chans=n_feat,
                                   embed_dim=n_feat,
                                   depths=[Depth[0]])
        
        self.down12_Y = DownSample(n_feat, scale_unetfeats)
        self.down12_R = DownSample(n_feat, scale_unetfeats)
        self.DSDC_2 = CGB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act, dilation=2)
        self.encoder_level2 = CMB(in_chans=n_feat + scale_unetfeats,
                                    embed_dim=n_feat + scale_unetfeats,
                                    depths=[Depth[1]])      
        self.dfa_2 = DynamicReflectionAttention(n_feat + scale_unetfeats, head2)
        self.cdgm_2_Y = CDGM(n_feat + scale_unetfeats)
        self.cdgm_2_R = CDGM(n_feat + scale_unetfeats)
        
        self.down23_Y = DownSample(n_feat + scale_unetfeats, scale_unetfeats)
        self.down23_R = DownSample(n_feat + scale_unetfeats, scale_unetfeats)
        
        self.DSDC_3 = CGB(n_feat + scale_unetfeats * 2, kernel_size, reduction, bias=bias, act=act, dilation=4)
        self.encoder_level3 = CMB(in_chans=n_feat + scale_unetfeats * 2,
                                   embed_dim=n_feat + scale_unetfeats * 2,
                                   depths=[Depth[2]])
        self.dfa_3 = DynamicReflectionAttention(n_feat + scale_unetfeats * 2, head3)
        self.cdgm_3_Y = CDGM(n_feat + scale_unetfeats * 2)
        self.cdgm_3_R = CDGM(n_feat + scale_unetfeats * 2)

        self.DSDC_D3 = CGB(n_feat + scale_unetfeats * 2, kernel_size, reduction, bias=bias, act=act, dilation=4)
        self.decoder_level3 = CMB(in_chans=n_feat + scale_unetfeats * 2,
                                   embed_dim=n_feat + scale_unetfeats * 2,
                                   depths=[Depth[2]])
        self.dfa_D3 = DynamicReflectionAttention(n_feat + scale_unetfeats * 2, head3)
        self.cdgm_D3_Y = CDGM(n_feat + scale_unetfeats * 2)
        self.cdgm_D3_R = CDGM(n_feat + scale_unetfeats * 2)

        self.up32_Y = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)
        self.up32_R = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)
        self.DSDC_D2 = CGB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act, dilation=2)
        self.decoder_level2 = CMB(in_chans=n_feat + scale_unetfeats,
                                   embed_dim=n_feat + scale_unetfeats,
                                   depths=[Depth[1]])
        self.dfa_D2 = DynamicReflectionAttention(n_feat + scale_unetfeats, head2)
        self.cdgm_D2_Y = CDGM(n_feat + scale_unetfeats)
        self.cdgm_D2_R = CDGM(n_feat + scale_unetfeats)

        self.up21_Y = SkipUpSample(n_feat, scale_unetfeats)
        self.up21_R = SkipUpSample(n_feat, scale_unetfeats)
        self.DSDC_D1 = CGB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.decoder_level1 = CMB(in_chans=n_feat,
                                   embed_dim=n_feat,
                                   depths=[Depth[0]])

        self.lcat = LCAT(n_feat)
        self.upsample_Y = nn.Sequential(nn.Conv2d(n_feat, 3 * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.upsample = nn.Sequential(nn.Conv2d(n_feat, 3 * 4, 3, 1, 1), nn.PixelShuffle(2))

    def forward(self, input):
        MoireRaw = input['MoireRaw']      

        YCrCb = self.raw2YCbCr(MoireRaw)
        MoireRaw = torch.cat([MoireRaw, YCrCb], dim=1) 

        MoireRaw_Feature_intro = self.shallow_feat1(MoireRaw)
        YCrCb_Feature_intro = self.shallow_feat2(YCrCb)
        
        YCrCb_Feature_enc1 = self.DSDC_1(YCrCb_Feature_intro)
        MoireRaw_Feature_enc1 = self.encoder_level1(MoireRaw_Feature_intro)
        MoireRaw_Feature = self.down12_R(MoireRaw_Feature_enc1)    
        YCrCb_Feature = self.down12_Y(YCrCb_Feature_enc1)    
        
        YCrCb_Feature_enc2 = self.DSDC_2(YCrCb_Feature)
        MoireRaw_Feature_enc2 = self.encoder_level2(MoireRaw_Feature)
        Fused_enc2 = self.dfa_2(MoireRaw_Feature_enc2, YCrCb_Feature_enc2)
        YCrCb_Feature_enc2 = self.cdgm_2_Y(YCrCb_Feature_enc2, Fused_enc2)
        MoireRaw_Feature_enc2 = self.cdgm_2_R(MoireRaw_Feature_enc2, Fused_enc2)
        
        YCrCb_Feature = self.down23_Y(YCrCb_Feature_enc2)         
        MoireRaw_Feature = self.down23_R(MoireRaw_Feature_enc2)
        YCrCb_Feature_enc3 = self.DSDC_3(YCrCb_Feature)
        MoireRaw_Feature_enc3 = self.encoder_level3(MoireRaw_Feature)
        Fused_enc3 = self.dfa_3(MoireRaw_Feature_enc3, YCrCb_Feature_enc3)
        YCrCb_Feature_enc3 = self.cdgm_3_Y(YCrCb_Feature_enc3, Fused_enc3)
        MoireRaw_Feature_enc3 = self.cdgm_3_R(MoireRaw_Feature_enc3, Fused_enc3)

        YCrCb_Feature_dec3 = self.DSDC_D3(YCrCb_Feature_enc3)
        MoireRaw_Feature_dec3 = self.decoder_level3(MoireRaw_Feature_enc3)
        Fused_dec3 = self.dfa_D3(MoireRaw_Feature_dec3, YCrCb_Feature_dec3)
        YCrCb_Feature_dec3 = self.cdgm_D3_Y(YCrCb_Feature_dec3, Fused_dec3)
        MoireRaw_Feature_dec3 = self.cdgm_D3_R(MoireRaw_Feature_dec3, Fused_dec3)
        
        YCrCb_Feature = self.up32_Y(YCrCb_Feature_dec3, YCrCb_Feature_enc2)
        MoireRaw_Feature = self.up32_R(MoireRaw_Feature_dec3, MoireRaw_Feature_enc2)
        YCrCb_Feature_dec2 = self.DSDC_D2(YCrCb_Feature)
        MoireRaw_Feature_dec2 = self.decoder_level2(MoireRaw_Feature)
        Fused_dec2 = self.dfa_D2(MoireRaw_Feature_dec2, YCrCb_Feature_dec2)
        YCrCb_Feature_dec2 = self.cdgm_D2_Y(YCrCb_Feature_dec2, Fused_dec2)            
        MoireRaw_Feature_dec2 = self.cdgm_D2_R(MoireRaw_Feature_dec2, Fused_dec2)   
        
        YCrCb_Feature = self.up21_Y(YCrCb_Feature_dec2, YCrCb_Feature_enc1)
        YCrCb_Feature_dec1 = self.DSDC_D1(YCrCb_Feature) + YCrCb_Feature_intro
        
        MoireRaw_Feature = self.up21_R(MoireRaw_Feature_dec2, MoireRaw_Feature_enc1)
        MoireRaw_Feature_dec1 = self.decoder_level1(MoireRaw_Feature) + MoireRaw_Feature_intro

        MoireRaw_Feature_dec1 = self.lcat(MoireRaw_Feature_dec1, YCrCb_Feature_dec1)

        DemoireYCrCb = self.upsample_Y(YCrCb_Feature_dec1)

        DemoireRGB = self.upsample(MoireRaw_Feature_dec1) 


        return DemoireRGB, DemoireYCrCb


    def check_image_size(self, x):
        self.padder_size = 8
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x