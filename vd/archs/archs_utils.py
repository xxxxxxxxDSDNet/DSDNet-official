import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=((kernel_size - 1) * dilation // 2), bias=bias, stride=stride, dilation=dilation)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) 


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class DynamicReflectionAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = 4
        self.temperature = nn.Parameter(torch.ones(4, 1, 1))

        self.conv_raw_kv = nn.Conv2d(d_model, d_model * 2, kernel_size=1, bias=False)
        self.conv_rgb_q = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)

        self.conv_rgb_kv = nn.Conv2d(d_model, d_model * 2, kernel_size=1, bias=False)
        self.conv_raw_q = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)
        
        self.gate_net = nn.Sequential(
            nn.Conv2d(2 * d_model, 4 * d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4 * d_model, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.fc_out = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.feedforward = FeedForward(d_model, ffn_expansion_factor=2.667, bias=False)
        
        self.norm_g1 = LayerNorm(d_model, LayerNorm_type='WithBias')
        self.norm_g2 = LayerNorm(d_model, LayerNorm_type='WithBias')
        self.norm_ff1 = LayerNorm(d_model, LayerNorm_type='WithBias')

    def forward(self, F_raw, F_Y):
        F_raw = self.norm_g1(F_raw)
        F_Y = self.norm_g1(F_Y)
        
        raw_kv = self.conv_raw_kv(F_raw)
        raw_q = self.conv_raw_q(F_raw)

        rgb_kv = self.conv_rgb_kv(F_Y)
        rgb_q = self.conv_rgb_q(F_Y)
        
        _, _, h, w = raw_kv.shape
        raw_k, raw_v = raw_kv.chunk(2, dim=1)
        rgb_k, rgb_v = rgb_kv.chunk(2, dim=1)

        raw_q = rearrange(raw_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        raw_k = rearrange(raw_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        raw_v = rearrange(raw_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        rgb_q = rearrange(rgb_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        rgb_k = rearrange(rgb_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        rgb_v = rearrange(rgb_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        raw_q = torch.nn.functional.normalize(raw_q, dim=-1)
        raw_k = torch.nn.functional.normalize(raw_k, dim=-1)

        rgb_q = torch.nn.functional.normalize(rgb_q, dim=-1)
        rgb_k = torch.nn.functional.normalize(rgb_k, dim=-1)

        attn_raw = (rgb_q @ raw_k.transpose(-2, -1).contiguous()) * self.temperature
        attn_raw = attn_raw.softmax(dim=-1)

        attn_rgb = (raw_q @ rgb_k.transpose(-2, -1).contiguous()) * self.temperature
        attn_rgb = attn_rgb.softmax(dim=-1)

        out_raw = (attn_raw @ raw_v)
        out_rgb = (attn_rgb @ rgb_v)
        
        fused_feature = torch.cat([F_raw, F_Y], dim=1)
        alpha = self.gate_net(fused_feature)    
        alpha = rearrange(alpha, 'b () h w -> b 1 1 (h w)') 
        
        context = alpha * out_raw + (1 - alpha) * out_rgb
        
        context = rearrange(context, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        
        out = self.fc_out(context) + F_raw + F_Y
        out = self.feedforward(self.norm_ff1(out)) + out
        return out
    
class CDGM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(channels, channels//2, 1),
            nn.ReLU(),
            nn.Conv2d(channels//2, channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, main_feat, aux_feat):
        gate = self.sigmoid(self.projection(aux_feat))
        gated_main = main_feat * gate
        
        return gated_main + aux_feat
    
class CGB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, dilation=1):
        super(CGB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias, dilation=dilation))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias, dilation=dilation))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x
    
class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv_5 = nn.Conv2d(hidden_features//4, hidden_features//4, kernel_size=5, stride=1, padding=2, groups=hidden_features//4, bias=bias)
        self.dwconv_dilated2_1 = nn.Conv2d(hidden_features//4, hidden_features//4, kernel_size=3, stride=1, padding=2, groups=hidden_features//4, bias=bias, dilation=2)
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)

        x1, x2 = x.chunk(2, dim=1)

        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1( x2 )

        x = F.mish( x2 ) * x1
        x = self.p_unshuffle(x)
        x = self.project_out(x)

        return x
    
class LCAT(nn.Module):
    def __init__(self, n_feat):
        super(LCAT, self).__init__()
        self.num_heads = 4
        self.temperature = nn.Parameter(torch.ones(4, 1, 1))

        self.projection = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, 1),
            nn.ReLU(),
            nn.Conv2d(n_feat//2, n_feat, 1)
        )
        self.sigmoid = nn.Sigmoid()
        
        self.conv_ycbcr = nn.Conv2d(n_feat, n_feat * 2, kernel_size=1, bias=False)
        self.conv_rgb = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=False)
        self.project_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=False)
        self.feedforward = FeedForward(n_feat, ffn_expansion_factor=2.667, bias=False)
        
        self.norm_g1 = LayerNorm(n_feat, LayerNorm_type='WithBias')
        self.norm_g2 = LayerNorm(n_feat, LayerNorm_type='WithBias')
        self.norm_ff1 = LayerNorm(n_feat, LayerNorm_type='WithBias')
        
    def forward(self, rgb, ycbcr):
        rgb = self.norm_g1(rgb)
        ycbcr = self.norm_g2(ycbcr)

        gate = self.sigmoid(self.projection(rgb))
        gated_main = ycbcr * gate + rgb
 
        qk = self.conv_ycbcr(gated_main)
        v = self.conv_rgb(rgb)

        _, _, h, w = qk.shape
        q, k = qk.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out) + rgb
        out = self.feedforward(self.norm_ff1(out))  + out
        return out
    
class RGB2YCbCr(nn.Module):
    def __init__(self, device='cuda'):
        super(RGB2YCbCr, self).__init__()
        
        self.rgb_to_ycbcr_matrix = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ], dtype=torch.float32).to(device)

        self.device = device

    def forward(self, raw):
        
        
        R = raw[:, 0:1, :, :]  
        G = (raw[:, 1:2, :, :] + raw[:, 2:3, :, :]) / 2  
        B = raw[:, 3:4, :, :]  

        RGB = torch.cat([R, G, B], dim=1)

        img = RGB.permute(0, 2, 3, 1).contiguous()
        
        ycbcr_img = torch.tensordot(img, self.rgb_to_ycbcr_matrix, dims=([-1], [1]))
        
        ycbcr_img[..., 1:3] += 0.5
        
        ycbcr_img = ycbcr_img.permute(0, 3, 1, 2).contiguous()
        
        return ycbcr_img