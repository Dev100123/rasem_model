# Copyright (c) OpenMMLab.
# AFNO adaptation of SegNeXt MSCAN-style backbone.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


def trunc_normal_init(module, std=0.02, bias=0.0):
    trunc_normal_(module.weight, std=std)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val=1.0, bias=0.0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0.0, std=1.0, bias=0.0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean=mean, std=std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def build_activation_layer(act_cfg):
    act_type = act_cfg.get("type", "GELU")
    if act_type == "GELU":
        return nn.GELU()
    if act_type == "ReLU":
        return nn.ReLU(inplace=True)
    if act_type == "ReLU6":
        return nn.ReLU6(inplace=True)
    raise ValueError(f"Unsupported activation type: {act_type}")


def build_norm_layer(norm_cfg, num_features):
    norm_type = norm_cfg.get("type", "BN")
    if norm_type in ["SyncBN", "BN"]:
        return norm_type, nn.BatchNorm2d(num_features)
    raise ValueError(f"Unsupported norm type: {norm_type}")


# -------------------------
# Backbone blocks
# -------------------------
class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels, act_cfg=dict(type="GELU"), norm_cfg=dict(type="SyncBN")):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            build_activation_layer(act_cfg),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x, H, W


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=3, stride=2, in_channels=3, embed_dim=768, norm_cfg=dict(type="SyncBN")):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x, H, W


class DWConv2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class Mlp2D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv2D(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=0.02, bias=0.0)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m, val=1.0, bias=0.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            normal_init(m, mean=0.0, std=math.sqrt(2.0 / fan_out), bias=0.0)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1.0, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.sparsity_threshold = float(sparsity_threshold)
        self.hard_fraction = float(hard_thresholding_fraction)
        self.hidden_size_factor = int(hidden_size_factor)

        scale = 0.02
        self.w1 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size))

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W
        bias = x

        x = x.float().view(B, H, W, C)
        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")

        x_real = x_fft.real.view(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        x_imag = x_fft.imag.view(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        total_modes = W // 2 + 1
        kept_modes = max(1, int(total_modes * self.hard_fraction))

        o1_real = torch.zeros(B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor, device=x.device)
        o1_imag = torch.zeros_like(o1_real)

        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x_real[:, :, :kept_modes], self.w1[0])
            - torch.einsum("...bi,bio->...bo", x_imag[:, :, :kept_modes], self.w1[1])
            + self.b1[0]
        )
        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x_imag[:, :, :kept_modes], self.w1[0])
            + torch.einsum("...bi,bio->...bo", x_real[:, :, :kept_modes], self.w1[1])
            + self.b1[1]
        )

        o2_real = torch.zeros(B, H, W // 2 + 1, self.num_blocks, self.block_size, device=x.device)
        o2_imag = torch.zeros_like(o2_real)

        o2_real[:, :, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_real[:, :, :kept_modes], self.w2[0])
            - torch.einsum("...bi,bio->...bo", o1_imag[:, :, :kept_modes], self.w2[1])
            + self.b2[0]
        )
        o2_imag[:, :, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_imag[:, :, :kept_modes], self.w2[0])
            + torch.einsum("...bi,bio->...bo", o1_real[:, :, :kept_modes], self.w2[1])
            + self.b2[1]
        )

        x_real = F.softshrink(o2_real, lambd=self.sparsity_threshold)
        x_imag = F.softshrink(o2_imag, lambd=self.sparsity_threshold)
        x_complex = torch.complex(x_real, x_imag).view(B, H, W // 2 + 1, C)

        x_out = torch.fft.irfft2(x_complex, s=(H, W), dim=(1, 2), norm="ortho")
        x_out = x_out.view(B, N, C).type_as(bias)
        return x_out + bias


class Block2D(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm,
                 afno_num_blocks=8, afno_sparsity_threshold=0.01, afno_hard_thresholding_fraction=1.0,
                 afno_hidden_size_factor=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.afno = AFNO2D(dim, afno_num_blocks, afno_sparsity_threshold, afno_hard_thresholding_fraction, afno_hidden_size_factor)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp2D(dim, int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.afno(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class RASEMEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dims=[64, 128, 320, 512], mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.0, drop_path_rate=0.0, depths=[3, 4, 6, 3], num_stages=4,
                 afno_num_blocks=8, afno_hard_thresholding_fraction=1.0,
                 afno_sparsity_threshold=0.01, afno_hidden_size_factor=1,
                 act_cfg=dict(type="GELU"), norm_cfg=dict(type="SyncBN")):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            patch_embed = StemConv(in_channels, embed_dims[0], act_cfg, norm_cfg) if i == 0 else OverlapPatchEmbed(
                patch_size=3, stride=2, in_channels=embed_dims[i - 1], embed_dim=embed_dims[i], norm_cfg=norm_cfg
            )

            blocks = nn.ModuleList([
                Block2D(
                    dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j],
                    norm_layer=nn.LayerNorm, afno_num_blocks=afno_num_blocks,
                    afno_hard_thresholding_fraction=afno_hard_thresholding_fraction,
                    afno_sparsity_threshold=afno_sparsity_threshold,
                    afno_hidden_size_factor=afno_hidden_size_factor
                ) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm)

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            blocks = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            for blk in blocks:
                x = blk(x, H, W)

            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs





class Head(nn.Module):
    def __init__(self,in_feature, out_feature,kernel_size=3, stride=1, dilation=1, drop_out=0.):
        super(Head,self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_feature, in_feature, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_feature, bias=False),
            nn.BatchNorm2d(in_feature),
            nn.Conv2d(in_feature, in_feature, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_feature),
            nn.ReLU6(),
            nn.Conv2d(in_feature, out_feature, kernel_size=1, bias=False))
        
    def forward(self,x):

        return self.head(x)


class FPN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        c1, c2, c3, c4 = in_channels
        self.lateral4 = nn.Conv2d(c4, in_channels[-3], 1)
        self.lateral3 = nn.Conv2d(c3, in_channels[-3], 1)
        self.lateral2 = nn.Conv2d(c2, in_channels[-3], 1)
        self.lateral1 = nn.Conv2d(c1, in_channels[-3], 1)
        self.output = nn.Conv2d(in_channels[-3], in_channels[-2], kernel_size=3, stride=1, padding=1)
        self.drop =nn.Dropout2d(0.3)


    def forward(self, c1, c2, c3, c4):
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="bilinear", align_corners=False)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[2:], mode="bilinear", align_corners=False)
        final_out = self.output(p1)
        return self.drop(final_out)




class Decoder_spatial_onehead(nn.Module):
    def __init__(self, num_class = 2, feature_list=None, drop_out=0.1):
        super(Decoder_spatial_onehead, self).__init__()
        #we use fpn

        self.head = Head(in_feature=feature_list[-2],out_feature=num_class,drop_out=drop_out)
        self.fpn = FPN(feature_list)

    def forward(self, c1, c2, c3,c4):
        fpn_out = self.fpn(c1, c2, c3, c4)
        seg_out_final = self.head(fpn_out)
        return fpn_out,seg_out_final
    
class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=1, padding=0, dilation=1,
                      bias=False),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class SqueezeBodyEdge(nn.Module):
    def __init__(self, inplane):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(SqueezeBodyEdge, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            # norm_layer(inplane),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            # norm_layer(inplane),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True)
        )
        self.flow_make = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        seg_down = self.down(x)
        seg_down = F.interpolate(seg_down, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output
    

class Decoder_edgeV3(nn.Module):
    def __init__(self):
        super(Decoder_edgeV3,self).__init__()
        self.ppm = PSPModule(features=512,out_features=256)
        self.squeeze_body_edge = SqueezeBodyEdge(256)
        self.edge_out = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.refine_out4x =  nn.Conv2d(64, 48, kernel_size=1, bias=False)
        self.edge_4x_fusion = nn.Conv2d(256 + 48, 256, 1, bias=False)
    def forward(self,out_32x,out_4x):
        out_32x_ppm =self.ppm(out_32x)
        seg_body, seg_edge = self.squeeze_body_edge(out_32x_ppm)
        refine_out4x = self.refine_out4x(out_4x)
        seg_edge = F.interpolate(seg_edge,scale_factor=8, mode='bilinear', align_corners=False)
        seg_edge = self.edge_4x_fusion(torch.cat([seg_edge, refine_out4x], dim=1))
        seg_edge_out = self.sigmoid(F.interpolate(self.edge_out(seg_edge),scale_factor=4, mode='bilinear', align_corners=False))
    
        seg_out = seg_edge + F.interpolate(seg_body,scale_factor=8, mode='bilinear', align_corners=False)
        aspp = F.interpolate(out_32x_ppm,scale_factor=8, mode='bilinear', align_corners=False)
        edge_feature = torch.cat([aspp, seg_out], dim=1)
        return edge_feature,seg_edge_out
    

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(out_chan),
                                  nn.ReLU6())
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

class Decoder_text(nn.Module):
    def __init__(self):
        super(Decoder_text,self).__init__()

        self.arm16 = AttentionRefinementModule(320, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU6())
        self.conv_head16 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU6())
        self.conv_avg = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU6())
    def forward(self,feat8,feat16,feat32):
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)
        feat16_up = F.interpolate(feat16_up,scale_factor = 2, mode='nearest')

        return feat16_up
    

    
class FeatureFusionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule,self).__init__()
        self.in_channels = in_channels
        self.convblock =nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU6())
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(
            1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class Head_4x(nn.Module):
    def __init__(self,in_feature, out_feature,kernel_size=3, stride=1, dilation=1, drop_out=0.):
        super(Head_4x,self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_feature, 256, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(),
            nn.Conv2d(256, out_feature, kernel_size=1, bias=False))
        
    def forward(self,x):
        
        return self.head(x)


class RASEMDecodeHead(nn.Module):
    def __init__(self, in_channels=(64, 128, 320, 512), num_classes=19):
        super().__init__()               
        c1, c2, c3, c4 = in_channels
        self.decoder_spatial = Decoder_spatial_onehead(num_class=2, feature_list=[c1,c2,c3,c4], drop_out=0.1)
        self.decoder_edge = Decoder_edgeV3()
        self.decoder_text = Decoder_text()
        self.feature_fusion = FeatureFusionModule(in_channels=448,out_channels=128)
        self.head = Head_4x(in_feature = 640,out_feature=num_classes)

    def forward(self,inputs):
        x1, x2, x3, x4 = inputs
        seg_out,_ = self.decoder_spatial(x1, x2, x3, x4)
        edge_out,_ = self.decoder_edge(x4,x1)
        text_out = self.decoder_text(x2,x3,x4)
        seg_atten_out = self.feature_fusion(seg_out,text_out)
        fusion_feature = torch.cat((seg_atten_out,edge_out),dim = 1)
        out = self.head(fusion_feature)
        out = F.interpolate(out,scale_factor = 4,mode='bilinear', align_corners=False)
        return out



class RASEM(nn.Module):
    def __init__(self, in_channels=3, num_classes=116,
                 embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4], depths=[3, 5, 27, 3],
                 drop_rate=0.0, drop_path_rate=0.1, afno_num_blocks=8,
                 afno_hard_thresholding_fraction=1.0, afno_sparsity_threshold=0.0):
        super().__init__()
        self.encoder = RASEMEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            mlp_ratios=mlp_ratios,
            depths=depths,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            afno_num_blocks=afno_num_blocks,
            afno_hard_thresholding_fraction=afno_hard_thresholding_fraction,
            afno_sparsity_threshold=afno_sparsity_threshold,
        )
        self.decode_head = RASEMDecodeHead(
            in_channels=(embed_dims),
            num_classes=num_classes
        )

    def forward(self, x):
        feats = self.encoder(x)        
        logits = self.decode_head(feats)  
        return logits