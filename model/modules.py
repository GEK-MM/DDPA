import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 基础组件 (Utility Components)
# ==========================================
class SineCosinePositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 不再在 __init__ 里 register_buffer 存 pe

    def forward(self, x):
        # x shape: [B, N, C], 其中 N = H * W
        B, N, C = x.shape
        H = int(math.sqrt(N))
        W = N // H

        # 动态生成 PE (在 forward 里生成，不占模型权重)
        device = x.device
        pe = torch.zeros(C, H, W, device=device)
        d_half = C // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_half, 2, device=device).float() / d_half))

        pos_h = torch.arange(H, device=device).float()
        pos_w = torch.arange(W, device=device).float()

        out_h = (pos_h[:, None] * inv_freq[None, :]).permute(1, 0).unsqueeze(2)
        pe[0:d_half:2, :, :] = torch.sin(out_h).repeat(1, 1, W)
        pe[1:d_half:2, :, :] = torch.cos(out_h).repeat(1, 1, W)

        out_w = (pos_w[:, None] * inv_freq[None, :]).permute(1, 0).unsqueeze(1)
        pe[d_half::2, :, :] = torch.sin(out_w).repeat(1, H, 1)
        pe[d_half + 1::2, :, :] = torch.cos(out_w).repeat(1, H, 1)

        # 叠加并返回
        return x + pe.flatten(1).transpose(0, 1).unsqueeze(0)


# GLU 确保隐藏层取整
class GLU(nn.Module):
    def __init__(self, dim, expansion_factor=2.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.gate = nn.Linear(dim, hidden_dim)
        self.linear = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.gate(x) * self.act(self.linear(x))
        return self.out(self.dropout(x))


class ASPP(nn.Module):
    """
    针对遥感图像优化的多尺度特征提取
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        rates: 空洞率列表 (不包含1x1分支和GAP分支)
        use_gap: 是否使用全局平均池化分支 (建议只在最低分辨率层使用)
    """

    def __init__(self, in_channels, out_channels, rates=[1, 2, 3], use_gap=False):
        super().__init__()
        self.use_gap = use_gap

        # 空洞卷积分支 (使用标准卷积，保证精度)
        self.branches = nn.ModuleList()
        for rate in rates:
            padding = rate  # 保持分辨率
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=padding, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # 1x1 卷积分支 (保持空间细节)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 全局平均池化分支 (可选，只在最低分辨率使用)
        if use_gap:
            self.global_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # 投影层融合所有分支
        num_branches = len(rates) + 1  # 空洞分支 + 1x1分支
        if use_gap:
            num_branches += 1  # + GAP分支

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 收集所有分支输出
        res = [branch(x) for branch in self.branches]  # 空洞卷积分支
        res.append(self.conv1x1(x))  # 1x1 分支

        if self.use_gap:
            # GAP 分支需要上采样回原尺寸
            gp = self.global_pool(x)
            gp = F.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=False)
            res.append(gp)

        # 拼接并投影
        return self.project(torch.cat(res, dim=1))


class GatedInjection(nn.Module):
    """跨尺度特征门控注入（轻量版）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 门控：先压缩再扩张，减少参数
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, low_feat, high_feat):
        high_feat_up = self.proj(self.up(high_feat))
        gate = self.gate(torch.cat([low_feat, high_feat_up], dim=1))
        return low_feat * gate + high_feat_up * (1 - gate)


class ChannelSelector(nn.Module):
    """多模态特征通道动态筛选器"""

    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.filter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, v1, v2):
        x = torch.cat([v1, v2], dim=1)
        return self.project(x * self.filter(self.avg_pool(x)))


class GroupASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[1, 2, 3, 5], groups=4):
        super(GroupASPP, self).__init__()

        assert in_channels % groups == 0, f"in_channels ({in_channels}) must be divisible by groups ({groups})"
        assert out_channels % groups == 0, f"out_channels ({out_channels}) must be divisible by groups ({groups})"

        self.branches = nn.ModuleList()

        for rate in rates:
            if rate == 1:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate,
                              dilation=rate, groups=groups, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            self.branches.append(branch)

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 1), out_channels,
                      kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 残差路径：用标准 1x1 卷积，保证信息完整传递
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # 不分组的 1x1
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = x
        size = x.shape[2:]

        res = [branch(x) for branch in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode='bilinear', align_corners=False)
        res.append(gp)

        out = torch.cat(res, dim=1)
        out = self.project(out)

        return F.relu(out + self.shortcut(identity), inplace=True)


# ==========================================
# 2. 核心交互模块 (Co-Attention)
# ==========================================

class Co_Attention(nn.Module):
    def __init__(self, img_dim=768, text_dim=512, n_heads=8,
                 img_h=14, img_w=14, dropout=0.1,
                 use_aspp=True, use_ti=True, use_self_attn=True,
                 rate=None, use_group_aspp=False, groups=4,
                 expansion_factor=2.0, use_pos=True):
        super().__init__()
        self.img_h, self.img_w = img_h, img_w
        self.img_dim, self.text_dim = img_dim, text_dim
        self.use_aspp = use_aspp
        self.use_ti = use_ti
        self.use_self_attn = use_self_attn
        self.use_group_aspp = use_group_aspp
        self.use_pos = use_pos

        if self.use_aspp:
            if self.use_group_aspp:
                self.aspp = GroupASPP(img_dim, text_dim, rates=rate or [1, 2, 3, 5], groups=groups)
            else:
                self.aspp = ASPP(img_dim, text_dim, rates=rate or [1, 2, 3, 5])

        input_dim = text_dim if self.use_aspp else img_dim
        self.img_proj = nn.Linear(input_dim, text_dim)

        if self.use_pos:
            self.img_pos_enc = SineCosinePositionalEncoding2D(text_dim)  # 只传维度

        if self.use_ti:
            self.cross_attn_txt2img = nn.MultiheadAttention(text_dim, n_heads, dropout=dropout, batch_first=True)
            self.txt_norm1 = nn.LayerNorm(text_dim)
            self.txt_ffn = GLU(text_dim, expansion_factor=expansion_factor, dropout=dropout)
            self.txt_ffn_norm = nn.LayerNorm(text_dim)

        self.cross_attn_img2txt = nn.MultiheadAttention(text_dim, n_heads, dropout=dropout, batch_first=True)
        self.img_cross_norm = nn.LayerNorm(text_dim)

        if self.use_self_attn:
            self.img_self_attn = nn.MultiheadAttention(text_dim, n_heads, dropout=dropout, batch_first=True)
            self.img_self_norm = nn.LayerNorm(text_dim)

        self.img_ffn = GLU(text_dim, expansion_factor=expansion_factor, dropout=dropout)
        self.img_ffn_norm = nn.LayerNorm(text_dim)
        self.img_out_proj = nn.Linear(text_dim, img_dim)

    def forward(self, img_feats, text_embeds, text_mask=None):
        B, C_img, H, W = img_feats.shape
        if self.use_aspp:
            img_feats = self.aspp(img_feats)

        img_seq = img_feats.flatten(2).transpose(1, 2)
        img_seq = self.img_proj(img_seq)

        if self.use_pos:
            img_seq = self.img_pos_enc(img_seq)

        if self.use_ti:
            txt2img, _ = self.cross_attn_txt2img(text_embeds, img_seq, img_seq)
            text_refined = self.txt_norm1(text_embeds + txt2img)
            text_refined = self.txt_ffn_norm(text_refined + self.txt_ffn(text_refined))
        else:
            text_refined = text_embeds

        img2txt, _ = self.cross_attn_img2txt(img_seq, text_refined, text_refined, key_padding_mask=text_mask)
        img_seq = self.img_cross_norm(img_seq + img2txt)

        if self.use_self_attn:
            img_self, _ = self.img_self_attn(img_seq, img_seq, img_seq)
            img_seq = self.img_self_norm(img_seq + img_self)

        img_seq = self.img_ffn_norm(img_seq + self.img_ffn(img_seq))
        img_feats_out = self.img_out_proj(img_seq).transpose(1, 2).view(B, self.img_dim, H, W)
        return img_feats_out, text_refined


# ==========================================
# 3. 主分支 (Main Branch)
# ==========================================

class Main_Branch(nn.Module):
    def __init__(self, img_dims=[128, 256, 512, 1024], text_dim=768, n_heads=8, img_sizes=[128, 64, 32, 16],
                 dropout=0.1, dropout_act=0.0):
        super().__init__()
        self.img_dims = img_dims
        self.text_dim = text_dim
        self.drop = nn.Dropout2d(p=dropout)

        self.feature_norms = nn.ModuleList([
            nn.GroupNorm(16, img_dims[0]),
            nn.GroupNorm(32, img_dims[1]),
            nn.GroupNorm(32, img_dims[2]),
            nn.GroupNorm(32, img_dims[3])
        ])

        self.v3_selector = ChannelSelector(img_dims[3] * 2, img_dims[3])
        self.v2_selector = ChannelSelector(img_dims[2] * 2, img_dims[2])

        # --- S3 阶段 (16×16): 恢复 1.5x 增强区分度 ---
        self.c_interact_s3 = Co_Attention(
            img_dims[3], text_dim, n_heads, img_sizes[3], img_sizes[3],
            use_self_attn=True, use_aspp=True, use_group_aspp=True, groups=4,
            rate=[1, 3, 5], expansion_factor=1.5, use_pos=True
        )
        self.t_interact_s3 = Co_Attention(
            img_dims[3], text_dim, n_heads, img_sizes[3], img_sizes[3],
            use_self_attn=True, use_aspp=True, use_group_aspp=True, groups=4,
            rate=[1, 2, 3], expansion_factor=1.5, use_pos=True
        )

        # --- S2 阶段 (32×32): 保持 1.5x ---
        self.c_interact_s2 = Co_Attention(
            img_dims[2], text_dim, n_heads, img_sizes[2], img_sizes[2],
            use_self_attn=False, use_aspp=True, use_group_aspp=True, groups=4,
            rate=[1, 3, 5], expansion_factor=1.5, use_pos=True
        )
        self.t_interact_s2 = Co_Attention(
            img_dims[2], text_dim, n_heads, img_sizes[2], img_sizes[2],
            use_self_attn=False, use_aspp=True, use_group_aspp=True, groups=4,
            rate=[1, 2, 3], expansion_factor=1.5, use_pos=True
        )

        # --- S1 阶段 (64×64): 保持 2.0x, 开启绝对位置编码 ---
        self.t_interact_s1 = Co_Attention(
            img_dims[1], text_dim, n_heads, img_sizes[1], img_sizes[1],
            use_self_attn=False, use_aspp=True, use_group_aspp=True, groups=4,
            rate=[1, 2, 3, 5], expansion_factor=2.0, use_pos=True
        )

        # --- S0 阶段处理: 新增底层 ASPP 预过滤噪声 ---
        self.s0_aspp = ASPP(img_dims[0], img_dims[0], rates=[1, 2, 4])
        # Grid 生成器保持不变
        self.grid_gen_c3 = self._make_grid_gen(img_dims[3], 1, 0, dropout=dropout_act)
        self.gate_c3 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Sigmoid())
        self.grid_gen_t3 = self._make_grid_gen(img_dims[3], 1, 0, dropout=dropout_act)
        self.gate_t3 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Sigmoid())

        self.grid_gen_c2 = self._make_grid_gen(img_dims[2], 3, 1, dropout=dropout_act)
        self.gate_c2 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Sigmoid())
        self.grid_gen_t2 = self._make_grid_gen(img_dims[2], 3, 1, dropout=dropout_act)
        self.gate_t2 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Sigmoid())

        self.grid_gen_t1 = self._make_grid_gen(img_dims[1], 3, 1, ratio=8, dropout=dropout_act)
        self.gate_t1 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Sigmoid())

        self.inj_3to2 = GatedInjection(img_dims[3], img_dims[2])
        self.inj_2to1 = GatedInjection(img_dims[2], img_dims[1])
        self.proj_1to0 = nn.Conv2d(img_dims[1], img_dims[0], kernel_size=1)

    def _make_grid_gen(self, in_dim, kernel_size, padding, dropout=0.0, ratio=16):
        return nn.Sequential(
            nn.Conv2d(in_dim, in_dim // ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(in_dim // ratio, 1, kernel_size=kernel_size, padding=padding),
            nn.Tanh()
        )

    def forward(self, feats, text_embeds, text_mask=None):
        f0, f1, f2, f3 = feats
        target_text = text_embeds[:, 0, :, :].mean(dim=1, keepdim=True)
        constrain_text = text_embeds[:, 1, :, :]

        # S3 阶段
        v3_c_raw, _ = self.c_interact_s3(f3, constrain_text, text_mask)
        g3_c = self.grid_gen_c3(v3_c_raw)
        v3_c = v3_c_raw + v3_c_raw * (g3_c * self.gate_c3(g3_c))
        v3_t_raw, _ = self.t_interact_s3(f3, target_text, text_mask)
        g3_t = self.grid_gen_t3(v3_t_raw)
        v3_t = v3_t_raw + v3_t_raw * (g3_t * self.gate_t3(g3_t))
        v3_final = self.drop(self.v3_selector(v3_t, v3_c))

        # S2 阶段
        f2_injected = self.drop(self.inj_3to2(f2, v3_final))
        v2_c_raw, _ = self.c_interact_s2(f2_injected, constrain_text, text_mask)
        g2_c = self.grid_gen_c2(v2_c_raw)
        v2_c = v2_c_raw + v2_c_raw * (g2_c * self.gate_c2(g2_c))
        v2_t_raw, _ = self.t_interact_s2(f2_injected, target_text, text_mask)
        g2_t = self.grid_gen_t2(v2_t_raw)
        v2_t = v2_t_raw + v2_t_raw * (g2_t * self.gate_t2(g2_t))
        v2_final = self.drop(self.v2_selector(v2_t, v2_c))

        # S1 阶段
        f1_injected = self.drop(self.inj_2to1(f1, v2_final))
        v1_t_raw, t1_t = self.t_interact_s1(f1_injected, target_text, text_mask)
        g1_t = self.grid_gen_t1(v1_t_raw)
        v1_t = v1_t_raw + v1_t_raw * (g1_t * self.gate_t1(g1_t))

        # S0 阶段增强
        v1_t_up = F.interpolate(v1_t, size=f0.shape[2:], mode='bilinear', align_corners=False)
        f0_refined = self.s0_aspp(f0)  # 预过滤底层噪声
        f0_injected = f0_refined + self.proj_1to0(v1_t_up)

        out_feats = [f0_injected, v1_t, v2_final, v3_final]
        out_feats = [norm(f) for norm, f in zip(self.feature_norms, out_feats)]

        return out_feats, t1_t, [
            g3_c * self.gate_c3(g3_c), g3_t * self.gate_t3(g3_t),
            g2_c * self.gate_c2(g2_c), g2_t * self.gate_t2(g2_t),
            g1_t * self.gate_t1(g1_t)
        ]
