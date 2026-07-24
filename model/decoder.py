# import torch.nn as nn
# from model.modules import ConvBatchNormReLU
# import torch.nn as nn
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from model.modules import ConvBatchNormReLU
#
#
# class mask_decoder(nn.Module):
#     def __init__(self, encoder_dim, v1_dim, state0_dim, img_size=406, leaky=True, dropout_rate=0):
#         super().__init__()
#
#         # 1. 第一阶段融合 (Encoder 32->64 + v1 64)
#         mid_c1 = 512
#         self.stem = ConvBatchNormReLU(encoder_dim, mid_c1, 3, 1, 1, 1, leaky=leaky)
#         self.drop1 = nn.Dropout2d(p=dropout_rate)
#
#         # 2. 第一次上采样 64 -> 128
#         self.up_to_128 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             ConvBatchNormReLU(mid_c1, 256, 3, 1, 1, 1, leaky=leaky)
#         )
#
#         # 3. 第二阶段融合 (x 128 + state0 128)
#         mid_c2 = 128
#         self.fusion_state0 = ConvBatchNormReLU(256 + state0_dim, mid_c2, 3, 1, 1, 1, leaky=leaky)
#         self.drop2 = nn.Dropout2d(p=dropout_rate)
#
#         # --- 新增：两次显式上采样增强细节 ---
#         # 128 -> 256
#         self.up_to_256 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             ConvBatchNormReLU(mid_c2, 64, 3, 1, 1, 1, leaky=leaky)
#         )
#
#         # 256 -> 512
#         self.up_to_512 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             ConvBatchNormReLU(64, 32, 3, 1, 1, 1, leaky=leaky)
#         )
#
#         # 4. 最后的投影输出
#         self.final_project = nn.Conv2d(32, 32, 1)  # 统一通道
#         self.mask_reduce = nn.Conv2d(32, 2, 1)
#         self.img_size = img_size
#
#     def forward(self, x_64, state0):
#         # ---- Step 1: 融合 64x64 ----
#         x = self.stem(x_64)
#         x = self.drop1(x)
#
#         # ---- Step 2: 上采样到 128 ----
#         x = self.up_to_128(x)
#
#         # ---- Step 3: 融合 128x128 (state0) ----
#         x = torch.cat([x, state0], dim=1)
#         x = self.fusion_state0(x)
#         x = self.drop2(x)  # 融合后的关键 Dropout
#
#         # ---- Step 4: 连续两次上采样 (128 -> 256 -> 512) ----
#         x = self.up_to_256(x)
#         # 这里可以再加一个轻微的 Dropout
#         x = F.dropout2d(x, p=0.1, training=self.training)
#
#         x = self.up_to_512(x)
#
#         # ---- Step 5: 插值到指定尺寸并投影 ----
#         # 现在的 x 已经是 512x512，缩放到 406 会比从 128 直接缩放精确得多
#         x = F.interpolate(x, (self.img_size, self.img_size), mode="bilinear", align_corners=False)
#
#         x = F.relu(self.final_project(x))
#         x = self.mask_reduce(x)
#
#         return x
#
#
# class MultiScaleMaskDecoder(nn.Module):
#     def __init__(self, input_1, backbone_channels, out_channels=2, leaky=True, img_size=406):
#         super().__init__()
#         # stem 输出通道
#         c1 = (input_1 + backbone_channels) // 2
#         c2 = c1 // 2  # 第一次上采样输出通道
#         c3 = c2 // 2  # 第二次上采样输出通道
#
#         # stem 多尺度卷积块
#         self.stem = MultiScaleConv(input_1 + backbone_channels, c1, leaky=leaky)
#
#         # 上采样模块，逐步降低通道
#         self.up_blocks = nn.ModuleList([
#             self._make_up_block(c1, c2, leaky),
#             self._make_up_block(c2, c3, leaky)
#         ])
#
#         self.out_proj = nn.Conv2d(c3, out_channels, 1)
#         self.img_size = img_size
#
#     def _make_up_block(self, in_ch, out_ch, leaky):
#         return nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             MultiScaleConv(in_ch, out_ch, leaky=leaky)
#         )
#
#     def forward(self, x, feat_backbone):
#         # stem 前拼接 backbone 特征
#         x = torch.cat([x, feat_backbone], dim=1)
#         x = self.stem(x)
#         # 上采样模块
#         for up_block in self.up_blocks:
#             x = up_block(x)
#         # 上采样到目标输出尺寸
#         x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
#         x = self.out_proj(x)
#         return x
#
#
#
# # 多尺度卷积块
# class MultiScaleConv(nn.Module):
#     def __init__(self, in_ch, out_ch, leaky=True):
#         super().__init__()
#         self.branch1 = ConvBatchNormReLU(in_ch, out_ch // 4, 1, 1, 0, 1, leaky=leaky)  # 1x1
#         self.branch2 = ConvBatchNormReLU(in_ch, out_ch // 4, 3, 1, 1, 1, leaky=leaky)  # 3x3
#         self.branch3 = ConvBatchNormReLU(in_ch, out_ch // 4, 5, 1, 2, 1, leaky=leaky)  # 5x5
#         self.branch4 = nn.Sequential(  # 3x3 dilated
#             nn.Conv2d(in_ch, out_ch // 4, 3, 1, 2, 2, bias=False),
#             nn.BatchNorm2d(out_ch // 4),
#             nn.LeakyReLU(0.2) if leaky else nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         out1 = self.branch1(x)
#         out2 = self.branch2(x)
#         out3 = self.branch3(x)
#         out4 = self.branch4(x)
#         return torch.cat([out1, out2, out3, out4], dim=1)
#
#
#
# # class PromptMaskDecoder(nn.Module):
# #     def __init__(self, feat_dims, txt_dim, num_heads=8, out_channels=2, refine_channels=64, attn_scale=None):
# #         """
# #         feat_dims: list of ints (假设所有元素相同, e.g. [C, C, C])
# #         txt_dim: dim of txt_final
# #         """
# #         super().__init__()
# #         assert len(feat_dims) >= 1
# #         self.feat_dims = feat_dims
# #         self.C = feat_dims[0]
# #         self.txt_dim = txt_dim
# #         self.out_channels = out_channels
# #
# #         # txt -> proj to match high feature dim (高层 feature dim)
# #         self.txt_proj = nn.Linear(txt_dim, self.C)
# #
# #         # single shared prompt->mask projection (保持语义一致)
# #         self.prompt_to_mask = nn.Linear(self.C, self.C)
# #
# #         # cross-attn: queries = txt_proj (B, L, C), kv = feat_flat (B, HW, C)
# #         self.cross_attn = nn.MultiheadAttention(embed_dim=self.C, num_heads=num_heads, batch_first=True)
# #
# #         # layernorms for stability
# #         self.ln_txt = nn.LayerNorm(self.C)
# #         self.ln_masktok = nn.LayerNorm(self.C)
# #
# #         # optional small learnable mask token (not required but helps)
# #         self.use_mask_token = True
# #         if self.use_mask_token:
# #             self.mask_token = nn.Parameter(torch.randn(1, 1, self.C) * 0.02)  # single token
# #         else:
# #             self.register_buffer('mask_token', torch.zeros(1,1,self.C))
# #
# #         # fuse and final
# #         self.fuse_conv = nn.Conv2d(self.C * len(feat_dims), self.C, kernel_size=1)
# #         self.conv_out = nn.Conv2d(self.C, out_channels, kernel_size=1)
# #
# #         # refine convs with residual
# #         self.refine_convs = nn.ModuleList([
# #             nn.Sequential(
# #                 nn.Conv2d(out_channels, refine_channels, kernel_size=3, padding=1),
# #                 nn.BatchNorm2d(refine_channels),
# #                 nn.ReLU(inplace=True),
# #                 nn.Conv2d(refine_channels, out_channels, kernel_size=3, padding=1)
# #             ) for _ in range(4)
# #         ])
# #
# #         # attention scale (sqrt dim) 可调整
# #         self.attn_scale = attn_scale if attn_scale is not None else (self.C ** 0.5)
# #
# #     def forward(self, feat_list, txt_final, target_size=(512,512)):
# #         """
# #         feat_list: list of feature maps, each (B, C, H_i, W_i), assume same C
# #         txt_final: (B, L, txt_dim)
# #         target_size: (H_out, W_out)
# #         """
# #         B = txt_final.shape[0]
# #         L = txt_final.shape[1]
# #
# #         # 1) project text
# #         txt_proj = self.txt_proj(txt_final)               # (B, L, C)
# #         txt_proj = self.ln_txt(txt_proj)
# #
# #         # 2) use cross-attn on highest level feature (last)
# #         high_feat = feat_list[-1]                         # (B, C, H, W)
# #         Bf, C, H, W = high_feat.shape
# #         assert Bf == B and C == self.C, "feat / txt dim mismatch"
# #
# #         feat_flat = high_feat.flatten(2).transpose(1,2)   # (B, HW, C)
# #
# #         # cross-attn: queries=txt_proj, kv=feat_flat
# #         attn_out, _ = self.cross_attn(txt_proj, feat_flat, feat_flat)  # (B, L, C)
# #         # average pooling across text tokens -> mask token representation
# #         mask_tokens = attn_out.mean(dim=1, keepdim=True)  # (B, 1, C)
# #
# #         # combine with learned mask_token if enabled
# #         if self.use_mask_token:
# #             # concatenate learned mask token with attn-derived token (or replace based on design)
# #             # here我们采用融合： learned + attn-derived
# #             learned = self.mask_token.expand(B, -1, -1)  # (B,1,C)
# #             mask_tokens = mask_tokens + learned
# #
# #         mask_tokens = self.ln_masktok(mask_tokens)  # (B,1,C)
# #
# #         # 3) prompt->mask projection (shared)
# #         mask_token_proj = self.prompt_to_mask(mask_tokens.squeeze(1))  # (B, C)
# #         mask_token_proj = torch.tanh(mask_token_proj)  # 稳定激活
# #         mask_token_proj = mask_token_proj.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
# #
# #         # 4) apply token to each feat (broadcast)
# #         mask_feats = []
# #         for feat in feat_list:
# #             # scale token to feat spatial size
# #             mask_token_resized = F.interpolate(mask_token_proj, size=feat.shape[2:], mode='nearest')  # (B,C,H_i,W_i)
# #             mask_feats.append(feat * mask_token_resized)
# #
# #         # 5) multi-scale fuse: upsample all to the highest-res among feat_list (choose first as reference)
# #         H_ref, W_ref = mask_feats[0].shape[2], mask_feats[0].shape[3]
# #         resized = [
# #             f if (f.shape[2], f.shape[3]) == (H_ref, W_ref)
# #             else F.interpolate(f, size=(H_ref,W_ref), mode='bilinear', align_corners=False)
# #             for f in mask_feats
# #         ]
# #         fused = torch.cat(resized, dim=1)  # (B, C * n, H_ref, W_ref)
# #         fused = self.fuse_conv(fused)      # (B, C, H_ref, W_ref)
# #
# #         # 6) initial mask logits
# #         mask_logits = self.conv_out(fused)  # (B, out_ch, H_ref, W_ref)
# #
# #         # 7) progressive upsample + refine (保持同你原实现)
# #         up_sizes = [
# #             (H_ref*2, W_ref*2),
# #             (H_ref*4, W_ref*4),
# #             target_size
# #         ]
# #         for i, size in enumerate(up_sizes):
# #             mask_logits = F.interpolate(mask_logits, size=size, mode='bilinear', align_corners=False)
# #             refine = self.refine_convs[i](mask_logits)
# #             mask_logits = mask_logits + refine  # residual refine
# #
# #         return mask_logits
#
#
# # class PromptMaskDecoder(nn.Module):
# #     def __init__(self, feat_dims, txt_dim, num_heads=8, out_channels=2):
# #         """
# #         feat_dims: list of feature_map channels for multi-scale features, e.g., [256, 512]
# #         txt_dim: dimension of prompt token
# #         """
# #         super().__init__()
# #         self.txt_dim = txt_dim
# #         self.out_channels = out_channels
# #         self.num_heads = num_heads
# #         self.feat_dims = feat_dims
# #
# #         # Linear projection: txt_final -> dynamic mask token per feature map
# #         self.prompt_to_mask = nn.ModuleList([nn.Linear(txt_dim, dim) for dim in feat_dims])
# #
# #         # Multihead cross-attention
# #         self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dims[-1], num_heads=num_heads, batch_first=True)
# #
# #         # Optional: fuse multi-scale features
# #         self.fuse_conv = nn.Conv2d(sum(feat_dims), feat_dims[-1], kernel_size=1)
# #
# #         # Final conv to get mask logits
# #         self.conv_out = nn.Conv2d(feat_dims[-1], out_channels, kernel_size=1)
# #
# #     def forward(self, feat_list, txt_final, target_size=None):
# #         """
# #         feat_list: list of multi-scale feature maps [B, C_i, H_i, W_i]
# #         txt_final: [B, L, txt_dim] prompt tokens
# #         target_size: (H, W) for output mask
# #         """
# #         B = txt_final.shape[0]
# #         L = txt_final.shape[1]
# #
# #         # 1. Cross-attention on highest-level feature
# #         high_feat = feat_list[-1]
# #         B, C, H, W = high_feat.shape
# #         feat_flat = high_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
# #
# #         # 如果 txt_dim != C，需要投影
# #         if txt_final.shape[2] != C:
# #             proj = nn.Linear(txt_final.shape[2], C).to(txt_final.device)
# #             txt_final_proj = proj(txt_final)
# #         else:
# #             txt_final_proj = txt_final
# #
# #         attn_out, _ = self.cross_attn(txt_final_proj, feat_flat, feat_flat)  # [B, L, C]
# #
# #         # 2. 动态 mask token: 平均 pooling prompt token
# #         mask_tokens = attn_out.mean(dim=1)  # [B, C]
# #
# #         # 3. 对每个 feature map 做动态卷积/线性组合
# #         mask_feats = []
# #         for i, feat in enumerate(feat_list):
# #             B, C_i, H_i, W_i = feat.shape
# #             # prompt -> dynamic weights
# #             mask_token_proj = self.prompt_to_mask[i](mask_tokens)  # [B, C_i]
# #             # 广播相乘
# #             mask_feat = feat * mask_token_proj.unsqueeze(-1).unsqueeze(-1)  # [B, C_i, H_i, W_i]
# #             mask_feats.append(mask_feat)
# #
# #         # 4. 多尺度融合
# #         # resize所有特征到最高分辨率
# #         H_max, W_max = mask_feats[0].shape[2], mask_feats[0].shape[3]
# #         mask_feats = [F.interpolate(f, size=(H_max, W_max), mode='bilinear', align_corners=False) if f.shape[2:] != (H_max, W_max) else f for f in mask_feats]
# #
# #         fused_feat = torch.cat(mask_feats, dim=1)  # [B, sum(C_i), H_max, W_max]
# #         fused_feat = self.fuse_conv(fused_feat)    # [B, C_high, H_max, W_max]
# #
# #         # 5. 最终 mask
# #         mask_logits = self.conv_out(fused_feat)    # [B, out_channels, H_max, W_max]
# #
# #         # 6. 插值到目标尺寸（微调用）
# #         if target_size is not None and target_size != (H_max, W_max):
# #             mask_logits = F.interpolate(mask_logits, size=target_size, mode='bilinear', align_corners=False)
# #
# #         return mask_logits  # [B, out_channels, H_target, W_target]
