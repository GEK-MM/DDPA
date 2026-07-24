import sys
import timm
from utils.utils import *
from .modules import *
from .position_encoding import *
from .decoder import *

sys.path.append('../')
from transformers import BertModel, AutoConfig
from transformers import AutoModel
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAdaptiveHead(nn.Module):
    def __init__(self, in_channels=[128, 256, 512, 1024], embedding_dim=512, target_size=(512, 512), dropout=0.1):
        super().__init__()
        self.target_size = target_size

        # 1. 投影层：加入 Dropout2d，让各尺度特征更“硬气”一点，不互相依赖
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embedding_dim, 1),
                nn.GroupNorm(32, embedding_dim),  # 配合 Dropout 使用 GN 更稳
                nn.Dropout2d(p=dropout)
            ) for c in in_channels
        ])

        # 2. 语义锚点：给深层语义加点扰动
        self.semantic_anchor = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
            nn.GroupNorm(32, embedding_dim),
            nn.GELU(),
            nn.Dropout2d(p=dropout)
        )

        self.fusion_blocks = nn.ModuleList([
            GatedFusionBlock(embedding_dim) for _ in range(3)
        ])

        # 3. 分类器：在最后的特征提纯阶段加重一点 Dropout
        self.classifier = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 2, 3, padding=1),
            nn.GroupNorm(16, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout2d(p=0.15),  # 决策关口，建议 p=0.2
            nn.Conv2d(embedding_dim // 2, 2, 1)
        )

    def forward(self, inputs):
        # inputs: [f0, f1, f2, f3]
        f1, f2, f3, f4 = [self.projs[i](inputs[i]) for i in range(4)]

        curr_feat = self.semantic_anchor(f4)
        # 级联融合
        curr_feat = self.fusion_blocks[0](f3, curr_feat)
        curr_feat = self.fusion_blocks[1](f2, curr_feat)
        curr_feat = self.fusion_blocks[2](f1, curr_feat)

        # 得到预测
        out = self.classifier(curr_feat)

        # 全局上采样
        out = F.interpolate(out, size=self.target_size, mode='bilinear', align_corners=False)

        return out


class GatedFusionBlock(nn.Module):
    """
    核心改进：不再盲目 cat，而是学习一个权衡权重
    """

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, 2, 1),
            nn.Softmax(dim=1)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, high_res, low_res_semantic):
        # 上采样深层语义
        low_res_up = F.interpolate(low_res_semantic, size=high_res.shape[-2:], mode='bilinear')

        # 计算门控权重：动态决定这一层特征的“话语权”
        # weight[:, 0] 是给高分辨率特征的，weight[:, 1] 是给深层语义的
        combined = torch.cat([high_res, low_res_up], dim=1)
        weight = self.gate(combined)

        # 动态加权融合
        fused = high_res * weight[:, 0:1] + low_res_up * weight[:, 1:2]

        return self.refine(fused)


class Model(nn.Module):
    def __init__(self, tunelang=False, num_query=20, length=17,
                 img_size=406, text_model_path="", drop=0.1, drop_act=0.0):
        super(Model, self).__init__()
        self.tunelang = tunelang
        self.length = length
        self.visu_dim = 768
        self.textdim = 768
        self.img_size = img_size
        self.h, self.w = img_size // 14, img_size // 14
        ## Init Encoders
        self.main_branch = Main_Branch(dropout=drop, dropout_act=drop_act)
        self.num_fusion = 3
        self.norm = nn.LayerNorm(512)
        self.bert = AutoModel.from_pretrained(text_model_path)
        for name, param in self.bert.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split(".")[2])  # 获取层编号（如layer.11）
                if layer_num < 8:  # 假设BERT有12层，冻结前8层
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        # 先冻结所有参数
        # self.visual = timm.create_model('swin_base_patch4_window7_224', pretrained=True,
        #                                 img_size=img_size, features_only=True, out_indices=(0, 1, 2, 3), )
        self.visual = timm.create_model('swin_base_patch4_window12_384', pretrained=True,
                                        img_size=img_size, features_only=True, out_indices=(0, 1, 2, 3), )
        self.decoder = ScaleAdaptiveHead()
        self.pos_embedding = PositionEmbeddingSine(self.visu_dim, h=32, w=32)
        self.split_num = 2
        self.num_tokens = num_query

    def forward(self, image, word_id, word_mask):
        ## word_mask:[B,14,max_word] ,word_id :[b,sn,max_word]
        B = image.size(0)
        input_ids = word_id.reshape((B * self.split_num, -1))
        word_mask = word_mask.reshape((B * self.split_num, -1))
        feat = self.visual(image)
        processed_feats = []
        for f in feat:
            f_correct = f.permute(0, 3, 1, 2).contiguous()
            processed_feats.append(f_correct)
        # text part
        text_feat = self.bert(input_ids, attention_mask=word_mask)
        # txt_states = text_feat.hidden_states
        token_embeddings = text_feat.last_hidden_state  # shape: [B*self.split_num, 20, hidden_size]
        token_embeddings = token_embeddings.reshape(B, self.split_num, self.num_tokens, self.textdim)
        if self.training:
            F_tf, txt_final, _ = self.main_branch(processed_feats, token_embeddings)
            out = self.decoder(F_tf)  # 输出 [B, 768, 32, 32]
            return out
        else:
            F_tf, txt_final, g = self.main_branch(processed_feats, token_embeddings)
            out = self.decoder(F_tf)  # 输出 [B, 768, 32, 32]
            return out, g
