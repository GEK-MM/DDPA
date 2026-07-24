# -*- coding: utf-8 -*-
import sys
import cv2
import re

sys.path.append('.')
import utils

sys.modules['utils'] = utils
cv2.setNumThreads(0)
from transformers import AutoTokenizer, DebertaV2Tokenizer, BertTokenizer
import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import random
from refer.refer import REFER
from torchvision.transforms import GaussianBlur, ColorJitter
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import spacy
import re
import torch.nn.functional as F
import torchvision.transforms.functional as TF

nlp = spacy.load("en_core_web_md")
# 自定义词性映射（除了介词，由列表单独处理）
POS_MAP = {
    "VERB": "V",  # 动词
    "AUX": "V",  # 助动词也当动词
    "NOUN": "N",
    "PROPN": "N",
    "PRON": "R",
    "DET": "D",
    "ADJ": "J",
    "ADV": "Q",
    "CCONJ": "C",
    "SCONJ": "S",
    "PART": "T",
    "PUNCT": "X",
    "NUM": "M",
    "SYM": "Y",
    "INTJ": "I",
}
DIRECTION_WORDS = ['top', 'bottom', 'left', 'right', 'upper', 'lower', 'north', 'south', 'east', 'west', 'between']


# Dataset configuration initialization
# tokenizer = AutoTokenizer.from_pretrained(model_root)
def add_random_boxes(img, min_num=20, max_num=60, size=32):
    h, w = size, size
    img = np.asarray(img).copy()
    img_size = img.shape[1]
    boxes = []
    num = random.randint(min_num, max_num)
    for k in range(num):
        y, x = random.randint(0, img_size - w), random.randint(0, img_size - h)
        img[y:y + h, x: x + w] = 0
        boxes.append((x, y, h, w))
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img


# nlp这个库用的中等版本，有时候识别词性可能会出问题
noun_words = {'marking', 'sidewalk', 'building'}
direction_words = {'left', 'right', 'top', 'bottom', 'front', 'back'}
VALID_START_TAGS = {'N', 'D', 'J', 'R', 'M'}

#******************* SSDM *************
def split_by_verb_prep_custom(text):
    text = ' '.join(text.split())
    sub_texts = [s.strip() for s in text.split(',')]
    extracted_np, final_subs, found = None, [], False
    for sub in sub_texts:
        if found:
            final_subs.append(sub)
            continue
        doc = nlp(sub)
        # 标签生成：注意这里把连字符单独映射为 L
        tags = ["N" if t.text.lower() in noun_words else
                "J" if t.text.lower() in direction_words else
                "L" if t.text == '-' else
                POS_MAP.get(t.pos_, "O") for t in doc]
        pos_seq = "".join(tags)

        # 判定条件：单句直接进，多句则首词必须符合白名单
        if len(sub_texts) == 1 or (pos_seq and pos_seq[0] in VALID_START_TAGS):
            # --- 正则修复：[CJQLX]* 是修饰前缀，N+(?:LN+)* 兼容单名词和复合名词 ---
            m = re.search(r'[CJQLX]*N+(?:LN+)*', pos_seq)
            if m:
                t_starts = [t.idx for t in doc] + [len(sub)]

                # 往前看一位，把冠词 "The/A" 也抓进 extracted_np
                full_start = m.start()
                extracted_np = sub[t_starts[full_start]:t_starts[m.end()]].strip()

                # --- 替换修复：只替换最后的名词簇，防止插入到 right-most 中间 ---
                all_n_matches = list(re.finditer(r'N+(?:LN+)*', pos_seq[m.start():m.end()]))
                if all_n_matches:
                    last_n = all_n_matches[-1]
                    ns = m.start() + last_n.start()
                    ne = m.start() + last_n.end()

                    prefix = sub[:t_starts[ns]].rstrip()
                    suffix = sub[t_starts[ne]:].lstrip()
                    final_subs.append(f"{prefix}  {suffix}".strip())
                    found = True
                    continue
        final_subs.append(sub)

    return [extracted_np, ", ".join(final_subs)] if extracted_np else [text, "  "]


class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)
        self.max_tokens = args.num_query

        self.ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(self.ref_ids)

        # 这里的 mask 逻辑如果是业务需要则保留，如果是增强则可以根据需求决定是否删除
        num_images_to_mask = int(len(self.ref_ids) * 0.2)
        self.images_to_mask = random.sample(self.ref_ids, num_images_to_mask)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)

        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer)
        self.eval_mode = eval_mode
        self.split_num = 2

        # --- 修改处：移除所有数据增强，仅保留 Tensor 转换 ---
        self.base_transform = A.Compose([
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def process_sentence(self, raw_text):
        raw_text = str(raw_text)
        raw_segments = split_by_verb_prep_custom(raw_text)
        raw_segments = raw_segments[:self.split_num]
        while len(raw_segments) < self.split_num:
            raw_segments.append("")

        tokens = self.tokenizer(
            raw_segments,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_tensors="pt",
            max_length=self.max_tokens
        )
        return tokens["input_ids"], tokens["attention_mask"]

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        # 加载图像
        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        # 加载 Mask
        ref = self.refer.loadRefs(this_ref_id)
        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        # 1. 执行基础转换（Resize/Normalize）
        if self.image_transforms is not None:
            img, target = self.image_transforms(img, annot)
        else:
            target = annot

        # 2. 转换为 Numpy 格式以配合 ToTensorV2
        img_np = img.permute(1, 2, 0).numpy()
        mask_np = np.array(target)

        # 3. 统一使用不含增强的 base_transform
        transformed = self.base_transform(image=img_np, mask=mask_np)
        img_tensor = transformed['image']
        target_tensor = transformed['mask'].long()

        ref_sentences = self.refer.Refs[this_ref_id]['sentences']

        # 处理文本和返回
        if self.split == "test" or self.split == "val":
            embeddings = []
            attn_masks = []
            if self.split == "val":
                raw = np.random.choice([s['raw'] for s in ref_sentences])
                e, a = self.process_sentence(raw)
                embeddings.append(e.unsqueeze(-1))
                attn_masks.append(a.unsqueeze(-1))
            else:  # test 模式返回该 ref 的所有句子
                for sent in ref_sentences:
                    raw = sent['raw']
                    e, a = self.process_sentence(raw)
                    embeddings.append(e.unsqueeze(-1))
                    attn_masks.append(a.unsqueeze(-1))

            tensor_embeddings = torch.cat(embeddings, dim=-1)
            attention_mask = torch.cat(attn_masks, dim=-1)
            return img_tensor, target_tensor, tensor_embeddings, attention_mask
        else:
            # 训练模式：随机选一个句子，不进行增强
            raw = np.random.choice([s['raw'] for s in ref_sentences])
            tensor_embeddings, attention_mask = self.process_sentence(raw)
            return img_tensor, target_tensor, tensor_embeddings, attention_mask


def _load_mask(mask_path):
    """加载分割掩码"""

    mask = Image.open(mask_path)
    # 转换为灰度
    mask = mask.convert('L')
    # 转换为 NumPy 数组并确保是整数类型
    return mask


# rrsis的数据集
def _load_image(image_path):
    """加载TIFF图像"""
    try:
        # 使用PIL打开tif文件
        img = Image.open(image_path)
        # 转换为RGB（如果是单通道则复制为3通道）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def _load_txt_file(txt_file):
    """读取txt文件，解析图像序号和文本"""
    data = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 分割序号和文本
            parts = line.split(' ', 1)
            if len(parts) == 2:
                image_id = parts[0].strip()
                text = parts[1].strip()
                data.append({
                    'image_id': image_id,
                    'text': text
                })
    return data


class SegmentationDataset(data.Dataset):
    """
    分割数据集，包含图像、分割掩码和文本描述
    已移除 Rotate, GaussianBlur, CoarseDropout 等数据增强
    """

    def __init__(self, args, split, txt_file, image_dir, mask_dir, image_transforms=None, target_transforms=None,
                 is_tif=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer)
        self.data = _load_txt_file(txt_file)
        self.split_num = 2
        self.is_tif = is_tif
        self.max_tokens = args.num_query
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split

        # 只保留基础的张量转换，去掉所有增强操作
        self.base_transform = A.Compose([
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def process_sentence(self, raw_text):
        raw_text = str(raw_text)
        raw_segments = split_by_verb_prep_custom(raw_text)
        raw_segments = raw_segments[:self.split_num]
        while len(raw_segments) < self.split_num:
            raw_segments.append("")

        tokens = self.tokenizer(
            raw_segments,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_tensors="pt",
            max_length=self.max_tokens
        )

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        return input_ids, attention_mask

    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = item['image_id']
        text = item['text']

        if self.is_tif:
            img_path = os.path.join(self.image_dir, f"{image_id}.tif")
            mask_path = os.path.join(self.mask_dir, f"{image_id}.tif")
        else:
            img_path = os.path.join(self.image_dir, f"{image_id}")
            mask_path = os.path.join(self.mask_dir, f"{image_id}")

        image = _load_image(img_path)
        mask = _load_mask(mask_path)

        # Mask 预处理
        mask_np = np.array(mask)
        mask_np[mask_np != 0] = 1
        mask = Image.fromarray(mask_np)

        # 执行基础的 image_transforms (通常是 Resize 和 Normalize)
        img, target = self.image_transforms(image, mask)

        # 转换为 Numpy 以适配 Albumentations 的 ToTensorV2
        img_np = img.permute(1, 2, 0).numpy()
        mask_np = np.array(target)

        # 统一使用基础转换 (仅转换为 Tensor)
        transformed = self.base_transform(image=img_np, mask=mask_np)
        img_tensor = transformed['image']
        target_tensor = transformed['mask'].long()

        if self.split == "test" or self.split == "val":
            # 测试/验证逻辑
            embeddings = []
            attn_masks = []
            e, a = self.process_sentence(text)

            embeddings.append(e.unsqueeze(-1))  # [1, max_len, 1]
            attn_masks.append(a.unsqueeze(-1))  # [1, max_len, 1]
            tensor_embeddings = torch.cat(embeddings, dim=-1)
            attention_mask = torch.cat(attn_masks, dim=-1)

            return img_tensor, target_tensor, tensor_embeddings, attention_mask
        else:
            # 训练逻辑 (现在与测试逻辑一致，不包含增强)
            tensor_embeddings, attention_mask = self.process_sentence(text)
            return img_tensor, target_tensor, tensor_embeddings, attention_mask

    def __len__(self):
        return len(self.data)
