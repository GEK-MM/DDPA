

---

## 目录

- [数据集准备](#数据集准备)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [预训练权重](#预训练权重)
- [训练](#训练)
- [测试](#测试)

---

## 数据集准备

从以下仓库下载对应数据集：

| 数据集 | 下载地址 |
|--------|----------|
| RISBench | [HIT-SIRS/CroBIM](https://github.com/HIT-SIRS/CroBIM) |
| RRSIS-D | [Lsan2401/RMSIN](https://github.com/Lsan2401/RMSIN) |
| RefSegRS | [zhu-xlab/rrsis](https://github.com/zhu-xlab/rrsis) |

下载并解压后，按如下结构放置于 `./refer` 目录：

```
refer/
├── refer.py
├── RefSegRS/
│   ├── images/
│   ├── masks/
│   ├── output_phrase_test.txt
│   ├── output_phrase_train.txt
│   └── output_phrase_val.txt
├── RISBench_dataset/
│   ├── images/
│   ├── masks/
│   ├── output_phrase_test.txt
│   ├── output_phrase_train.txt
│   └── output_phrase_val.txt
└── rrsisd-data/
    ├── images/
    │   └── rrsisd/
    │       └── JPEGImages/
    └── rrsisd/
        ├── instances.json
        └── refs(unc).p
```

---

## 项目结构

```
.
├── bert-base-uncased/    # BERT 预训练权重
├── dataset/              # 数据加载
├── engine/               # 训练与评估引擎
├── example.ipynb
├── logs/
├── model/                # 模型定义
├── refer/                # 数据集目录
├── saved_models/         # 保存的 checkpoint
├── test_sample/
├── utils/
├── train.py
├── test.py
├── train.sh
├── test.sh
├── transforms.py
├── visual_results/
└── requirements.txt
```

---

## 环境配置

### 推荐版本

已在以下环境中验证通过：

- Python 3.10
- PyTorch 2.5.0
- CUDA 12.1
- torchvision 0.20.0
- torchaudio 2.5.0

### 安装依赖

```bash
pip install -r requirements.txt
```

---

## 预训练权重

### Swin Transformer

首次运行时，`timm` 会自动下载 Swin 预训练权重。也可手动加载：

```python
import timm

timm.create_model(
    "swin_base_patch4_window12_384",
    pretrained=True,
    img_size=img_size,
    features_only=True,
    out_indices=(0, 1, 2, 3),
)
```

### BERT

从 [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) 下载权重，放置于 `./bert-base-uncased/`。

### spaCy

安装 spaCy 及英文模型：

```bash
conda install spacy
python -m spacy download en
python -m spacy download en_core_web_md
```

验证安装：

```python
import spacy

nlp = spacy.load("en_core_web_md")
```

若 `en_core_web_md` 安装失败，可尝试从 [Release en_core_web_md-3.8.0](https://github.com/explosion/spacy-models/releases/tag/en_core_web_md-3.8.0) 手动下载。

## DDPA 预训练权重
https://drive.google.com/drive/folders/1uvHtp1qmwrKdg_gGJT2m4ULbqQuuhjZm?usp=drive_link

---

## 训练

编辑 `train.sh`，取消注释目标数据集对应的命令，并配置 GPU 数量等参数。示例：

```bash
export CUDA_VISIBLE_DEVICES=1
python train.py --dataset rrsisd --ngpu 2 --time 17 --savename Temp \
    --visulize 0 --batch_size 8 --nb_epoch 40 --lr 3e-5

# python train.py --dataset refsegrs --ngpu 2 --time 17 --savename Temp \
#     --visulize 0 --batch_size 4 --nb_epoch 55 --lr 5e-5

# python train.py --dataset risbench --ngpu 2 --time 17 --savename Temp \
#     --visulize 0 --batch_size 8 --nb_epoch 40 --lr 3e-5
```

启动训练：

```bash
bash train.sh 0,1
```

> **提示：** RefSegRS 最低学习率为 `1e-6`，其余数据集为 `1e-7`。

---

## 测试

编辑 `test.sh`，取消注释目标数据集对应的命令，并修改 `--pretrain` 指向你的模型权重路径。示例：
例如
```bash
export CUDA_VISIBLE_DEVICES=1
python test.py --dataset risbench \
    --pretrain ./saved_models/risbench_best.pth --visulize 0
```

启动测试：

```bash
bash test.sh 0
```
