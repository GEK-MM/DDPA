import torch
# from model.model import Model
from model.model import Model

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume
import os


def Prepare_Model(args, rank, logger):
    model = Model(tunelang=args.tunelang, num_query=args.num_query,
                  img_size=args.img_size, text_model_path=args.bert_tokenizer, drop=args.drop_fusion,drop_act=args.drop_act).cuda(rank)

    # 保持你原来的加载顺序
    if args.pretrain and os.path.isfile(args.pretrain):
        model = load_pretrain(model, args, logger, rank)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model_without_ddp = model.module

    # 简单清晰的参数分组
    visu_param = [p for n, p in model_without_ddp.named_parameters() if n.startswith('visual') and p.requires_grad]
    text_param = [p for n, p in model_without_ddp.named_parameters() if n.startswith('bert') and p.requires_grad]
    rest_param = [p for n, p in model_without_ddp.named_parameters() if p.requires_grad
                  and not n.startswith('visual') and not n.startswith('bert')]


    optimizer = torch.optim.AdamW([
        {'params': rest_param, 'lr': args.lr},  # 新模块 3e-5
        {'params': visu_param, 'lr': args.lr},  # Swin 稍微降一点，给 1.5e-5
        {'params': text_param, 'lr': args.lr},  # Bert 降到 3e-6 (Bert非常容易碎，给小点)
    ], weight_decay=0.01, betas=(0.9, 0.98))  # weight_decay 降回 0.01，防止过度压缩

    return model, optimizer
