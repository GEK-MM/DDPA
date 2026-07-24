import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import datetime
import itertools
import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data.distributed
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Normalize
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler
from engine.engine import *
from dataset.data_loader import *
from utils.losses import *
from utils.utils import *
import math
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume
from transformers import DebertaV2Tokenizer, AutoModel
from utils.logger import setup_logger
from model.generate_model import *
import torch
import numpy as np
from dataset.data_loader import ReferDataset
import transforms as T
from thop import profile

mpl.use('Agg')

def get_args():
    parser = argparse.ArgumentParser(description='Dataloader test')
    parser.add_argument('--gpu', default='2', help='gpu id')
    parser.add_argument('--ngpu', default=2, type=int, help='gpu num')
    parser.add_argument('--workers', default=4, type=int, help='num workers for data loading')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    parser.add_argument('--nb_epoch', default=40, type=int, help='training epoch')
    parser.add_argument('--lr', default=0.00003, type=float, help='batch size learning rate')
    parser.add_argument('--power', default=0.1, type=float, help='lr poly power')
    parser.add_argument('--steps', default=[18, 28], type=list, help='in which step lr decay by power')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--img_size', default=512, type=int, help='image size')
    parser.add_argument('--dataset', default='rrsisd', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--drop_fusion', default=0.1, help='dropout for fusion')
    parser.add_argument('--drop_act', default=0, help='dropout for activate')
    parser.add_argument('--min_lr', default=1e-7, help='min learning rate')

    parser.add_argument('--num_query', default=20, type=int, help='the number of query')
    parser.add_argument('--w_seg', default=0.1, type=float, help='weight of the seg loss')
    parser.add_argument('--w_coord', default=5, type=float, help='weight of the reg loss')
    parser.add_argument('--tunelang', dest='tunelang', default=True, action='store_true',
                        help='if finetune language model')
    parser.add_argument('--anchor_imsize', default=416, type=int,
                        help='scale used to calculate anchors defined in model cfg file')
    parser.add_argument('--time', default=17, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='path to ReferIt splits data folder')

    parser.add_argument('--fusion_dim', default=768, type=int,
                        help='fusion module embedding dimensions')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, while have no loss saved as '
                             'resume')
    parser.add_argument('--print_freq', '-p', default=90, type=int,
                        metavar='N', help='print frequency (default: 1e3)')
    parser.add_argument('--savename', default="default", type=str, help='Name head for saved model')

    parser.add_argument('--seg_thresh', default=0.35, type=float, help='seg score above this value means foreground')
    parser.add_argument('--seg_out_stride', default=2, type=int, help='the seg out stride')
    parser.add_argument('--best_iou', default=-float('Inf'), type=int, help='the best accu')

    parser.add_argument('--model_name', default="DPPA", type=str, help='name of model')
    parser.add_argument('--visulize', default=0, type=int, help='visulize of picture')

    parser.add_argument('--data_root', default='./refer', help='Root directory for all datasets')
    parser.add_argument('--refer_data_root', default='./refer/rrsisd-data/', help='REFER dataset root directory')
    parser.add_argument('--split', default='train', help='only used when testing')
    parser.add_argument('--splitBy', default='unc',
                        help='change to umd or google when the datasset is G-Ref (RefCOCOg)')
    parser.add_argument('--bert_tokenizer', default='./bert-base-uncased/', help='BERT tokenizer')

    global args, anchors_full, writer, logger
    args = parser.parse_args()
    args.date = datetime.datetime.now().strftime('%Y%m%d')
    if args.savename == 'default':
        args.savename = '%s_model_v1_%s_batch%d_%s' % (args.model_name, args.dataset, args.batch_size, args.date)
    os.makedirs(args.log_dir, exist_ok=True)
    print('*********************************************************')
    # print(sys.argv[0])
    # print(args)
    print('*********************************************************')
    return args


def main(args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print("Running DDP with {} GPUs".format(n_gpus))
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, args,))
    else:
        print("Please use GPU for training")


def get_transform(args):
    transforms = [
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return T.Compose(transforms)


def get_dataset(image_set, transform, args):
    # 统一获取基础路径
    data_root = args.data_root
    if args.dataset == 'rrsisd':
        args.refer_data_root = os.path.join(data_root, 'rrsisd-data')
        ds = ReferDataset(args,
                          split=image_set,
                          image_transforms=transform,
                          target_transforms=None)
        args.min_lr = 1e-7
        args.drop_act = 0.12

    else:
        # refsegrs 和 risbench 走统一的路径拼接逻辑
        dataset_folder = 'RefSegRS' if args.dataset == 'refsegrs' else 'RISBench_dataset'
        curr_root = os.path.join(data_root, dataset_folder)
        # 自动生成相对路径
        txt_file = os.path.join(curr_root, f'output_phrase_{image_set}.txt')
        image_dir = os.path.join(curr_root, 'images')
        mask_dir = os.path.join(curr_root, 'masks')

        # 是否为 tif 格式的判断（仅 risbench 为 False，或者根据实际需求改）
        is_tif = (args.dataset == 'refsegrs')

        ds = SegmentationDataset(args,
                                 split=image_set,
                                 txt_file=txt_file,
                                 image_dir=image_dir,
                                 mask_dir=mask_dir,
                                 image_transforms=transform,
                                 target_transforms=None,
                                 is_tif=is_tif)

        # 设置超参数
        if args.dataset == 'refsegrs':
            args.min_lr = args.lr * 0.05
            args.drop_act = 0.0
        else:  # risbench
            args.min_lr = 1e-7
            args.drop_act = 0.12

    num_classes = 2
    return ds, num_classes


def run(rank, n_gpus, args):
    print("rank:", rank)

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.cuda.set_device(rank)
    ## fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    ## save logs
    logger = setup_logger(output=os.path.join(args.log_dir, args.savename), distributed_rank=rank, color=False,
                          name="model-v1")

    train_dataset, num_classes = get_dataset("train",
                                             get_transform(args=args),
                                             args=args)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=n_gpus, rank=rank,
                                                                    shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=args.workers, sampler=train_sampler,
                              )

    if rank == 0:
        val_dataset_test, _ = get_dataset("val",
                                          get_transform(args=args),
                                          args=args)
        print(f"Using full validation set (Total: {len(val_dataset_test)})")

        val_loader = DataLoader(val_dataset_test, batch_size=1, shuffle=False,
                                pin_memory=True, drop_last=False, num_workers=args.workers)
    model, optimizer = Prepare_Model(args, rank, logger)
    # Initialization
    scaler = GradScaler()
    best_miou_seg = -float('Inf')
    if args.resume:
        model = load_resume(model, optimizer, args, logger, rank)
        model.to(rank)
        best_miou_seg = args.best_iou

    # 1. 基础步数设定
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.nb_epoch

    # 2. 预热设定 (建议设为 3 个 Epoch，或者总步数的 5%-10%)
    warmup_epochs = 3
    warmup_steps = warmup_epochs * steps_per_epoch

    # 3. 学习率保底
    target_lr = args.lr
    min_lr = args.min_lr
    min_ratio = min_lr / target_lr

    def lr_lambda(current_step):
        # --- 第一阶段：Linear Warmup ---
        if current_step < warmup_steps:
            # 从一个极小值（比如 0.1 倍 target_lr）线性增长到 1.0
            return 0.1 + 0.9 * float(current_step) / float(max(1, warmup_steps))

        # --- 第二阶段：Cosine Annealing ---
        # 计算余弦退火阶段的进度 (从 0.0 到 1.0)
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))

        # 余弦曲线计算
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # 确保最终不低于设定的最小比率 (防止学习率归零)
        return max(min_ratio, cosine_decay)

    # 使用 LambdaLR 实现
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(str(sys.argv))
    logger.info(str(args))
    for epoch in range(args.nb_epoch):
        # 1. 训练阶段
        # 每一个 epoch 开始前必须设置 sampler
        train_sampler.set_epoch(epoch)
        model.train()
        # 正常训练，DDP 内部会自动处理梯度同步
        loss = train_epoch(rank, args, train_loader, model, optimizer, epoch, scaler, logger, lr_scheduler)
        is_val_epoch = (epoch == 0 or epoch >= 20 or epoch % 5 == 0)

        if is_val_epoch:
            if rank == 0:
                # 只有 Rank 0 真正跑测试逻辑
                miou_seg_avg, _, _ = validate_epoch(args, val_loader, model.module, logger, 'Val', epoch)
                is_best = miou_seg_avg > best_miou_seg
                best_miou_seg = max(miou_seg_avg, best_miou_seg)
                # --- 恢复模型保存逻辑 ---
                if epoch >= 30 or is_best:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),  # DDP模式下必须保存.module
                        'best_iou': best_miou_seg,
                        'optimizer': optimizer.state_dict(),
                    }, is_best, args, filename=args.savename, epoch=epoch)
            dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        logger.info(f'\nFinal Best Accu: {best_miou_seg:.4f}\n')


if __name__ == "__main__":
    args = get_args()
    main(args)
