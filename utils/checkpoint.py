import os
import shutil
import torch

from collections import defaultdict


def analyze_state_dict(state, name="模型"):
    """
    直接分析state_dict，不用先保存文件
    """
    print(f"\n{'=' * 60}")
    print(f"🔍 分析 {name}")
    print(f"{'=' * 60}")

    # 如果是完整checkpoint（包含state_dict等）
    if isinstance(state, dict) and 'state_dict' in state:
        print("📦 完整Checkpoint结构:")
        for key in state.keys():
            if key != 'state_dict':
                print(f"  - {key}")

        # 提取state_dict
        state_dict = state['state_dict']
    else:
        # 直接就是state_dict
        state_dict = state
        print("📦 直接StateDict")

    # 统计
    total_params = 0
    total_size = 0
    module_params = {}

    # 按模块分组
    for name, tensor in state_dict.items():
        if not hasattr(tensor, 'numel'):
            continue

        param_count = tensor.numel()
        param_size = param_count * tensor.element_size()  # 字节

        # 提取模块名（取第一个点之前的部分）
        module_name = name.split('.')[0] if '.' in name else 'root'

        if module_name not in module_params:
            module_params[module_name] = {'count': 0, 'size_mb': 0}

        module_params[module_name]['count'] += param_count
        module_params[module_name]['size_mb'] += param_size / 1024 ** 2

        total_params += param_count
        total_size += param_size

    # 打印汇总
    print(f"\n📈 总计:")
    print(f"  总参数量: {total_params / 1e6:.2f} M")
    print(f"  总大小: {total_size / 1024 ** 3:.2f} GB")

    # 按模块排序
    print("\n📋 按模块统计（从大到小）:")
    sorted_modules = sorted(module_params.items(), key=lambda x: x[1]['size_mb'], reverse=True)
    for module_name, stats in sorted_modules:
        print(f"  {module_name:30s}: {stats['count'] / 1e6:6.2f} M params, {stats['size_mb']:6.2f} MB")

    # 找出最大的tensor
    print("\n🔥 最大的30个tensor:")
    tensors = []
    for name, tensor in state_dict.items():
        if hasattr(tensor, 'numel'):
            size_mb = tensor.numel() * tensor.element_size() / 1024 ** 2
            tensors.append((name, tensor.shape, tensor.numel() / 1e6, size_mb))

    tensors.sort(key=lambda x: x[3], reverse=True)
    for name, shape, params_m, size_mb in tensors[:50]:
        name_short = name[:50] + "..." if len(name) > 50 else name
        print(f"  {name_short:55s} | shape={str(shape):20s} | {params_m:5.2f}M | {size_mb:5.2f}MB")


# def save_checkpoint(state, is_best, args, filename='default'):
#     if filename == "default":
#         filename = 'mcn_%s_batch%d' % (args.dataset, args.samples_per_gpu)
#     print("=> saving checkpoint '{}'".format(filename))
#     if not os.path.exists('./saved_models'):
#         os.makedirs('./saved_models')
#     checkpoint_name = './saved_models/%s_checkpoint.pth.tar' % (filename)
#     best_name = './saved_models/%s_model_best.pth.tar' % (filename)
#     torch.save(state, checkpoint_name)
#     if is_best:
#         print("=> saving best model '{}'".format(best_name))
#         shutil.copyfile(checkpoint_name, best_name)

def save_checkpoint(state, is_best, args, filename='default', epoch=0):
    # 路径准备
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    # 定义基础文件名：数据集 + 状态
    # 例如：RefSegRS_temp.pth 和 RefSegRS_best.pth
    temp_name = './saved_models/%s_temp.pth' % (args.dataset)
    best_name = './saved_models/%s_best.pth' % (args.dataset)
    # analyze_state_dict(state)
    # 1. 存临时文件 (Temp Checkpoint)
    # 这里依然存全家桶，确保万一服务器宕机，你能接上继续跑
    torch.save(state, temp_name)

    # 2. 存最好的模型 (Best Checkpoint)
    if is_best:
        # 只提取 state_dict，扔掉优化器，让它变成你期待的 1GB 左右
        pure_weights = state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state
        torch.save(pure_weights, best_name)
        print("=> Saved BEST weights to: %s" % best_name)


# def load_pretrain(model, args, logging, rank):
#     if os.path.isfile(args.pretrain):
#         checkpoint = torch.load(args.pretrain)
#         pretrained_dict = checkpoint['state_dict']
#         if hasattr(model, 'module'):
#             model_dict = model.module.state_dict()
#         else:
#             model_dict = model.state_dict()
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#         assert (len([k for k, v in pretrained_dict.items()]) != 0)
#         model_dict.update(pretrained_dict)
#         if hasattr(model, 'module'):
#             model.module.load_state_dict(model_dict)
#         else:
#             model.load_state_dict(model_dict)
#         print("=> loaded pretrain model at {}"
#               .format(args.pretrain))
#         if rank == 0:
#             logging.info("=> loaded pretrain model at {}"
#                          .format(args.pretrain))
#         del checkpoint  # dereference seems crucial
#         torch.cuda.empty_cache()
#     else:
#         print(("=> no pretrained file found at '{}'".format(args.pretrain)))
#         if rank == 0:
#             logging.info("=> no pretrained file found at '{}'".format(args.pretrain))
#     return model
def load_pretrain(model, args, logging, rank):
    if os.path.isfile(args.pretrain):
        # map_location 防止从 GPU 存的权重在没有 GPU 的地方报错
        checkpoint = torch.load(args.pretrain, map_location='cpu')

        # 核心改动：自动识别格式
        # 如果字典里有 'state_dict' 键，说明是旧的全家桶；否则视为纯权重
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
            print("=> Detected full checkpoint format.")
        else:
            pretrained_dict = checkpoint
            print("=> Detected pure weights format.")

        # 处理分布式训练 (DDP) 的前缀问题
        if hasattr(model, 'module'):
            model_dict = model.module.state_dict()
        else:
            model_dict = model.state_dict()

        # 过滤掉不匹配的 key (比如你改了架构后的旧权重)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        if len(pretrained_dict) == 0:
            msg = "=> WARNING: No matching weights found!"
            print(msg)
            if rank == 0: logging.info(msg)
            return model

        model_dict.update(pretrained_dict)

        if hasattr(model, 'module'):
            model.module.load_state_dict(model_dict)
        else:
            model.load_state_dict(model_dict)

        msg = "=> Loaded pretrain model at {}".format(args.pretrain)
        print(msg)
        if rank == 0: logging.info(msg)

        del checkpoint
        torch.cuda.empty_cache()
    else:
        msg = "=> no pretrained file found at '{}'".format(args.pretrain)
        print(msg)
        if rank == 0: logging.info(msg)

    return model


def load_pretrain_ddp(model, args):
    if os.path.isfile(args.pretrain):
        checkpoint = torch.load(args.pretrain)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert (len([k for k, v in pretrained_dict.items()]) != 0)
        model_dict.update(pretrained_dict)
        if hasattr(model, 'module'):
            state_dict = model.module.state_dict()
            model.module.load_state_dict(model_dict)
        else:
            state_dict = model.state_dict()
            model.load_state_dict(model_dict)
        print("load ")
        print("=> loaded pretrain model at {}"
              .format(args.pretrain))
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        print(("=> no pretrained file found at '{}'".format(args.pretrain)))
    return model


def load_resume(model, optimizer, args, logging, rank):
    if os.path.isfile(args.resume):
        print(("=> loading checkpoint '{}'".format(args.resume)))
        if rank == 0:
            logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        print("epoch: ", args.start_epoch)
        args.best_iou = checkpoint['best_iou']
        print("best iou: ", args.best_iou)
        state_dict = checkpoint['state_dict']

        if hasattr(model, 'module'):
            model_dict = model.module.state_dict()
        else:
            model_dict = model.state_dict()
        new_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(new_state_dict)

        if hasattr(model, 'module'):
            model.module.load_state_dict(model_dict)
        else:
            model.load_state_dict(model_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
        print("load successfully!")
    else:
        print(("=> no checkpoint found at '{}'".format(args.resume)))
        if rank == 0:
            logging.info(("=> no checkpoint found at '{}'".format(args.resume)))
    return model
