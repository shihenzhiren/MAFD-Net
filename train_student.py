"""
the general training framework
"""

from __future__ import print_function

import os
import re
import argparse
import time
import sys

import numpy
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tensorboard_logger as tb_logger
import torch.nn.functional as F

from models import model_dict
from models.util import ConvReg, SelfA, SRRL, SimKD

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader,  get_dataloader_sample
#from dataset.imagenet_dali import get_dali_data_loader
from dataset.QAX2024 import get_QAX2024_dataloaders, get_QAX2024_dataloaders_sample  # 导入 QAX2024 数据集的加载函数
from dataset.unsw import get_unsw_dataloaders, get_unsw_dataloaders_sample  # 导入 UNSW 数据集的加载函数

from helper.loops import train_distill, validate_vanilla, validate_distill
from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate

from crd.criterion import CRDLoss
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, VIDLoss, SemCKDLoss, AFD

from helper.meters import AverageMeter

split_symbol = '~' if os.name == 'nt' else ':'
# 在原来的代码基础上进行修改，确保支持多个蒸馏方法及其权重参数

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    # 基础参数
    parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    # 优化参数
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'],
                        help='optimizer type: sgd (default), adam, adamw')

    # 数据集和模型
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'imagenet', 'nsl', 'QAX2024', 'unsw'], help='dataset')
    parser.add_argument('--model_s', type=str, default='resnet8x4')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # 蒸馏参数
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--distill', nargs='+', default=['kd'], 
                        choices=['kd', 'hint', 'attention', 'similarity', 'vid', 'crd', 'semckd', 'srrl', 'simkd', 'afd'],
                        help='specify multiple distillation methods')
    
    # AFD相关参数
    parser.add_argument('--qk_dim', type=int, default=128, help='feature dimension in AFD attention')
    parser.add_argument('--beta_afd', type=float, default=1.0, help='weight for AFD loss')
    parser.add_argument('--hint_layers', type=str, default='1,3,5', 
                        help='layers for student model in AFD')
    parser.add_argument('--guide_layers', type=str, default='1,3,5', 
                        help='layers for teacher model in AFD')
    parser.add_argument('--use_improved_afd', action='store_true', 
                        help='use improved AFD distillation')
    parser.add_argument('--use_channel_selection', action='store_true', 
                        help='use channel selection in improved AFD')
    parser.add_argument('--use_dynamic_selection', action='store_true', 
                        help='use dynamic layer selection in improved AFD')
    
    # 各种损失的权重参数
    parser.add_argument('-c', '--cls', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-d', '--div', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='weight balance for other losses')
    parser.add_argument('--beta_kd', type=float, default=1.0, help='weight for KD distillation loss')
    parser.add_argument('--beta_crd', type=float, default=1.0, help='weight for CRD distillation loss')
    parser.add_argument('--beta_simkd', type=float, default=1.0, help='weight for SimKD distillation loss')
    parser.add_argument('--beta_hint', type=float, default=1.0, help='weight for hint loss')

    # CRD和NCE相关参数
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'],
                        help='CRD matching mode')
    parser.add_argument('--nce_k', default=4096, type=int,
                        help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, 
                        help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, 
                        help='momentum for non-parametric updates')
    parser.add_argument('--feat_dim', default=128, type=int, 
                        help='feature dimension')

    # Hint相关参数（恢复）
    parser.add_argument('--hint_layer', default=2, type=int, 
                        help='hint layer for hint loss')

    # CRD相关参数调整
    parser.add_argument('--use_improved_sampling', action='store_true',
                        help='whether to use improved sampling strategy')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='softmax temperature when computing similarity')
    parser.add_argument('--weight_schedule', type=str, default='linear',
                        choices=['linear', 'cosine', 'constant'],
                        help='how to schedule sampling weight ratio')
    parser.add_argument('--weight_schedule_start', type=float, default=0.0,
                        help='initial sampling weight ratio')
    parser.add_argument('--weight_schedule_end', type=float, default=0.8,
                        help='final sampling weight ratio')

    # 其他必要参数（恢复）
    parser.add_argument('--soft_labels', action='store_true', help='use soft labels')
    parser.add_argument('--factor', type=int, default=2, help='factor for simkd')

    # 多进程训练
    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--deterministic', action='store_true', help='Make results reproducible')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation of teacher')
    
    # CRD specific parameters
    parser.add_argument('--crd_mode', type=str, default='exact', 
                        choices=['exact', 'relax'], help='CRD matching mode')
    parser.add_argument('--crd_T', type=float, default=0.07, 
                        help='temperature parameter for CRD')
    parser.add_argument('--crd_momentum', type=float, default=0.5,
                        help='momentum for non-parametric updates')

    # 内存优化参数
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='number of gradient accumulation steps')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='use mixed precision training')
    parser.add_argument('--max_split_size_mb', type=int, default=512,
                        help='maximum size of memory splits in MB')
    
    # 断点续训参数
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint for resuming training')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='starting epoch for resuming training (use with --resume)')

    opt = parser.parse_args()

    # 为MobileNet/ShuffleNet模型设置不同的学习率
    if opt.model_s in ['MobileNetV2', 'MobileNetV2_1_0', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_1_5']:
        opt.learning_rate = 0.01

    # 设置模型和tensorboard的路径
    opt.model_path = './save/students/models'
    opt.tb_path = './save/students/tensorboard'

    # 解析学习率衰减epoch
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # 获取教师模型名称
    opt.model_t = get_teacher_name(opt.path_t)

    # 生成模型名称
    model_name_template = split_symbol.join(['S', '{}_T', '{}_{}_{}_r', '{}_a', '{}_b', '{}_{}'])
    opt.model_name = model_name_template.format(opt.model_s, opt.model_t, opt.dataset, '+'.join(opt.distill),
                                                opt.cls, opt.div, opt.beta, opt.trial)

    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    # 创建tensorboard和模型保存目录
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    # CRD parameter post-processing
    if 'crd' in opt.distill:
        opt.n_data = {
            'cifar100': 50000,
            'imagenet': 1281167,
            'nsl': 113622,  # Set based on actual dataset size
            'QAX2024': 49690,  # 更新为实际数据集大小
            'unsw': 175341,   # UNSW 训练集大小
        }.get(opt.dataset)
        
        # Ensure necessary parameters exist
        assert opt.n_data is not None, f"Dataset {opt.dataset} not supported for CRD"
        
        # 改为打印weight_schedule信息
        print(f"Using weight schedule: {opt.weight_schedule} "
              f"({opt.weight_schedule_start}->{opt.weight_schedule_end})")

    return opt


def get_teacher_name(model_path):
    """解析教师模型名称"""
    directory = model_path.split('/')[-2]
    pattern = ''.join(['S', split_symbol, '(.+)', '_T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1]
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    return segments[0]


def load_teacher(model_path, n_cls, gpu=None, opt=None):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt.multiprocessing_distributed else 0)}
    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    print('==> done')
    return model


best_acc = 0
total_time = time.time()
def main():
    opt = parse_option()

    # 设置 CUDA 设备
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        # 分布式训练
        world_size = 1
        opt.world_size = ngpus_per_node * world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)


def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))
        # 设置CUDA内存分配器
        if opt.max_split_size_mb > 0:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{opt.max_split_size_mb}'

    if opt.multiprocessing_distributed:
        # 初始化分布式训练
        opt.rank = gpu
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
        opt.batch_size = int(opt.batch_size / ngpus_per_node)
        opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    if opt.deterministic:
        torch.manual_seed(12345)
        cudnn.deterministic = True
        cudnn.benchmark = False
        numpy.random.seed(12345)

    # 设置混合精度训练
    if opt.use_mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # 模型
    n_cls = {
        'cifar100': 100,
        'imagenet': 1000,
        'nsl': 5,
        'QAX2024': 81,  # QAX2024数据集有81个类别
        'unsw': 10,     # UNSW 数据集有10个类别
    }.get(opt.dataset, None)

    model_t = load_teacher(opt.path_t, n_cls, opt.gpu, opt)
    try:
        model_s = model_dict[opt.model_s](num_classes=n_cls)
    except KeyError:
        print("This model is not supported.")

    # 设置模型为评估模式
    model_t.eval()
    model_s.eval()

    # 使用较小的输入尺寸进行测试
    if opt.dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32)
    elif opt.dataset == 'imagenet':
        data = torch.randn(2, 3, 64, 64)
    elif opt.dataset == 'nsl':
        data = torch.randn(2, 3, 12, 12)
    elif opt.dataset == 'QAX2024':
        data = torch.randn(2, 3, 224, 224)  # 更新为224x224的输入尺寸
    elif opt.dataset == 'unsw':
        data = torch.randn(2, 3, 64, 64)  # UNSW Resize到64x64

    # 获取特征
    with torch.no_grad():
        feat_t, _ = model_t(data, is_feat=True)
        feat_s, _ = model_s(data, is_feat=True)

    # 清理内存
    del data
    torch.cuda.empty_cache()

    # 获取特征图形状
    s_shapes = [f.shape for f in feat_s]
    t_shapes = [f.shape for f in feat_t]
    
    # 预先提取所有可能需要的维度信息
    # 针对CRD
    opt.s_dim = feat_s[-1].shape[1]
    opt.t_dim = feat_t[-1].shape[1]
    
    # 针对VID
    s_n = [f.shape[1] for f in feat_s[1:-1]]
    t_n = [f.shape[1] for f in feat_t[1:-1]]
    
    # 初始化模块列表和可训练模块列表
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    # 初始化损失函数
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    criterion_kd_list = nn.ModuleList([])  # 多个蒸馏损失函数

    # 保存特征形状，以便在hint方法中使用
    hint_shapes = {}
    for i in range(len(feat_s)):
        if i < len(feat_t):
            hint_shapes[i] = (feat_s[i].shape, feat_t[i].shape)

    # 清理特征内存
    del feat_s, feat_t
    torch.cuda.empty_cache()

    # 根据蒸馏方法初始化对应的损失函数
    for distill_method in opt.distill:
        if distill_method == 'kd':
            criterion_kd = DistillKL(opt.kd_T)
            criterion_kd.weight = opt.beta_kd
        elif distill_method == 'hint':
            criterion_kd = HintLoss()
            # 使用预先保存的形状信息
            s_shape, t_shape = hint_shapes.get(opt.hint_layer, (None, None))
            if s_shape is None or t_shape is None:
                raise ValueError(f"Invalid hint_layer: {opt.hint_layer}, max layer is {len(hint_shapes)-1}")
            regress_s = ConvReg(s_shape, t_shape)
            # 记录 regress_s 在 module_list 中的位置，供 loops.py 使用
            opt.hint_regress_idx = len(module_list)
            module_list.append(regress_s)
            trainable_list.append(regress_s)
            criterion_kd.weight = opt.beta_hint
        elif distill_method == 'attention':
            criterion_kd = Attention()
            criterion_kd.weight = opt.beta
        elif distill_method == 'similarity':
            criterion_kd = Similarity()
            criterion_kd.weight = opt.beta
        elif distill_method == 'vid':
            # 使用预先计算的s_n和t_n
            criterion_kd = nn.ModuleList(
                [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
            )
            trainable_list.append(criterion_kd)  # VIDLoss 包含可训练参数
            criterion_kd.weight = opt.beta
        elif distill_method == 'crd':
            # 使用预先保存的维度信息
            # 注意：在训练中特征可能被展平，因此需要适配维度
            if len(s_shapes) > 0 and len(t_shapes) > 0:
                s_dim_total = s_shapes[-1][1] * s_shapes[-1][2] * s_shapes[-1][3] if len(s_shapes[-1]) > 3 else s_shapes[-1][1]
                t_dim_total = t_shapes[-1][1] * t_shapes[-1][2] * t_shapes[-1][3] if len(t_shapes[-1]) > 3 else t_shapes[-1][1]
                opt.s_dim = s_dim_total
                opt.t_dim = t_dim_total
                print(f"Using CRD with dimensions: s_dim={opt.s_dim}, t_dim={opt.t_dim}, n_data={opt.n_data}")
            else:
                print("警告：无法获取特征维度信息，使用默认值")
            
            criterion_kd = CRDLoss(opt)
            module_list.append(criterion_kd.embed_s)
            module_list.append(criterion_kd.embed_t)
            trainable_list.append(criterion_kd.embed_s)
            trainable_list.append(criterion_kd.embed_t)
            criterion_kd.weight = opt.beta_crd
        elif distill_method == 'afd':
            # 解析层数选择
            hint_layers = [int(i) for i in opt.hint_layers.split(',')]
            guide_layers = [int(i) for i in opt.guide_layers.split(',')]
            
            # 获取选定层的特征图形状 - 使用预先保存的s_shapes和t_shapes
            s_shapes_afd = []
            t_shapes_afd = []
            
            for i in hint_layers:
                if i < len(s_shapes):
                    s_shapes_afd.append(s_shapes[i])
                else:
                    raise ValueError(f"Invalid hint_layer: {i}, max available student layer is {len(s_shapes)-1}")
            
            for i in guide_layers:
                if i < len(t_shapes):
                    t_shapes_afd.append(t_shapes[i])
                else:
                    raise ValueError(f"Invalid guide_layer: {i}, max available teacher layer is {len(t_shapes)-1}")
            
            # 根据参数选择使用原始AFD还是改进的AFD
            if opt.use_improved_afd:
                from distiller_zoo import AFD_improved
                criterion_kd = AFD_improved(
                    s_shapes_afd, 
                    t_shapes_afd, 
                    qk_dim=opt.qk_dim,
                    use_channel_selection=opt.use_channel_selection,
                    use_dynamic_selection=opt.use_dynamic_selection
                )
                print("====> 使用改进版AFD蒸馏方法")
                if opt.use_channel_selection:
                    print("      启用通道选择")
                if opt.use_dynamic_selection:
                    print("      启用动态层选择")
            else:
                from distiller_zoo import AFD
                criterion_kd = AFD(s_shapes_afd, t_shapes_afd, opt.qk_dim)
                print("====> 使用原始AFD蒸馏方法")
            
            criterion_kd.hint_layers = hint_layers
            criterion_kd.guide_layers = guide_layers
            module_list.append(criterion_kd)
            trainable_list.append(criterion_kd)
            criterion_kd.weight = opt.beta_afd
        elif distill_method == 'srrl':
            s_n = feat_s[-1].shape[1]
            t_n = feat_t[-1].shape[1]
            model_fmsr = SRRL(s_n=s_n, t_n=t_n)
            criterion_kd = nn.MSELoss()
            module_list.append(model_fmsr)
            trainable_list.append(model_fmsr)
            criterion_kd.weight = opt.beta
        elif distill_method == 'simkd':
            s_n = feat_s[-2].shape[1]
            t_n = feat_t[-2].shape[1]
            model_simkd = SimKD(s_n=s_n, t_n=t_n, factor=opt.factor)
            criterion_kd = nn.MSELoss()
            module_list.append(model_simkd)
            trainable_list.append(model_simkd)
            criterion_kd.weight = opt.beta_simkd
        else:
            raise NotImplementedError(distill_method)
        criterion_kd_list.append(criterion_kd)

    # 合并所有损失函数
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # 分类损失
    criterion_list.append(criterion_div)  # KL 散度损失
    criterion_list.extend(criterion_kd_list)  # 蒸馏损失列表

    # 将教师模型添加到模块列表
    module_list.append(model_t)

    # 初始化优化器
    if opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(trainable_list.parameters(),
                                lr=opt.learning_rate,
                                weight_decay=opt.weight_decay)
        print(f"=> Using AdamW optimizer (lr={opt.learning_rate}, weight_decay={opt.weight_decay})")
    elif opt.optimizer.lower() == 'adam':
        optimizer = optim.Adam(trainable_list.parameters(),
                               lr=opt.learning_rate,
                               weight_decay=opt.weight_decay)
        print(f"=> Using Adam optimizer (lr={opt.learning_rate}, weight_decay={opt.weight_decay})")
    else:
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
        print(f"=> Using SGD optimizer (lr={opt.learning_rate}, momentum={opt.momentum}, weight_decay={opt.weight_decay})")

    # 使用 GPU
    if torch.cuda.is_available():
        if opt.multiprocessing_distributed and opt.gpu is not None:
            torch.cuda.set_device(opt.gpu)
            module_list.cuda(opt.gpu)
            distributed_modules = []
            for module in module_list:
                DDP = torch.nn.parallel.DistributedDataParallel
                distributed_modules.append(DDP(module, device_ids=[opt.gpu]))
            module_list = distributed_modules
            criterion_list.cuda(opt.gpu)
        else:
            criterion_list.cuda()
            module_list.cuda()
        if not opt.deterministic:
            cudnn.benchmark = True

    # 数据加载器
    if opt.dataset == 'cifar100':
        if 'crd' in opt.distill:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                              num_workers=opt.num_workers,
                                                                              k=opt.nce_k,
                                                                              mode=opt.mode)
        else:
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                               num_workers=opt.num_workers)
    elif opt.dataset == 'imagenet':
        if opt.dali is None:
            if 'crd' in opt.distill:
                train_loader, val_loader, n_data, _, train_sampler = get_dataloader_sample(dataset=opt.dataset, batch_size=opt.batch_size,
                                                                                          num_workers=opt.num_workers, is_sample=True,
                                                                                          k=opt.nce_k, multiprocessing_distributed=opt.multiprocessing_distributed)
            else:
                train_loader, val_loader, train_sampler = get_imagenet_dataloader(dataset=opt.dataset, batch_size=opt.batch_size,
                                                                                 num_workers=opt.num_workers,
                                                                                 multiprocessing_distributed=opt.multiprocessing_distributed)
        else:
            train_loader, val_loader = get_dali_data_loader(opt)
    elif opt.dataset == 'nsl':
        if 'crd' in opt.distill:  # 如果使用 CRD 蒸馏方法
            if opt.soft_labels:  # 如果使用软标签
                train_loader, val_loader, n_data = get_nsl_dataloaders_sample(
                    batch_size=opt.batch_size,
                    num_workers=opt.num_workers,
                    k=opt.nce_k,
                    sampling_mode=opt.sampling_mode,
                    teacher_soft_labels=None  # 动态生成软标签
                )
            else:  # 不使用软标签
                train_loader, val_loader, n_data = get_nsl_dataloaders_sample(
                    batch_size=opt.batch_size,
                    num_workers=opt.num_workers,
                    k=opt.nce_k
                )
        else:  # 其他蒸馏方法
            train_loader, val_loader = get_nsl_dataloaders(
                batch_size=opt.batch_size,
                num_workers=opt.num_workers
            )
    elif opt.dataset == 'QAX2024':
        if 'crd' in opt.distill:  # 如果使用 CRD 蒸馏方法
            if opt.soft_labels:  # 如果使用软标签
                train_loader, val_loader, n_data = get_QAX2024_dataloaders_sample(
                    batch_size=opt.batch_size,
                    num_workers=opt.num_workers,
                    k=opt.nce_k,
                    mode=opt.mode,
                    is_sample=True
                )
            else:  # 不使用软标签
                train_loader, val_loader, n_data = get_QAX2024_dataloaders_sample(
                    batch_size=opt.batch_size,
                    num_workers=opt.num_workers,
                    k=opt.nce_k
                )
            # 更新实际数据集大小用于CRD损失计算
            opt.n_data = n_data
        else:  # 其他蒸馏方法
            train_loader, val_loader = get_QAX2024_dataloaders(
                batch_size=opt.batch_size,
                num_workers=opt.num_workers
            )
    elif opt.dataset == 'unsw':
        if 'crd' in opt.distill:  # 如果使用 CRD 蒸馏方法
            train_loader, val_loader, n_data = get_unsw_dataloaders_sample(
                batch_size=opt.batch_size,
                num_workers=opt.num_workers,
                k=opt.nce_k,
                mode=opt.mode,
                is_sample=True
            )
            # 更新实际数据集大小用于CRD损失计算
            opt.n_data = n_data
        else:  # 其他蒸馏方法
            train_loader, val_loader = get_unsw_dataloaders(
                batch_size=opt.batch_size,
                num_workers=opt.num_workers
            )
    else:
        raise NotImplementedError(opt.dataset)


    # 初始化 TensorBoard 日志
    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # 验证教师模型精度
    if not opt.skip_validation:
        teacher_acc, _, _ = validate_vanilla(val_loader, model_t, criterion_cls, opt)
        if opt.dali is not None:
            val_loader.reset()
        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print('Teacher accuracy: ', teacher_acc)
    else:
        print('Skipping teacher validation.')

    # 加载 checkpoint（如果指定了 --resume）
    if opt.resume:
        if os.path.isfile(opt.resume):
            print(f"=> loading checkpoint '{opt.resume}'")
            checkpoint = torch.load(opt.resume, map_location=f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu')
            
            # 加载模型权重
            if 'model' in checkpoint:
                model_s.load_state_dict(checkpoint['model'])
                print(f"=> loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")
            
            # 加载 optimizer 状态
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded optimizer state")
            
            # 恢复 best_acc
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
                print(f"=> restored best_acc: {best_acc}")
            
            # 加载投影层（如果有 simkd）
            if 'simkd' in opt.distill and 'proj' in checkpoint:
                trainable_list[-1].load_state_dict(checkpoint['proj'])
                print("=> loaded projection layer weights")
            
            print(f"=> successfully loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            print(f"=> no checkpoint found at '{opt.resume}'")
    
    # 设置起始 epoch
    start_epoch = opt.start_epoch if opt.start_epoch > 0 else 1
    print(f"=> starting from epoch {start_epoch}")

    # 训练循环
    for epoch in range(start_epoch, opt.epochs + 1):
        torch.cuda.empty_cache()
        if opt.multiprocessing_distributed and opt.dali is None:
            train_sampler.set_epoch(epoch)

        # 调整学习率
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> Training...")

        # 添加时间记录
        time1 = time.time()
        
        # 使用统一的train_distill函数进行训练
        train_acc, train_acc_top5, train_loss = train_distill(
            epoch, train_loader, module_list, criterion_list, optimizer, opt)

        # 分布式训练汇总
        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_acc_top5, train_loss]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss = reduced.tolist()

        # 记录训练结果
        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            train_time = time.time() - time1
            print(' * Epoch {}, GPU {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}'.format(
                epoch, opt.gpu, train_acc, train_acc_top5, train_time))
            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_loss', train_loss, epoch)

        # 验证
        print('GPU %d Validating...' % (opt.gpu))
        test_acc, test_acc_top5, test_loss = validate_distill(val_loader, module_list, criterion_cls, opt)

        if opt.dali is not None:
            train_loader.reset()
            val_loader.reset()

        # 记录验证结果
        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))
            logger.log_value('test_acc', test_acc, epoch)
            logger.log_value('test_loss', test_loss, epoch)
            logger.log_value('test_acc_top5', test_acc_top5, epoch)

            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }   
                if 'simkd' in opt.distill:
                    state['proj'] = trainable_list[-1].state_dict() 
                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
                
                test_merics = {
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'test_acc_top5': test_acc_top5,
                    'epoch': epoch
                }
                
                metrics_file = os.path.join(opt.save_folder, "test_best_metrics.json")
                save_dict_to_json(test_merics, metrics_file)
                print(f'saving the best model to {save_file}!')
                torch.save(state, save_file)
            
            # 每个epoch保存checkpoint（用于断点续训）
            checkpoint_file = os.path.join(opt.save_folder, 'checkpoint_latest.pth')
            checkpoint_state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
            }
            if 'simkd' in opt.distill:
                checkpoint_state['proj'] = trainable_list[-1].state_dict()
            torch.save(checkpoint_state, checkpoint_file)
            print(f'checkpoint saved at epoch {epoch}')
            
    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        # This best accuracy is only for printing purpose.
        print('best accuracy:', best_acc)
        
        # save parameters
        save_state = {k: v for k, v in opt._get_kwargs()}
        # No. parameters(M)
        num_params = (sum(p.numel() for p in model_s.parameters())/1000000.0)
        save_state['Total params'] = num_params
        save_state['Total time'] =  (time.time() - total_time)/3600.0
        params_json_path = os.path.join(opt.save_folder, "parameters.json") 
        save_dict_to_json(save_state, params_json_path)

if __name__ == '__main__':
    main()