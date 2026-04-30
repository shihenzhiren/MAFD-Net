from __future__ import print_function, division
from cProfile import label

import sys
import time
import torch
from .util import AverageMeter, accuracy, reduce_tensor
from crd.criterion import CRDLoss
import torch.nn as nn
from torch.cuda.amp import autocast

def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        if opt.dali is None:
            if len(batch_data) == 3:
                images, labels, _ = batch_data  # QAX2024 / UNSW 返回 (img, label, index)
            else:
                images, labels = batch_data
        else:
            images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()
        
        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # ===================forward=====================
        if opt.fp16 and hasattr(opt, 'scaler') and opt.scaler is not None:
            with autocast():
                output = model(images)
                loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)

            # ===================backward=====================
            optimizer.zero_grad()
            opt.scaler.scale(loss).backward()
            opt.scaler.step(optimizer)
            opt.scaler.update()
        else:
            output = model(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                   epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
            
    return top1.avg, top5.avg, losses.avg

def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd_list = criterion_list[2:]

    model_s = module_list[0]
    model_t = module_list[-1]

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 用于跟踪梯度累积
    if hasattr(opt, 'gradient_accumulation_steps') and opt.gradient_accumulation_steps > 1:
        optimizer.zero_grad()
        steps_since_update = 0

    for idx, data in enumerate(train_loader):
        if opt.distill == ['crd'] or 'crd' in opt.distill:
            if len(data) == 4:
                input, target, index, contrast_idx = data
            else:
                input, target, index = data
                contrast_idx = None
        else:
            input, target, index = data
            contrast_idx = None

        # 将数据移动到GPU
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if contrast_idx is not None:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        if hasattr(opt, 'use_mixed_precision') and opt.use_mixed_precision:
            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                feat_s, logit_s = model_s(input, is_feat=True)
                with torch.no_grad():
                    feat_t, logit_t = model_t(input, is_feat=True)
        else:
            feat_s, logit_s = model_s(input, is_feat=True)
            with torch.no_grad():
                feat_t, logit_t = model_t(input, is_feat=True)

        # 分类损失
        loss_cls = criterion_cls(logit_s, target)
        # KD损失
        loss_div = criterion_div(logit_s, logit_t)

        # 其他蒸馏损失
        loss_kd_list = []
        for i, criterion_kd in enumerate(criterion_kd_list):
            distill_method = opt.distill[i] if i < len(opt.distill) else opt.distill[-1]
            if distill_method == 'kd':
                loss_kd = criterion_kd(logit_s, logit_t)
            elif distill_method == 'hint':
                f_s = feat_s[opt.hint_layer]
                f_t = feat_t[opt.hint_layer]
                # 使用记录的 regress_s 在 module_list 中的精确索引
                regress_idx = getattr(opt, 'hint_regress_idx', i + 1)
                f_s = module_list[regress_idx](f_s)
                loss_kd = criterion_kd(f_s, f_t)
            elif distill_method == 'attention':
                attention_s = [f.pow(2).mean(1) for f in feat_s[1:-1]]
                attention_t = [f.pow(2).mean(1) for f in feat_t[1:-1]]
                loss_kd = criterion_kd(attention_s, attention_t)
            elif distill_method == 'similarity':
                g_s = [feat_s[i].view(feat_s[i].shape[0], -1) for i in range(len(feat_s))]
                g_t = [feat_t[i].view(feat_t[i].shape[0], -1) for i in range(len(feat_t))]
                loss_kd = criterion_kd(g_s, g_t)
            elif distill_method == 'vid':
                # 由于vid损失是多个损失的集合，需要特殊处理
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = []
                for l, crit in enumerate(criterion_kd):
                    if l >= len(g_s):
                        break
                    loss_group.append(crit(g_s[l], g_t[l]) * opt.beta)
                loss_kd = sum(loss_group)
            elif distill_method == 'crd':
                # 处理CRD特征维度
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                
                # 如果特征是多维的，将其展平为2D
                if f_s.dim() > 2:
                    f_s = f_s.view(f_s.size(0), -1)
                if f_t.dim() > 2:
                    f_t = f_t.view(f_t.size(0), -1)
                
                # 检查与预期的维度是否匹配，如果不匹配，打印警告
                if f_s.size(1) != opt.s_dim or f_t.size(1) != opt.t_dim:
                    print(f"警告: 特征维度不匹配! 预期: s_dim={opt.s_dim}, t_dim={opt.t_dim}; 实际: s_dim={f_s.size(1)}, t_dim={f_t.size(1)}")
                    # 动态更新维度
                    if hasattr(criterion_kd, 'embed_s') and hasattr(criterion_kd.embed_s, 'linear'):
                        old_weight = criterion_kd.embed_s.linear.weight
                        old_bias = criterion_kd.embed_s.linear.bias
                        in_features = f_s.size(1)
                        out_features = old_weight.size(0)
                        criterion_kd.embed_s.linear = nn.Linear(in_features, out_features).cuda()
                        
                        if hasattr(criterion_kd, 'embed_t') and hasattr(criterion_kd.embed_t, 'linear'):
                            old_weight_t = criterion_kd.embed_t.linear.weight
                            old_bias_t = criterion_kd.embed_t.linear.bias
                            in_features_t = f_t.size(1)
                            out_features_t = old_weight_t.size(0)
                            criterion_kd.embed_t.linear = nn.Linear(in_features_t, out_features_t).cuda()
                            
                            print(f"已重新初始化CRD嵌入层: s ({in_features} -> {out_features}), t ({in_features_t} -> {out_features_t})")
                            opt.s_dim = in_features
                            opt.t_dim = in_features_t
                
                loss_kd = criterion_kd(f_s, f_t, index, contrast_idx, teacher_logits=logit_t)
            elif distill_method == 'afd':
                # 获取特定层的特征
                hint_layers = criterion_kd.hint_layers
                guide_layers = criterion_kd.guide_layers
                
                f_s = [feat_s[i] for i in hint_layers if i < len(feat_s)]
                f_t = [feat_t[i] for i in guide_layers if i < len(feat_t)]
                
                loss_kd = criterion_kd(f_s, f_t)
            elif distill_method == 'srrl':
                loss_kd = 0
                # SRRL使用最后一层特征
                if len(feat_s) > 0 and len(feat_t) > 0:
                    last_s = feat_s[-1]
                    last_t = feat_t[-1]
                    # 确保特征维度匹配
                    if last_s.dim() > 2:
                        last_s = last_s.view(last_s.size(0), -1)
                    if last_t.dim() > 2:
                        last_t = last_t.view(last_t.size(0), -1)
                    try:
                        srrl_module = module_list[1]  # 模型在module_list中的位置
                        last_s_transformed = srrl_module(last_s)
                        loss_kd = criterion_kd(last_s_transformed, last_t)
                    except Exception as e:
                        print(f"SRRL错误: {e}")
            elif distill_method == 'simkd':
                loss_kd = 0
                # SimKD使用倒数第二层特征
                if len(feat_s) > 1 and len(feat_t) > 1:
                    second_last_s = feat_s[-2]
                    second_last_t = feat_t[-2]
                    # 确保特征维度匹配
                    if second_last_s.dim() > 2:
                        second_last_s = second_last_s.view(second_last_s.size(0), -1)
                    if second_last_t.dim() > 2:
                        second_last_t = second_last_t.view(second_last_t.size(0), -1)
                    try:
                        simkd_module = module_list[1]  # 模型在module_list中的位置
                        second_last_s_transformed = simkd_module(second_last_s)
                        loss_kd = criterion_kd(second_last_s_transformed, second_last_t)
                    except Exception as e:
                        print(f"SimKD错误: {e}")
            else:
                raise NotImplementedError(f"未实现的蒸馏方法: {distill_method}")
                
            # 添加到列表
            loss_kd_list.append(loss_kd * getattr(criterion_kd, 'weight', opt.beta))

        # 合并所有损失
        loss = opt.cls * loss_cls + opt.div * loss_div
        for loss_kd in loss_kd_list:
            loss += loss_kd
        
        losses.update(loss.item(), input.size(0))

        # ===================backward=====================
        # 梯度累积 (如果需要)
        if hasattr(opt, 'gradient_accumulation_steps') and opt.gradient_accumulation_steps > 1:
            loss = loss / opt.gradient_accumulation_steps
            if hasattr(opt, 'use_mixed_precision') and opt.use_mixed_precision:
                scaler = optimizer.scaler
                scaler.scale(loss).backward()
                steps_since_update += 1
                if steps_since_update >= opt.gradient_accumulation_steps:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    steps_since_update = 0
            else:
                loss.backward()
                steps_since_update += 1
                if steps_since_update >= opt.gradient_accumulation_steps:
                    optimizer.step()
                    optimizer.zero_grad()
                    steps_since_update = 0
        else:
            # 标准优化步骤
            if hasattr(opt, 'use_mixed_precision') and opt.use_mixed_precision:
                scaler = optimizer.scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ===================meters=====================
        # 精度计算
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # 计算平均损失和精度
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), loss=losses, top1=top1, top5=top5))

        # 清理内存
        if idx % 50 == 0:
            torch.cuda.empty_cache()

    # 确保在梯度累积时完成最后一步更新
    if hasattr(opt, 'gradient_accumulation_steps') and opt.gradient_accumulation_steps > 1 and steps_since_update > 0:
        optimizer.step()
        optimizer.zero_grad()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg



def validate_vanilla(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            # 处理不同的数据格式
            if isinstance(batch_data, (tuple, list)):
                if len(batch_data) == 4:  # CRD format
                    images, labels, _, _ = batch_data
                elif len(batch_data) == 3:  # Instance format
                    images, labels, _ = batch_data
                else:  # Standard format
                    images, labels = batch_data
            else:
                images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, labels)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def validate_distill(val_loader, module_list, criterion, opt):
    """validation"""

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    for module in module_list:
        module.eval()

    model_s = module_list[0]
    model_t = module_list[-1]
    n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):

            # Handle batch_data format
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) == 2:
                    images, labels = batch_data
                else:
                    # Extract the first two values (images and labels)
                    images, labels = batch_data[0], batch_data[1]
            elif isinstance(batch_data, dict):
                # Handle DALI format
                images, labels = batch_data['data'], batch_data['label'].squeeze().long()
            else:
                raise ValueError(f"Unsupported batch_data format: {type(batch_data)}")

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            if opt.distill == 'simkd':
                feat_s, _ = model_s(images, is_feat=True)
                feat_t, _ = model_t(images, is_feat=True)
                feat_t = [f.detach() for f in feat_t]
                cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else \
                model_t.get_feat_modules()[-1]
                _, _, output = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            else:
                output = model_s(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                    idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1)  # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret

    return top1.avg, top5.avg, losses.avg

