from __future__ import print_function, division
from cProfile import label

import sys
import time
import torch
from .util import AverageMeter, accuracy, reduce_tensor
from crd.criterion import CRDLoss

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
            images, labels = batch_data
        else:
            images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()
        
        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # ===================forward=====================
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
    """one epoch distillation"""
    # 设置模式
    for module in module_list[:-1]:
        module.train()
    module_list[-1].eval()

    # 获取模型和损失函数
    model_s = module_list[0]
    model_t = module_list[-1]
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd_list = criterion_list[2:]

    # 更新CRD损失的epoch信息和类别信息
    if 'crd' in opt.distill:
        for criterion in criterion_list:
            if isinstance(criterion, CRDLoss):
                criterion.update_epoch(epoch)
                # 如果数据集有类别索引信息，则更新到CRD中
                if hasattr(train_loader.dataset, 'class_indices'):
                    criterion.update_class_info(train_loader.dataset.class_indices)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size

    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        # 处理输入数据
        if opt.dali is None:
            if 'crd' in opt.distill:
                images, labels, index, contrast_idx = data
            else:
                images, labels = data
                index = contrast_idx = None
        else:
            images, labels = data[0]['data'], data[0]['label'].squeeze().long()
            index = contrast_idx = None

        # 移动数据到GPU
        if opt.gpu is not None:
            images = images.cuda(opt.gpu, non_blocking=True)
            labels = labels.cuda(opt.gpu, non_blocking=True)
            if index is not None:
                index = index.cuda(opt.gpu, non_blocking=True)
            if contrast_idx is not None:
                contrast_idx = contrast_idx.cuda(opt.gpu, non_blocking=True)

        # 前向传播
        feat_s, logit_s = model_s(images, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(images, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        # 计算分类损失和KL散度损失
        loss_cls = criterion_cls(logit_s, labels)
        loss_div = criterion_div(logit_s, logit_t)

        # 计算其他蒸馏损失
        loss_kd_total = 0
        for i, distill_method in enumerate(opt.distill):
            criterion_kd = criterion_kd_list[i]
            if distill_method == 'kd':
                loss_kd = criterion_kd(logit_s, logit_t)
                loss_kd_total += opt.beta_kd * loss_kd
            elif distill_method == 'crd':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd = criterion_kd(f_s, f_t, index, contrast_idx, teacher_logits=logit_t)
                loss_kd_total += opt.beta_crd * loss_kd
            elif distill_method == 'hint':
                f_s = feat_s[opt.hint_layer]
                f_t = feat_t[opt.hint_layer]
                loss_kd = criterion_kd(f_s, f_t)
                loss_kd_total += opt.beta_hint * loss_kd
            elif distill_method == 'afd':
                # 使用指定的层进行蒸馏
                selected_feat_s = [feat_s[i] for i in criterion_kd.hint_layers]
                selected_feat_t = [feat_t[i] for i in criterion_kd.guide_layers]
                loss_kd = criterion_kd(selected_feat_s, selected_feat_t)
                loss_kd_total += opt.beta_afd * loss_kd
            else:
                raise NotImplementedError(distill_method)

        # 总损失
        loss = opt.cls * loss_cls + opt.div * loss_div + loss_kd_total

        # 计算准确率
        metrics = accuracy(logit_s, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计时
        batch_time.update(time.time() - end)
        end = time.time()

        # 打印信息
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, n_batch, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

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

