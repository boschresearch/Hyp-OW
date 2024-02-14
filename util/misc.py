# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor



import random
from collections import deque
import numpy as np
from models.pmath import  poincare_mean
import pdb
import sys
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
            
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
if float(torchvision.__version__.split('.')[1]) < 5:
    import math
    from torchvision.ops.misc import _NewEmptyTensorOp
    def _check_size_scale_factor(dim, size, scale_factor):
        # type: (int, Optional[List[int]], Optional[float]) -> None
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if not (scale_factor is not None and len(scale_factor) != dim):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )
    def _output_size(dim, input, size, scale_factor):
        # type: (int, Tensor, Optional[List[int]], Optional[float]) -> List[int]
        assert dim == 2
        _check_size_scale_factor(dim, size, scale_factor)
        if size is not None:
            return size
        # if dim is not 2 or scale_factor is iterable use _ntuple instead of concat
        assert scale_factor is not None and isinstance(scale_factor, (int, float))
        scale_factors = [scale_factor, scale_factor]
        # math.floor might return float in py2.7
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
        ]
elif float(torchvision.__version__.split('.')[1]) < 7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def combine_dict(input_dict, average=True):
    world_size = get_world_size()
    output = []

    if world_size < 2:
        return [input_dict]
    
    for i in range(world_size):
        output.append(None)
        
    with torch.no_grad():
        dist.all_gather_object(output, input_dict)
    return output


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ['LOCAL_SIZE'])


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__.split('.')[1]) < 7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        if float(torchvision.__version__.split('.')[1]) < 5:
            return _NewEmptyTensorOp.apply(input, output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class Store:
    def __init__(self, total_num_classes, items_per_class, shuffle=False,cfg=None):
        self.shuffle = shuffle
        self.items_per_class = items_per_class
        self.total_num_classes = total_num_classes
        self.cfg=cfg
        if 'debug' in cfg.train_set:
            self.store = [deque(maxlen=self.items_per_class)  if i<self.total_num_classes-2 else deque(maxlen=self.items_per_class*20) for i in range(self.total_num_classes)]
        else:
            
            self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]
        self.mean_list=None
        self.poincare_mean=None
        self.poincare_family_mean=None
        
        ## there are 11 categories
        # self.poincare_family_mean=torch.zeros(11,self.cfg.hidden_dim).to(self.cfg.device)
        
        if 'HIERARCHICAL' in self.cfg.dataset:
            if 't1' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1]).long(),\
                                    1:torch.Tensor([2,3]).long(),  \
                                    2:torch.Tensor([4,5,6]).long(),\
                                      3:torch.Tensor([7]).long(),\
                                      4:torch.Tensor([8,9]).long(),\
                                      5:torch.Tensor([10,11]).long(),\
                                      6:torch.Tensor([12,13]).long(),\
                                      7:torch.Tensor([14,15]).long(),\
                                      8:torch.Tensor([16]).long(),\
                                       9:torch.Tensor([17,18]).long(),\
                                      10:torch.Tensor([19]).long(),\
                                   }
            elif 't2' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,  21,22  ]).long(),\
                                    1:torch.Tensor([2,3,   23]).long(),  \
                                    2:torch.Tensor([4,5,6,    24,25]).long(),\
                                      3:torch.Tensor([7,   26]).long(),\
                                      4:torch.Tensor([8,9,   27,28]).long(),\
                                      5:torch.Tensor([10,11,   29]).long(),\
                                      6:torch.Tensor([12,13,   30,31,32]).long(),\
                                      7:torch.Tensor([14,15,   33,34]).long(),\
                                      8:torch.Tensor([16,       37]).long(),\
                                       9:torch.Tensor([17,18,    35,36]).long(),\
                                      10:torch.Tensor([19,   38,39]).long(),\
                                     
                                     11:torch.Tensor([20]).long(),
                                    
                                   }
            elif 't3' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,      21,22 ,      40,41 ]).long(),\
                                    1:torch.Tensor([2,3,       23,          42]).long(),  \
                                    2:torch.Tensor([4,5,6,     24,25,       43,44,45]).long(),\
                                      3:torch.Tensor([7,       26,          46]).long(),\
                                      4:torch.Tensor([8,9,     27,28,       47,48,49]).long(),\
                                      5:torch.Tensor([10,11,   29,          50,51]).long(),\
                                      6:torch.Tensor([12,13,   30,31,32,    52,53]).long(),\
                                      7:torch.Tensor([14,15,   33,34,       54]).long(),\
                                      8:torch.Tensor([16,      37,          55,56]).long(),\
                                       9:torch.Tensor([17,18,  35,36,       57]).long(),\
                                      10:torch.Tensor([19,     38,39,        58,59]).long(),\
                                     
                                     11:torch.Tensor([20]).long(),
                                    
                                   }
            elif 't4' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,      21,22 ,      40,41,          60,61 ]).long(),\
                                    1:torch.Tensor([2,3,       23,          42,             62]).long(),  \
                                    2:torch.Tensor([4,5,6,     24,25,       43,44,45,       63,64]).long(),\
                                      3:torch.Tensor([7,       26,          46,             65,66,]).long(),\
                                      4:torch.Tensor([8,9,     27,28,       47,48,49,       67,68,69]).long(),\
                                      5:torch.Tensor([10,11,   29,          50,51,          70,71]).long(),\
                                      6:torch.Tensor([12,13,   30,31,32,    52,53,          72,73,74]).long(),\
                                      7:torch.Tensor([14,15,   33,34,       54,             75]).long(),\
                                      8:torch.Tensor([16,      37,          55,56,          76]).long(),\
                                       9:torch.Tensor([17,18,  35,36,       57,             77]).long(),\
                                      10:torch.Tensor([19,     38,39,        58,59,         78,79]).long(),\
                                     
                                     11:torch.Tensor([20]).long(),
                                    
                                   }
                
                
                
        elif 'TOWOD' in self.cfg.dataset:
            if 't1' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,5,6,13,18]).long(),\
                                1:torch.Tensor([2,7,9,11,12,16]).long(),  \
                                2:torch.Tensor([4]).long(),\
                                  3:torch.Tensor([8,10,15,17]).long(),\
                                  4:torch.Tensor([14]).long(),\
                                  5:torch.Tensor([19]).long(),
                               }
            elif 't2' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,5,6,13,18,              20]).long(),\
                                1:torch.Tensor([2,7,9,11,12,16,              26,27,28,29 ]).long(),  \
                                2:torch.Tensor([4]).long(),\
                                  3:torch.Tensor([8,10,15,17]).long(),\
                                  4:torch.Tensor([14]).long(),\
                                  5:torch.Tensor([19]).long(),
                                     6:torch.Tensor([                               21,22,23,24,25]).long(),
                                     7:torch.Tensor([                               30,31,32,33,34]).long(),
                                      8:torch.Tensor([                               35,36,37,38,39]).long(),
                               }
                
            elif 't3' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,5,6,13,18,              20]).long(),\
                                1:torch.Tensor([2,7,9,11,12,16,               26,27,28,29 ]).long(),  \
                                2:torch.Tensor([4]).long(),\
                                  3:torch.Tensor([8,10,15,17]).long(),\
                                  4:torch.Tensor([14]).long(),\
                                  5:torch.Tensor([19]).long(),
                                     6:torch.Tensor([                               21,22,23,24,25]).long(),
                                     7:torch.Tensor([                               30,31,32,33,34]).long(),
                                      8:torch.Tensor([                               35,36,37,38,39]).long(),
                                     9:torch.Tensor([                                40,41,42,43,44,45,46,47,48,49]).long(),\
                                     10:torch.Tensor([                                50,51,52,53,54,55,56,57,58,59]).long(),
                               }
            elif 't4' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,5,6,13,18,              20]).long(),\
                                1:torch.Tensor([2,7,9,11,12,16,               26,27,28,29 ]).long(),  \
                                2:torch.Tensor([4,                                                                                        74,75,76,77,78,79]).long(),\
                                  3:torch.Tensor([8,10,15,17,                                                                             60,61     ]).long(),\
                                  4:torch.Tensor([14]).long(),\
                                  5:torch.Tensor([19,                                                                                     62,63,64,65,66]).long(),
                                     6:torch.Tensor([                               21,22,23,24,25]).long(),
                                     7:torch.Tensor([                               30,31,32,33,34]).long(),
                                      8:torch.Tensor([                               35,36,37,38,39]).long(),
                                     9:torch.Tensor([                                40,41,42,43,44,45,46,47,48,49]).long(),\
                                     10:torch.Tensor([                                50,51,52,53,54,55,56,57,58,59]).long(),
                                     11:torch.Tensor([                                                                                    67,68,69,70,71,72,73]).long(),
                               }
                
                
        elif 'OWDETR' in self.cfg.dataset:
            if 't1' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                2:torch.Tensor([18]).long(),\
                               }
            elif 't2' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                2:torch.Tensor([18]).long(),\
                                     
                                3:torch.Tensor([                                 19,20,21,22,23]).long(),\
                                4:torch.Tensor([                                 24,25,26,37,38,39]).long(),
                                5:torch.Tensor([                                 27,28,29,30,31]).long(),
                                6:torch.Tensor([                                 32,33,34,35,36]).long(),
                            
                               }
            elif 't3' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                2:torch.Tensor([18]).long(),\
                                     
                                3:torch.Tensor([                                 19,20,21,22,23]).long(),\
                                4:torch.Tensor([                                 24,25,26,37,38,39]).long(),
                                5:torch.Tensor([                                 27,28,29,30,31]).long(),
                                6:torch.Tensor([                                 32,33,34,35,36]).long(),
                                7:torch.Tensor([                                                        40,41,42,43,44,45,46,47,48,49]).long(),
                                8:torch.Tensor([                                                        50,51,52,53,54,55,56,57,58,59]).long(),
                            
                               }
            elif 't4' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                2:torch.Tensor([18]).long(),\
                                     
                                3:torch.Tensor([                                 19,20,21,22,23]).long(),\
                                4:torch.Tensor([                                 24,25,26,37,38,39]).long(),
                                5:torch.Tensor([                                 27,28,29,30,31]).long(),
                                6:torch.Tensor([                                 32,33,34,35,36]).long(),
                                7:torch.Tensor([                                                        40,41,42,43,44,45,46,47,48,49]).long(),
                                8:torch.Tensor([                                                        50,51,52,53,54,55,56,57,58,59]).long(),
                                     
                                9:torch.Tensor([                                                                                             60,61,62,63,64,78]).long(),
                                10:torch.Tensor([                                                                                            65,66,67,68,69,70,71]).long(),
                                11:torch.Tensor([                                                                                            72,73,74,75,76,77,79]).long(),
                            
                               }
            
                
        
    
    def update(self,args):
        self.cfg=args
        if 'HIERARCHICAL' in self.cfg.dataset:
            if 't1' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1]).long(),\
                                    1:torch.Tensor([2,3]).long(),  \
                                    2:torch.Tensor([4,5,6]).long(),\
                                      3:torch.Tensor([7]).long(),\
                                      4:torch.Tensor([8,9]).long(),\
                                      5:torch.Tensor([10,11]).long(),\
                                      6:torch.Tensor([12,13]).long(),\
                                      7:torch.Tensor([14,15]).long(),\
                                      8:torch.Tensor([16]).long(),\
                                       9:torch.Tensor([17,18]).long(),\
                                      10:torch.Tensor([19]).long(),\
                                   }
            elif 't2' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,  21,22  ]).long(),\
                                    1:torch.Tensor([2,3,   23]).long(),  \
                                    2:torch.Tensor([4,5,6,    24,25]).long(),\
                                      3:torch.Tensor([7,   26]).long(),\
                                      4:torch.Tensor([8,9,   27,28]).long(),\
                                      5:torch.Tensor([10,11,   29]).long(),\
                                      6:torch.Tensor([12,13,   30,31,32]).long(),\
                                      7:torch.Tensor([14,15,   33,34]).long(),\
                                      8:torch.Tensor([16,       37]).long(),\
                                       9:torch.Tensor([17,18,    35,36]).long(),\
                                      10:torch.Tensor([19,   38,39]).long(),\
                                     
                                     11:torch.Tensor([20]).long(),
                                    
                                   }
            elif 't3' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,      21,22 ,      40,41 ]).long(),\
                                    1:torch.Tensor([2,3,       23,          42]).long(),  \
                                    2:torch.Tensor([4,5,6,     24,25,       43,44,45]).long(),\
                                      3:torch.Tensor([7,       26,          46]).long(),\
                                      4:torch.Tensor([8,9,     27,28,       47,48,49]).long(),\
                                      5:torch.Tensor([10,11,   29,          50,51]).long(),\
                                      6:torch.Tensor([12,13,   30,31,32,    52,53]).long(),\
                                      7:torch.Tensor([14,15,   33,34,       54]).long(),\
                                      8:torch.Tensor([16,      37,          55,56]).long(),\
                                       9:torch.Tensor([17,18,  35,36,       57]).long(),\
                                      10:torch.Tensor([19,     38,39,        58,59]).long(),\
                                     
                                     11:torch.Tensor([20]).long(),
                                    
                                   }
            elif 't4' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,      21,22 ,      40,41,          60,61 ]).long(),\
                                    1:torch.Tensor([2,3,       23,          42,             62]).long(),  \
                                    2:torch.Tensor([4,5,6,     24,25,       43,44,45,       63,64]).long(),\
                                      3:torch.Tensor([7,       26,          46,             65,66,]).long(),\
                                      4:torch.Tensor([8,9,     27,28,       47,48,49,       67,68,69]).long(),\
                                      5:torch.Tensor([10,11,   29,          50,51,          70,71]).long(),\
                                      6:torch.Tensor([12,13,   30,31,32,    52,53,          72,73,74]).long(),\
                                      7:torch.Tensor([14,15,   33,34,       54,             75]).long(),\
                                      8:torch.Tensor([16,      37,          55,56,          76]).long(),\
                                       9:torch.Tensor([17,18,  35,36,       57,             77]).long(),\
                                      10:torch.Tensor([19,     38,39,        58,59,         78,79]).long(),\
                                     
                                     11:torch.Tensor([20]).long(),
                                    
                                   }
                
        elif 'TOWOD' in self.cfg.dataset:
            if 't1' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,5,6,13,18]).long(),\
                                1:torch.Tensor([2,7,9,11,12,16]).long(),  \
                                2:torch.Tensor([4]).long(),\
                                  3:torch.Tensor([8,10,15,17]).long(),\
                                  4:torch.Tensor([14]).long(),\
                                  5:torch.Tensor([19]).long(),
                               }
            elif 't2' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,5,6,13,18,              20]).long(),\
                                1:torch.Tensor([2,7,9,11,12,16,                 26,27,28,29 ]).long(),  \
                                2:torch.Tensor([4]).long(),\
                                  3:torch.Tensor([8,10,15,17]).long(),\
                                  4:torch.Tensor([14]).long(),\
                                  5:torch.Tensor([19]).long(),
                                     6:torch.Tensor([                               21,22,23,24,25]).long(),
                                     7:torch.Tensor([                               30,31,32,33,34]).long(),
                                      8:torch.Tensor([                               35,36,37,38,39]).long(),
                               }
                
            elif 't3' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,5,6,13,18,              20]).long(),\
                                1:torch.Tensor([2,7,9,11,12,16,                  26,27,28,29 ]).long(),  \
                                2:torch.Tensor([4]).long(),\
                                  3:torch.Tensor([8,10,15,17]).long(),\
                                  4:torch.Tensor([14]).long(),\
                                  5:torch.Tensor([19]).long(),
                                     6:torch.Tensor([                               21,22,23,24,25]).long(),
                                     7:torch.Tensor([                               30,31,32,33,34]).long(),
                                      8:torch.Tensor([                               35,36,37,38,39]).long(),
                                     9:torch.Tensor([                                40,41,42,43,44,45,46,47,48,49]).long(),\
                                     10:torch.Tensor([                                50,51,52,53,54,55,56,57,58,59]).long(),
                               }
            elif 't4' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,5,6,13,18,              20]).long(),\
                                1:torch.Tensor([2,7,9,11,12,16,               26,27,28,29 ]).long(),  \
                                2:torch.Tensor([4,                                                                                        74,75,76,77,78,79]).long(),\
                                  3:torch.Tensor([8,10,15,17,                                                                             60,61     ]).long(),\
                                  4:torch.Tensor([14]).long(),\
                                  5:torch.Tensor([19,                                                                                     62,63,64,65,66]).long(),
                                     6:torch.Tensor([                               21,22,23,24,25]).long(),
                                     7:torch.Tensor([                               30,31,32,33,34]).long(),
                                      8:torch.Tensor([                               35,36,37,38,39]).long(),
                                     9:torch.Tensor([                                40,41,42,43,44,45,46,47,48,49]).long(),\
                                     10:torch.Tensor([                                50,51,52,53,54,55,56,57,58,59]).long(),
                                     11:torch.Tensor([                                                                                    67,68,69,70,71,72,73]).long(),
                               }
                
                
        elif 'OWDETR' in self.cfg.dataset:
            if 't1' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                2:torch.Tensor([18]).long(),\
                               }
            elif 't2' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                2:torch.Tensor([18]).long(),\
                                     
                                3:torch.Tensor([                                 19,20,21,22,23]).long(),\
                                4:torch.Tensor([                                 24,25,26,37,38,39]).long(),
                                5:torch.Tensor([                                 27,28,29,30,31]).long(),
                                6:torch.Tensor([                                 32,33,34,35,36]).long(),
                            
                               }
            elif 't3' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                2:torch.Tensor([18]).long(),\
                                     
                                3:torch.Tensor([                                 19,20,21,22,23]).long(),\
                                4:torch.Tensor([                                 24,25,26,37,38,39]).long(),
                                5:torch.Tensor([                                 27,28,29,30,31]).long(),
                                6:torch.Tensor([                                 32,33,34,35,36]).long(),
                                7:torch.Tensor([                                                        40,41,42,43,44,45,46,47,48,49]).long(),
                                8:torch.Tensor([                                                        50,51,52,53,54,55,56,57,58,59]).long(),
                            
                               }
            elif 't4' in self.cfg.train_set:
                self.family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                2:torch.Tensor([18]).long(),\
                                     
                                3:torch.Tensor([                                 19,20,21,22,23]).long(),\
                                4:torch.Tensor([                                 24,25,26,37,38,39]).long(),
                                5:torch.Tensor([                                 27,28,29,30,31]).long(),
                                6:torch.Tensor([                                 32,33,34,35,36]).long(),
                                7:torch.Tensor([                                                        40,41,42,43,44,45,46,47,48,49]).long(),
                                8:torch.Tensor([                                                        50,51,52,53,54,55,56,57,58,59]).long(),
                                     
                                9:torch.Tensor([                                                                                             60,61,62,63,64,78]).long(),
                                10:torch.Tensor([                                                                                            65,66,67,68,69,70,71]).long(),
                                11:torch.Tensor([                                                                                            72,73,74,75,76,77,79]).long(),
                            
                               }
       
        
    def add(self, items, class_ids):
        for idx, class_id in enumerate(class_ids):
            # ForkedPdb().set_trace()
            self.store[class_id].append(items[idx])
            
    def retrieve(self, class_id):
        if class_id != -1:
            items = []
            for item in self.store[class_id]:
                #items.extend(list(item))
                items.extend(item.view(1,-1))
            
            if self.shuffle:
                random.shuffle(items)
            if len(items)>0:
                items=torch.stack(items)
                
            return items
        else:
            all_items = []
            for i in range(self.total_num_classes):
                items = []
                for item in self.store[i]:
                    #items.append(list(item))
                    items.extend(item.view(1,-1))
                if len(items)>0:
                    all_items.append(torch.stack(items))
                else:
                    all_items.append(items)
            # ForkedPdb().set_trace()
            return all_items
    
    def form_poincare_mean(self,pointcare_mean=None):
        if pointcare_mean is None:
            cc=self.retrieve(-1)
            self.poincare_mean=torch.zeros(self.cfg.num_classes,self.cfg.hidden_dim).to(self.cfg.device)
            for i in range(len(cc)):
                if len(cc[i])>0:
                    #ForkedPdb().set_trace()
                    self.poincare_mean[i]=poincare_mean(cc[i],c=self.cfg.hyperbolic_c)
        else:
            self.poincare_mean=pointcare_mean
        # self.poincare_mean=self.poincare_mean.to(self.cfg.device)
        
                
        if self.cfg.family_regularizer: 
            #ForkedPdb().set_trace()
            self.poincare_family_mean=torch.zeros(len(self.family_mapping.keys()),self.cfg.hidden_dim).to(self.cfg.device)
            #ForkedPdb().set_trace()
            for key in self.family_mapping.keys():
            # ForkedPdb().set_trace()
            # self.family_mapping[key]
            
                for ind in self.family_mapping[key]:
                    if len(cc[ind])>0:
                        try:
                            mat=torch.cat((mat,cc[ind]))
                        except:
                            mat=cc[ind]
  
                self.poincare_family_mean[key]=poincare_mean(mat,c=self.cfg.hyperbolic_c)

        
    
    
        
                
            
        
    def return_positive_pairs(self,target_classes_o):
        liste=[[] for _ in range(self.cfg.samples_per_category)]
        # counter=0

            
        for idx,element in enumerate(target_classes_o):
            if len(self.store[element])>0:
                
                index=torch.randint(low=0, high=len(self.store[element]),size=(self.cfg.samples_per_category,))
                
                torch.randint(low=0, high=1,size=(3,))
    
                for ij in range(len(liste)):
                    liste[ij].append(self.store[element][index[ij]])
                
                assert len(liste[ij])!=0
               
            else:
                # counter=1
                return None
     
        final_liste=[torch.stack(liste[oo]) for oo in range(len(liste))]
      
        return torch.stack(final_liste) ## [3,batch,256]
                
        
    def retrieve_and_tensorize(self):
       all_elements=self.retrieve(-1)
       elements_in_list=[element for element in all_elements if len(element)>0]
       return torch.cat(elements_in_list)

    def __str__(self):
        s = self.__class__.__name__ + '('
        for idx, item in enumerate(self.store):
            s += '\n Class ' + str(idx) + ' --> ' + str(len(list(item))) + ' items'
        s = s + ' )'
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return sum([len(s) for s in self.store])
    
    def update_params(self,min_length=1):
        all_elements=self.retrieve(-1)
        mean_list=[element.mean(dim=0) for element in all_elements if len(element)>min_length]
        
    
        try:
            
        
            self.mean_list=torch.stack(mean_list)
            # self.icov_list=torch.stack(icov_list)
        except:
            print('Not enough Data in the list')