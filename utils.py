import copy
import datetime
import errno
import hashlib
import os
import time
from collections import defaultdict, deque, OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

import random
import numpy as np
import pandas as pd
from zipfile import ZipFile
from io import BytesIO

from transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import wandb
import re
import matplotlib.pyplot as plt
import captum
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def read_csv_from_zip(csv_file, zipfile, zip_dir):
    csv_data = None
    zipfile_path = os.path.join(zip_dir, zipfile)
    with ZipFile(zipfile_path, 'r') as zipObj:
        zipRead = zipObj.read(csv_file)
        csv_data = pd.read_csv(BytesIO(zipRead))

    return csv_data

def get_collate_fn(args, num_classes):
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes
    )
    if mixup_cutmix is not None:
        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate
    return collate_fn

def get_optimizer(args, parameters):
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")
    return optimizer

def get_lr_scheduler(args, optimizer):
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    elif args.lr_scheduler == "plateau":
        main_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler
    return lr_scheduler

def ig_transform(args, normalize=True):
    transforms = []
    backend = args.backend.lower()
    if backend == "tensor":
        transforms.append(T.PILToTensor())
    elif backend != "pil":
        raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

    transforms += [
        T.Resize(args.val_resize_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(args.val_crop_size),
    ]

    if backend == "pil":
        transforms.append(T.PILToTensor())

    if normalize:
        transforms += [
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]

    transform = T.Compose(transforms)

    return transform

def log_correct_ig_results(model, le, img_data, args, device):
    model.eval()
    integrated_gradients = IntegratedGradients(model)
    columns = ["ig_result", "prediction"]
    dt = wandb.Table(columns = columns)
    print("inte")
    for key, val in img_data["correct"].items() :
        input_img = ig_transform(args, True)(val).unsqueeze(0)
        original_img = ig_transform(args, False)(val)
        # target_idxs.append(key)
    
        input_img = input_img.to(device, non_blocking=True)
        target_idx = torch.tensor(key).to(device, non_blocking=True)
        baseline = torch.ones(input_img.shape).to(device, non_blocking=True)
        b_attributions_ig = integrated_gradients.attribute(input_img, target=target_idx, n_steps=100)
        w_attributions_ig = integrated_gradients.attribute(input_img, baselines=baseline, target=target_idx, n_steps=100)
        attr = b_attributions_ig + w_attributions_ig
        
        attr = np.transpose(attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        original_img = np.transpose(original_img.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        # attr = np.clip(attr, 0, 1)
        plt, _ =  viz.visualize_image_attr_multiple(attr, 
                                                    original_img, 
                                                    ["original_image", "heat_map", "heat_map", "masked_image"],
                                                    ["all", "positive", "negative", "positive"], 
                                                    show_colorbar=True,
                                                    titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                                    fig_size=(32, 8),
                                                    use_pyplot=False)
        row = [wandb.Image(plt), le.classes_[key]]
        dt.add_data(*row)
        
    wandb.log({"correct_ig_result": dt}, commit=False)
    
def log_incorrect_ig_results(model, le, img_data, args, device):
    model.eval()
    integrated_gradients = IntegratedGradients(model)
    columns = ["pred_ig_result", "prediction", "truth_ig_result", "truth"]
    dt = wandb.Table(columns = columns)
    
    for key, val in img_data["incorrect"].items() :
        input_img = ig_transform(args, True)(val).unsqueeze(0)
        original_img = ig_transform(args, False)(val)
        
        input_img = input_img.to(device, non_blocking=True)
        baseline = torch.ones(input_img.shape).to(device, non_blocking=True)
        pred_target_idx = torch.tensor(key[0]).to(device, non_blocking=True)
        true_target_idx = torch.tensor(key[1]).to(device, non_blocking=True)
        b_attributions_ig = integrated_gradients.attribute(input_img, target=pred_target_idx, n_steps=100)
        w_attributions_ig = integrated_gradients.attribute(input_img, baselines=baseline, target=pred_target_idx, n_steps=100)
        pred_attr = b_attributions_ig + w_attributions_ig
        
        b_attributions_ig = integrated_gradients.attribute(input_img, target=true_target_idx, n_steps=100)
        w_attributions_ig = integrated_gradients.attribute(input_img, baselines=baseline, target=true_target_idx, n_steps=100)
        true_attr = b_attributions_ig + w_attributions_ig

        pred_attr = np.transpose(pred_attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        true_attr = np.transpose(true_attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        # pred_attr = np.clip(pred_attr, 0, 1)
        # true_attr = np.clip(true_attr, 0, 1)
        original_img = np.transpose(original_img.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        plt1, _ =  viz.visualize_image_attr_multiple(pred_attr, 
                                                    original_img, 
                                                    ["original_image", "heat_map", "heat_map", "masked_image"],
                                                    ["all", "positive", "negative", "positive"], 
                                                    show_colorbar=True,
                                                    titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                                    fig_size=(32, 8),
                                                    use_pyplot=False)
        plt2, _ =  viz.visualize_image_attr_multiple(true_attr, 
                                                    original_img, 
                                                    ["original_image", "heat_map", "heat_map", "masked_image"],
                                                    ["all", "positive", "negative", "positive"], 
                                                    show_colorbar=True,
                                                    titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                                    fig_size=(32, 8),
                                                    use_pyplot=False)
        row = [wandb.Image(plt1), le.classes_[key[0]], wandb.Image(plt2), le.classes_[key[1]]]
        dt.add_data(*row)
        
    wandb.log({"incorrect_ig_result": dt})
    
        
def log_bar_plot(acc_analysis):
    correct_plt, c_ax = plt.subplots()
    c_ax.bar(acc_analysis["correct"].keys(), acc_analysis["correct"].values(), color='b')
    incorrect_pred_plt, ip_ax = plt.subplots()
    ip_ax.bar(acc_analysis["incorrect_pred"].keys(), acc_analysis["incorrect_pred"].values(), color='r')
    incorrect_traget_plt, it_ax = plt.subplots()
    it_ax.bar(acc_analysis["incorrect_target"].keys(), acc_analysis["incorrect_target"].values(), color='g')
    
    wandb.log({"correct": correct_plt, "incorrect_pred": incorrect_pred_plt, "incorrect_target": incorrect_traget_plt}, commit=False)
    
class SmoothedValue:
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
        t = reduce_across_processes([self.count, self.total])
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
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
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
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                # log_data = {
                #     "eta": eta_seconds,
                #     "iter_time": iter_time.value,
                #     "data_time": data_time.value,
                # }
                # for name, meter in self.meters.items():
                #     log_data[name] = meter.value
                    
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                    
                    # log_data["memory"] = torch.cuda.max_memory_allocated() / MB
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
                
                wandb.log({"lr": self.meters["lr"].value}, step=i) 
            i += 1
            end = time.time()
        
        # wandb log
        match = re.search(r"Epoch: \[(\d+)\]", header)
        epoch_number = 0
        if match:
            epoch_number = int(match.group(1))
        log_data = {
            "iter_time": iter_time.global_avg,
            "data_time": data_time.global_avg,
            "epoch": epoch_number
        }
        for name, meter in self.meters.items():
            log_data[name] = meter.global_avg
        if torch.cuda.is_available():
            log_data["memory"] = torch.cuda.max_memory_allocated() / MB
        log_data["lr"] = self.meters["lr"].value
        wandb.log(log_data, step=i)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
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


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights. Original implementation taken from:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16

    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f, map_location=(lambda s, _: torch.serialization.default_restore_location(s, "cpu")), weights_only=True
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                f"For checkpoint {f}, expected list of params: {params_keys}, but found: {model_params_keys}"
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def store_model_weights(model, checkpoint_path, checkpoint_key="model", strict=True):
    """
    This method can be used to prepare weights files for new models. It receives as
    input a model architecture and a checkpoint from the training script and produces
    a file with the weights ready for release.

    Examples:
        from torchvision import models as M

        # Classification
        model = M.mobilenet_v3_large(weights=None)
        print(store_model_weights(model, './class.pth'))

        # Quantized Classification
        model = M.quantization.mobilenet_v3_large(weights=None, quantize=False)
        model.fuse_model(is_qat=True)
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
        _ = torch.ao.quantization.prepare_qat(model, inplace=True)
        print(store_model_weights(model, './qat.pth'))

        # Object Detection
        model = M.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
        print(store_model_weights(model, './obj.pth'))

        # Segmentation
        model = M.segmentation.deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=True)
        print(store_model_weights(model, './segm.pth', strict=False))

    Args:
        model (pytorch.nn.Module): The model on which the weights will be loaded for validation purposes.
        checkpoint_path (str): The path of the checkpoint we will load.
        checkpoint_key (str, optional): The key of the checkpoint where the model weights are stored.
            Default: "model".
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

    Returns:
        output_path (str): The location where the weights are saved.
    """
    # Store the new model next to the checkpoint_path
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.dirname(checkpoint_path)

    # Deep copy to avoid side effects on the model object.
    model = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Load the weights to the model to validate that everything works
    # and remove unnecessary weights (such as auxiliaries, etc.)
    if checkpoint_key == "model_ema":
        del checkpoint[checkpoint_key]["n_averaged"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint[checkpoint_key], "module.")
    model.load_state_dict(checkpoint[checkpoint_key], strict=strict)

    tmp_path = os.path.join(output_dir, str(model.__hash__()))
    torch.save(model.state_dict(), tmp_path)

    sha256_hash = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hh = sha256_hash.hexdigest()

    output_path = os.path.join(output_dir, "weights-" + str(hh[:8]) + ".pth")
    os.replace(tmp_path, output_path)

    return output_path


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups