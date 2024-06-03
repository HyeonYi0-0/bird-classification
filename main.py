import datetime
import os
import time
import hashlib

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from torch import nn
from torchvision.transforms.functional import InterpolationMode
from dataset import ZipDataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import json
from models import BaseModel

import wandb

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        # automatic mixed precision (float16 <-> float32) 
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        # backpropagation을 하기전에 gradients를 zero로 만들어주고 시작을 해야함 
        optimizer.zero_grad()
        # gradient clipping to prevent gradient explosion and gradient vanishing (주로 RNN 계열, 학습 안정화)
        if scaler is not None:
            # gradient scaling to prevent underflow (gradient vanishing problem)
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        # metric_logger.update(train_loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["train/loss"].update(loss.item())
        metric_logger.meters["train/acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["train/acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"
    
    loss_key = "eval/loss" if log_suffix == "" else f"eval/{log_suffix}_loss"
    acc1_key = "eval/acc1" if log_suffix == "" else f"eval/{log_suffix}_acc1"
    acc5_key = "eval/acc5" if log_suffix == "" else f"eval/{log_suffix}_acc5"

    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            
            batch_size = image.shape[0]
            # metric_logger.update(eval_loss=loss.item())
            metric_logger.meters[loss_key].update(loss.item())
            metric_logger.meters[acc1_key].update(acc1.item(), n=batch_size)
            metric_logger.meters[acc5_key].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    
    # wandb.log({"Acc@1": metric_logger.eval_acc1.global_avg, "Acc@5": metric_logger.eval_acc5.global_avg})

    acc1 = metric_logger.meters[acc1_key].global_avg
    acc5 = metric_logger.meters[acc5_key].global_avg
    print(f"{header} Acc@1 {acc1:.3f} Acc@5 {acc5:.3f}")
    return metric_logger.meters[acc1_key].global_avg

def inference(model, test_loader, device):
    model.eval()
    preds, true_target = [], []
    with torch.inference_mode():
        for image, target in iter(test_loader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(image)

            preds += output.argmax(1).detach().cpu().numpy().tolist()
            true_target += target.detach().cpu().numpy().tolist()

    return preds, true_target

def get_debug_source(le, dataset_test, preds, true_targets):
    acc_analysis = {key: {k:0 for k in le.classes_} for key in ["correct", "incorrect_pred", "incorrect_target"]}
    img_data = {"correct" : {}, "incorrect" : {}}
    for i, (pred, true_target) in enumerate(zip(preds, true_targets)):
        if pred == true_target:
            acc_analysis["correct"][le.classes_[true_target]] += 1
            if ((len(img_data["correct"].keys()) < 5) and (true_target not in img_data["correct"].keys())):
                img_data["correct"][true_target] = dataset_test.data[i]
        else :
            acc_analysis["incorrect_pred"][le.classes_[pred]] += 1
            acc_analysis["incorrect_target"][le.classes_[true_target]] += 1
            if ((len(img_data["incorrect"].keys()) < 10) and ((pred, true_target) not in img_data["incorrect"].keys())):
                img_data["incorrect"][(pred, true_target)] = dataset_test.data[i]
                
    return acc_analysis, img_data

def _get_cache_path(filename, ver: int):

    root_dir = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha1(filename.encode()).hexdigest()
    version = "T{0}".format(ver)
    cache_path = os.path.join("datasets", version, h[:10] + ".pt")
    cache_path = os.path.join(root_dir, cache_path)
    return cache_path


def load_data(train_data, val_data, file_name, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)
    data_file_path = os.path.join(args.data_path, file_name)
    transform_log = {}

    print("Loading training data")
    st = time.time()
    trainT_ver = "trainT{0}".format(args.transform_ver)
    cache_path = _get_cache_path(trainT_ver, args.transform_ver)
    if (args.cache_dataset and os.path.exists(cache_path)) and (not args.sweep_state):
        print(f"Loading dataset_train from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        # We need a default value for the variables below because args may come
        # from train_quantization.py which doesn't define them.
        # auto_augment_policy = getattr(args, "auto_augment", None)
        # random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        preprocessing = presets.ClassificationPresetTrain(
                            crop_size=train_crop_size,
                            interpolation=interpolation,
                            auto_augment_policy=args.auto_augment,
                            random_erase_prob=args.random_erase,
                            ra_magnitude=ra_magnitude,
                            augmix_severity=augmix_severity,
                            backend=args.backend,
                        )
        
        dataset = ZipDataset(
            img_path_list=train_data['img_path'].values,
            label_list=train_data['label'].values,
            zipfile_path=data_file_path,
            transform=preprocessing
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, trainT_ver), cache_path)
            
        transform_log["train_transform"] = [str(t) for t in preprocessing.transforms.transforms]
    print("Took", time.time() - st)

    print("Loading validation data")
    st = time.time()
    valT_ver = "valT{0}".format(args.transform_ver)
    cache_path = _get_cache_path(valT_ver, args.transform_ver)
    if args.cache_dataset and os.path.exists(cache_path):
        print(f"Loading dataset_test from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])
                transform_log["eval_transform"] = [str(t) for t in preprocessing.transforms]
            else :
              transform_log["eval_transform"] = [str(preprocessing)]

        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
            )

            transform_log["eval_transform"] = [str(t) for t in preprocessing.transforms.transforms]

        dataset_test = ZipDataset(
            img_path_list=val_data['img_path'].values,
            label_list=val_data['label'].values,
            zipfile_path=data_file_path,
            transform=preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valT_ver), cache_path)
    print("Took", time.time() - st)
    
    cache_dir = os.path.dirname(cache_path)
    if not os.path.exists(cache_dir) :
        utils.mkdir(cache_dir)
        save_path = os.path.join(cache_dir, "transforms.json")
        with open(save_path, "w") as f :
            json.dump(transform_log, f, indent=2)
    
    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

def main(args):
    if args.output_dir:
        resume_path = os.path.join(args.output_dir, args.resume) if args.resume != "" else None
        args.resume = resume_path
        
        output_dir = os.path.join(args.output_dir, args.exp)
        utils.mkdir(output_dir)
        args.output_dir = output_dir
    
    print(args)
    
    wandb.init(name=args.exp, config=args, id=hashlib.sha1(args.exp.encode()).hexdigest()[:10], resume="allow")

    # args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    
    utils.seed_everything(args.seed)

    zip_file_name = 'open.zip'
    df = utils.read_csv_from_zip('train.csv', zip_file_name, args.data_path)
    # 학습, 검증 데이터 분리
    train_data, val_data, _, _ = train_test_split(df, df['label'], test_size=0.3, stratify=df['label'], random_state=args.seed)
    # 데이터 라벨 인코딩
    le = preprocessing.LabelEncoder()
    train_data['label'] = le.fit_transform(train_data['label'])
    val_data['label'] = le.transform(val_data['label'])
    
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_data, val_data, zip_file_name, args)

    num_classes = len(le.classes_)
    collate_fn = utils.get_collate_fn(args, num_classes=num_classes)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    model = torchvision.models.efficientnet_b0(num_classes=num_classes, pretrained=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )
    
    # Optimizer
    optimizer = utils.get_optimizer(args, parameters)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Learning rate scheduler
    lr_scheduler = utils.get_lr_scheduler(args, optimizer)

    model_without_ddp = model

    model_ema = None
    if args.model_ema:
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only and not args.model_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1
        if model_ema and not args.model_only:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            # evaluate(model, criterion, data_loader_test, device=device)
            preds, true_targets = inference(model, data_loader_test, device=device)
            acc_analysis, img_data = get_debug_source(le, dataset_test, preds, true_targets)
            print(acc_analysis)
            print(img_data)
            utils.log_bar_plot(acc_analysis)
            print("fin")
            utils.log_correct_ig_results(model, le, img_data, args, device)
            utils.log_incorrect_ig_results(model, le, img_data, args, device)
            
        return
    
    # wandb.watch(model, log_freq=args.print_freq)

    print("Start training")
    start_time = time.time()
    best_score = 0
    # stop_cnt = 0
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        val_score = evaluate(model, criterion, data_loader_test, device=device)
        if model_ema:
            val_score = evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            # utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
            if best_score < val_score:
                best_score = val_score
                # stop_cnt = 0
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, "best_model.pth"))
            # else :
            #     stop_cnt += 1
            #     if stop_cnt > 5:
            #         break
        lr_scheduler.step()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--exp", default="test", type=str, help="experiment name")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed (default: 42)")
    parser.add_argument("--transform-ver", default=0, type=int, metavar="N", help="transform version (default: 0)")
    parser.add_argument("--sweep-state", action="store_true", help="only use model only")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--model-only", action="store_true", help="only use model only")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    # optimizer
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
       
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    
    # augmentation strategies
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="linear", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)