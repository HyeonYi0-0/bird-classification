import wandb
import datetime
import os
import time
import warnings

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from torch import nn
from main import evaluate, train_one_epoch, load_data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import json

arg = None

def main():
    global arg
    wandb.init(config=arg, project="bird-classification-sweeps")
    args = wandb.config
    print(args)

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
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        val_score = evaluate(model, criterion, data_loader_test, device=device)
        lr_scheduler.step()
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed (default: 42)")
    parser.add_argument("--transform-ver", default=0, type=int, metavar="N", help="transform version (default: 0)")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
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
    
    # parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    # parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
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
    arg = get_args_parser().parse_args()
    #wandb sweep setting
    # current dir path
    json_path = os.path.join(os.getcwd(), "configs", "sweep_config.json")
    # open json file
    with open(json_path, "r") as f:
        sweep_config = json.load(f)
    
    sweep_id = wandb.sweep(sweep_config, project="bird-classification-sweeps")
    wandb.agent(sweep_id, function=main, count=3)