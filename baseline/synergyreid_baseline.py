from __future__ import absolute_import, print_function

import argparse
import os.path as osp
import sys

import numpy as np
import torch
from reid import datasets, models
from reid.dist_metric import DistanceMetric
from reid.evaluators import Evaluator
from reid.trainers import Trainer
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import (
    load_checkpoint,
    save_checkpoint,
    write_mat_csv,
)
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader


def get_device(device):
    "return torch device"
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps":
        return torch.device("mps")
    if device == "cpu":
        return torch.device("cpu")
    raise NotImplementedError


def get_data(
    name,
    split_id,
    data_dir,
    height,
    width,
    batch_size,
    workers,
    combine_traintest,
):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_set = dataset.traintest if combine_traintest else dataset.train
    num_classes = (
        dataset.num_traintest_ids
        if combine_traintest
        else dataset.num_train_ids
    )

    train_transformer = T.Compose(
        [
            T.RandomSizedRectCrop(height, width),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
        ]
    )

    test_transformer = T.Compose(
        [
            T.RectScale(height, width),
            T.ToTensor(),
            normalizer,
        ]
    )

    train_loader = DataLoader(
        Preprocessor(
            train_set, root=dataset.images_dir, transform=train_transformer
        ),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        Preprocessor(
            list(set(dataset.query_test) | set(dataset.gallery_test)),
            root=dataset.images_dir,
            transform=test_transformer,
        ),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=True,
    )

    challenge_loader = DataLoader(
        Preprocessor(
            list(
                set(dataset.query_challenge) | set(dataset.gallery_challenge)
            ),
            root=dataset.images_dir,
            transform=test_transformer,
        ),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=True,
    )

    return dataset, num_classes, train_loader, test_loader, challenge_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, "log.txt"))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (256, 128)
    (
        dataset,
        num_classes,
        train_loader,
        test_loader,
        challenge_loader,
    ) = get_data(
        args.dataset,
        args.split,
        args.data_dir,
        args.height,
        args.width,
        args.batch_size,
        args.workers,
        args.combine_traintest,
    )
    # Device
    device = get_device(args.device)

    # Create model
    model = models.create(
        args.arch,
        num_features=args.features,
        dropout=args.dropout,
        num_classes=num_classes,
    )

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint["epoch"]
        best_top1 = checkpoint["best_top1"]
        print(
            "=> Start epoch {}  best top1 {:.1%}".format(
                start_epoch, best_top1
            )
        )
    model = nn.DataParallel(model).to(device)

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        metric.train(model, train_loader)
        print("Test set:")
        dist_matrix = evaluator.evaluate(
            test_loader, dataset.query_test, dataset.gallery_test, metric
        )
        top1 = evaluator.compute_score(
            dist_matrix, dataset.query_test, dataset.gallery_test
        )
        write_mat_csv(
            osp.join(args.logs_dir, "test_distance_matrix.csv"),
            dist_matrix,
            dataset.query_test,
            dataset.gallery_test,
        )
        print("Test set Top1 : {}".format(top1))
        print("Challenge:")
        dist_matrix = evaluator.evaluate(
            challenge_loader,
            dataset.query_challenge,
            dataset.gallery_challenge,
            metric,
        )
        write_mat_csv(
            osp.join(args.logs_dir, "challenge_distance_matrix.csv"),
            dist_matrix,
            dataset.query_challenge,
            dataset.gallery_challenge,
        )
        return

    # Criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Optimizer
    if hasattr(model.module, "base"):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [
            p for p in model.parameters() if id(p) not in base_param_ids
        ]
        param_groups = [
            {"params": model.module.base.parameters(), "lr_mult": 0.1},
            {"params": new_params, "lr_mult": 1.0},
        ]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(
        param_groups,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # Trainer
    trainer = Trainer(model, criterion, device)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 60 if args.arch == "inception" else 40
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)
        if epoch < args.start_save:
            continue
        dist_matrix = evaluator.evaluate(
            test_loader, dataset.query_test, dataset.gallery_test
        )
        top1 = evaluator.compute_score(
            dist_matrix, dataset.query_test, dataset.gallery_test
        )
        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint(
            {
                "state_dict": model.module.state_dict(),
                "epoch": epoch + 1,
                "best_top1": best_top1,
            },
            is_best,
            fpath=osp.join(args.logs_dir, "checkpoint.pth.tar"),
        )

        print(
            "\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n".format(
                epoch, top1, best_top1, " *" if is_best else ""
            )
        )

    # # Final test
    print("Test with best model:")
    checkpoint = load_checkpoint(osp.join(args.logs_dir, "model_best.pth.tar"))
    model.module.load_state_dict(checkpoint["state_dict"])
    metric.train(model, train_loader)
    dist_matrix = evaluator.evaluate(
        challenge_loader,
        dataset.query_challenge,
        dataset.gallery_challenge,
        metric,
    )
    write_mat_csv(
        osp.join(args.logs_dir, "challenge_distance_matrix.csv"),
        dist_matrix,
        dataset.query_challenge,
        dataset.gallery_challenge,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synergy ReID baseline")
    # data
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="synergyreid",
        choices=datasets.names(),
    )
    parser.add_argument("-b", "--batch-size", type=int, default=128)
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument(
        "--height", type=int, help="input height, default: 256 for resnet*"
    )
    parser.add_argument(
        "--width", type=int, help="input width, default: 128 for resnet*"
    )
    parser.add_argument(
        "--combine-traintest",
        action="store_true",
        help="train and test sets together for training, "
        "test set alone for validation",
    )
    # model
    parser.add_argument(
        "-a", "--arch", type=str, default="resnet50", choices=models.names()
    )
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    # optimizer
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="learning rate of new parameters, for pretrained "
        "parameters it is 10 times smaller than this",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    # training configs
    parser.add_argument("--resume", type=str, default="", metavar="PATH")
    parser.add_argument(
        "--evaluate", action="store_true", help="evaluation only"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--start_save",
        type=int,
        default=0,
        help="start saving checkpoints after specific epoch",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print-freq", type=int, default=1)
    # metric learning
    parser.add_argument(
        "--dist-metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "kissme", "lsml"],
    )
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument(
        "--data-dir",
        type=str,
        metavar="PATH",
        default=osp.join(working_dir, "data"),
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        metavar="PATH",
        default=osp.join(working_dir, "logs"),
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cuda",
    )
    main(parser.parse_args())
