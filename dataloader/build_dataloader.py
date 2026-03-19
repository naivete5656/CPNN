import torch
from torch.utils.data import DataLoader

from .dataset import SimpleDataset, collate_fn
from .our_dataset import PropDataset


def build_dataloader(args, cfg):
    trainset = eval(cfg.DATASET.dataset)("train", args, cfg)
    train_loader = DataLoader(
        trainset,
        batch_size=cfg.TRAINING.batch_size,
        num_workers=cfg.TRAINING.num_workers,
        collate_fn=eval(cfg.DATASET.collate_fn),
        pin_memory=True,
        shuffle=True,
    )

    valset = eval(cfg.DATASET.testdataset)("val", args, cfg)
    test_batch_size = 1
    if cfg.DATASET.test_batch_size != {}:
        test_batch_size = cfg.DATASET.test_batch_size
    val_loader = DataLoader(
        valset,
        batch_size=test_batch_size,
        num_workers=cfg.TRAINING.num_workers,
        collate_fn=eval(cfg.DATASET.collate_fn),
        pin_memory=True,
        shuffle=False,
    )

    testset = eval(cfg.DATASET.testdataset)("test", args, cfg)
    test_loader = DataLoader(
        testset,
        batch_size=test_batch_size,
        num_workers=cfg.TRAINING.num_workers,
        collate_fn=eval(cfg.DATASET.collate_fn),
        pin_memory=True,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def build_train_test_loader(fold, cfg):
    trainset = eval(cfg.DATASET.testdataset)(split="train", fold=fold, run_name=cfg.GENERAL.run_name, **cfg.DATASET)
    custom_collate_test = eval(cfg.DATASET.collate_fn_test)

    test_loader = DataLoader(
        trainset,
        batch_size=1,
        collate_fn=custom_collate_test,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )

    return test_loader
