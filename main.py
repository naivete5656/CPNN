import argparse
from pathlib import Path

import pytorch_lightning as pl

from utils import load_config, load_loggers, load_callbacks, naming_function
from dataloader import build_dataloader
from model import build_lgmodel

import os


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir",
        type=str,
        default="./config",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ProtoSum",
        choices=[
            "AbMIL",
            "CLAM_MB",
            "DSMIL",
            "ILRA",
            "AbRegMIL",
            "S4Model",
            "SumExpModel",
            "HE2RNA",
            "tRNAsformer",
            "SEQUOIA",
            "SEQUOIA_VIS",
            "SRMambaMIL",
            "MambaMILvanira",
            "MambaMIL_2D",
            "ProtoSum",
        ],
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="DeconvExp",
        choices=[
            "",
            "ComparisonTrainer",
            "DeconvExp",
            "Mamba2DTrainer",
        ],
    )
    parser.add_argument("--dataset", type=str, default="BRCA")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--freeze_weight", action="store_false")
    parser.add_argument("--version", type=str, default="1reg_mse_reg_1e3")
    parser.add_argument("--feat_name", type=str, default="feature_conch")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_type", type=str, default="ts", choices=["ds", "ts", "all"])
    parser.add_argument("--resolution", type=str, default="leiden_res_0.02")
    parser.add_argument("--reg_weight", type=float, default=1e3)
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        choices=["32-true", "64-true", "16-true"],
    )
    parser.add_argument(
        "--projector",
        type=str,
        default="",
        choices=["adapter", "convnex", "", "longformer"],
    )

    args = parser.parse_args()

    return args


def main(args, cfg):
    # define dataset
    train_loader, val_loader, test_loader = build_dataloader(args, cfg)

    # add options
    args.run_name = naming_function(args)

    # define model
    model = build_lgmodel(args, cfg)

    # define logger and callbacks
    cfg.GENERAL.ckpt_path = f"{cfg.GENERAL.ckpt_path}/{args.fold}"
    loggers = load_loggers(args.run_name, cfg)
    callbacks = load_callbacks(args.run_name, cfg)

    # Train or test
    trainer = pl.Trainer(
        gradient_clip_val=0.5,
        # accumulate_grad_batches=16,
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.TRAINING.num_epochs,
        logger=loggers,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        callbacks=callbacks,
        enable_model_summary=True,
        precision=args.precision,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(f"{cfg.GENERAL.ckpt_path}/{args.run_name}/best_model.ckpt", weights_only=True)

    model.save_path = Path(f"outputs/{args.fold}")
    model.save_path.joinpath("validation").mkdir(parents=True, exist_ok=True)
    model.save_path.joinpath("prediction").mkdir(parents=True, exist_ok=True)

    model.save_path.mkdir(parents=True, exist_ok=True)
    trainer.test(model, test_loader, ckpt_path="best")


if __name__ == "__main__":
    args = get_parse()
    if args.method not in ["ProtoSum"]:
        args.config_dir = f"{args.config_dir}/train_cfg.yaml"
    else:
        args.config_dir = f"{args.config_dir}/{args.method}.yaml"
    cfg = load_config(args.config_dir, args.trainer)

    if "mse" in args.version:
        cfg.MODEL.loss_type = "NegativeBinomial"

    main(args, cfg)
