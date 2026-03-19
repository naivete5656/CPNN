import argparse
from pathlib import Path

import pytorch_lightning as pl

from utils import load_config, load_loggers, load_callbacks, naming_function
from dataloader import build_dataloader
from model import build_lgmodel
from main import get_parse


def main(args, cfg):
    # define dataset
    train_loader, val_loader, test_loader = build_dataloader(args, cfg)

    # add options
    args.run_name = naming_function(args)

    # define model
    model = build_lgmodel(args, cfg)

    # test
    trainer = pl.Trainer(
        accelerator="gpu",
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        enable_model_summary=True,
    )

    model.save_path = Path(f"outputs")
    model.save_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(f"{cfg.GENERAL.ckpt_path}/{args.run_name}/best_model.ckpt")

    trainer.test(
        model,
        test_loader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    args = get_parse()
    if args.method in ["AbMIL", "CLAM_MB", "DSMIL", "ILRA", "SumExpModel"]:
        args.config_dir = f"{args.config_dir}/train_cfg.yaml"
    else:
        args.config_dir = f"{args.config_dir}/{args.method}.yaml"
    cfg = load_config(args.config_dir, args.trainer)

    main(args, cfg)
