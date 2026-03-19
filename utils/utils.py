import os
from pathlib import Path
import yaml
import random

from addict import Dict
import numpy as np

import torch

from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import BasePredictionWriter


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Load config
def load_config(config_name: str, trainer: str):
    """load config file in Dict

    Args:
        config_name (str): Name of config file.

    Returns:
        Dict: Dict instance containing configuration.
    """

    with open(config_name, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if trainer != "":
        trainer_config_path = Path(config_name).with_name(trainer + ".yaml")
        if trainer_config_path.exists():
            # {trainer}.yaml を読み込む
            with open(Path(config_name).with_name(trainer + ".yaml"), "r") as f:
                consistency_config = yaml.load(f, Loader=yaml.FullLoader)

            # 既存の設定に consistency.yaml をマージ
            for key, value in consistency_config.items():
                if key in config and isinstance(config[key], dict):
                    # ネストされた辞書があれば更新
                    config[key].update(value)
                else:
                    # 新しいキーを追加
                    config[key] = value

    return Dict(config)


# Load loggers
def load_loggers(run_name, cfg: Dict):
    """Return Logger instance for Trainer in List.

    Args:
        cfg (Dict): Dict containing configuration.

    Returns:
        List: _description_
    """

    csv_logger = CSVLogger(cfg.GENERAL.log_path, name=run_name)

    if cfg.DATASET.debug or "debug" in run_name:
        loggers = [csv_logger]
    elif cfg.GENERAL.wandb:
        wandb_logger = WandbLogger(project="Bulksc", name=run_name, dir=cfg.GENERAL.log_path)
        loggers = [csv_logger, wandb_logger]
    else:
        loggers = [csv_logger]

    return loggers


# load Callback
def load_callbacks(run_name, cfg: Dict):
    """Return Early stopping and Checkpoint Callbacks.

    Args:
        cfg (Dict): Dict containing configuration.

    Returns:
        List: Return List containing the Callbacks.
    """
    ckpt_path = f"{cfg.GENERAL.ckpt_path}/{run_name}"

    Mycallbacks = []
    fname = run_name + "-{epoch:02d}"

    if cfg.TRAINING.early_stopping != {}:
        target = cfg.TRAINING.early_stopping.monitor
        patience = cfg.TRAINING.early_stopping.patience
        mode = cfg.TRAINING.early_stopping.mode

        early_stop_callback = EarlyStopping(monitor=target, min_delta=0.00, patience=patience, verbose=True, mode=mode)
        Mycallbacks.append(early_stop_callback)
        save_top_k = 1

        checkpoint_callback = ModelCheckpoint(
            monitor=target,
            dirpath=ckpt_path,
            filename=fname,
            verbose=True,
            save_last=False,
            save_top_k=save_top_k,
            mode=mode,
            save_weights_only=True,
            enable_version_counter=False,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_path,
            filename=fname,
            every_n_epochs=5,
            save_top_k=-1,
            save_weights_only=True,
            enable_version_counter=False,
        )

    Mycallbacks.append(checkpoint_callback)

    return Mycallbacks


class CustomWriter(BasePredictionWriter):
    def __init__(self, pred_dir, write_interval, emb_dir=None, names=None):
        super().__init__(write_interval)
        self.pred_dir = pred_dir
        self.emb_dir = emb_dir
        self.names = names

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        for i, batch in enumerate(batch_indices[0]):
            torch.save(predictions[0][i][0], os.path.join(self.pred_dir, f"{self.names[i]}.pt"))
            torch.save(predictions[0][i][1], os.path.join(self.emb_dir, f"{self.names[i]}.pt"))


def naming_function(args):
    args.run_name = f"{args.dataset}-{args.method}"

    if args.version != "":
        args.run_name = args.run_name + f"_{args.version}"

    # if args.projector is not None:
    args.run_name = args.run_name + str(args.fold) + args.projector + args.trainer + args.data_type + args.resolution

    return args.run_name
