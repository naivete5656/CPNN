import torch

from .trainers import (
    LightningModel,
    OurTrainer,
    ComparisonTrainer,
    DeconvExp,
    Mamba2DTrainer,
)
from model.comparisons import (
    SumExpModel,
    AbMIL,
    CLAM_MB,
    DSMIL,
    ILRA,
    AbRegMIL,
    HE2RNA,
    tRNAsformer,
    SEQUOIA,
    SEQUOIA_VIS,
    S4Model,
)


try:
    from model.comparisons import (
        MambaMILvanira,
        SRMambaMIL,
        MambaMIL_2D,
    )
except ImportError:
    pass

from .model import ProtoSum


def build_lgmodel(args, cfg):
    OUT_DIM = {
        "KIRC": 14305,
        "LUAD": 14523,
        "BRCA": 14052,
    }
    IN_DIM = {"feature": 768, "feature_opt": 1536, "feature_conch": 512}
    cfg.MODEL.output_dim = OUT_DIM[args.dataset]
    cfg.MODEL.input_dim = IN_DIM[args.feat_name]

    model = eval(args.method)(args, cfg)

    if args.trainer == "":
        lgmodel = LightningModel(model, args, cfg)
    else:
        lgmodel = eval(args.trainer)(model, args, cfg)

    return lgmodel


def build_model(args, cfg):
    model = eval(args.method)(args, cfg)

    if args.trainer == "DeconvExp":
        lgmodel = DeconvExp
    elif args.trainer == "ComparisonTrainer":
        lgmodel = ComparisonTrainer
    else:
        lgmodel = LightningModel

    return model, lgmodel
