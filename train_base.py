import argparse
import os
import shutil

import torch
import numpy as np
import random

from data_utils import CustomDataModule

import pytorch_lightning as pl
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from lightning_utils import LightningBaseNet

def argparser():
    """
    Parse command line arguments for base model trainer.
    """
    parser = argparse.ArgumentParser(description="Base network trainer")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--min_factor",
        type=float,
        default=0.3,
        help="minimum learning rate factor for linear/cosine scheduler",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="l2 regularization"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--image_size",
        type=int,
        default=-1,
        help="image input size to model, set to -1 to use dataset's default value",
    )
    parser.add_argument(
        "--base_image_size",
        type=int,
        default=-1,
        help="image input size to base model, set to -1 to use dataset's default value",
    )
    parser.add_argument(
        "--architecture", type=str, default="cifar-resnet-50", help="Model Type "
    )
    parser.add_argument("--optimizer", type=str, default="sgd", help="optimizer")
    parser.add_argument(
        "--scheduler", type=str, default="step", help="learning rate scheduler"
    )
    parser.add_argument(
        "--scheduler_step_gamma",
        type=float,
        default=0.2,
        help="scheduler reduction fraction for step scheduler",
    )
    parser.add_argument(
        "--scheduler_step_fraction",
        type=float,
        default=0.3,
        help="scheduler fraction of steps between decays",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=0.0, help="gradient clipping"
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0, help="label_smoothing"
    )
    parser.add_argument("--dataset", type=str, default="cinic10/0_16", help="dataset")
    parser.add_argument(
        "--model_root",
        type=str,
        default="./models/",
        help="model directory",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/",
        help="dataset root directory",
    )
    
    # Differential Privacy parameters
    parser.add_argument("--enable_dp", action="store_true", help="Enable differential privacy training")
    parser.add_argument("--dp_noise_multiplier", type=float, default=1.0, help="Noise multiplier for DP-SGD")
    parser.add_argument("--dp_max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for DP clipping")
    parser.add_argument("--dp_target_epsilon", type=float, default=None, help="Target epsilon for DP (if None, noise_multiplier is used)")
    parser.add_argument("--dp_target_delta", type=float, default=1e-5, help="Target delta for DP")
    parser.add_argument("--dp_secure_rng", action="store_true", help="Use cryptographically secure RNG for DP")

    parser.add_argument(
        "--data_mode",
        type=str,
        default="base",
        help="data mode, either base, mia, or eval",
    )
    parser.add_argument(
        "--DEBUG",
        action="store_true",
        help="debug mode, set to True to run on CPU and with fewer epochs",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="rerun training even if checkpoint exists",
    )
    args = parser.parse_args()

    # Create DP suffix for directory naming
    dp_suffix = ""
    if args.enable_dp:
        dp_parts = []
        if args.dp_target_epsilon is not None:
            dp_parts.append(f"eps{args.dp_target_epsilon}")
            if args.dp_target_delta != 1e-5:  # Only include delta if non-standard
                dp_parts.append(f"delta{args.dp_target_delta}")
        else:
            dp_parts.append(f"noise{args.dp_noise_multiplier}")
        dp_parts.append(f"clip{args.dp_max_grad_norm}")
        if args.dp_secure_rng:
            dp_parts.append("secure")
        dp_suffix = "_dp_" + "_".join(dp_parts)

    args.base_checkpoint_path = os.path.join(
        args.model_root,
        "base",
        args.dataset,
        args.architecture + dp_suffix
    )

    # Set number of base classes.
    if "cifar100" in args.dataset.lower():
        args.num_base_classes = 100
    elif "imagenet-1k" in args.dataset.lower():
        args.num_base_classes = 1000
    elif "cifar20" in args.dataset.lower():
        args.num_base_classes = 20
    elif "purchase" in args.dataset.lower():
        args.num_base_classes = 100
    elif "texas" in args.dataset.lower():
        args.num_base_classes = 100
    else:
        args.num_base_classes = 10

    # Set random seed.
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return args

def train_model(config, args, callbacks=None, rerun=False):
    """
    Pretrain a classification model on a dataset to use as a model to run a QMIA attack on.
    """
    callbacks = callbacks or []
    save_handle = "model.pickle"
    checkpoint_path = os.path.join(args.base_checkpoint_path, save_handle)
    print('Checkpoint path:', checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if (
        os.path.exists(checkpoint_path)
        and not rerun
    ):
        print(f"Checkpoint already exists at {checkpoint_path}. Skipping base model training.")
        return
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    datamodule = CustomDataModule(
        dataset_name=args.dataset,
        num_workers=32 if not args.DEBUG or args.enable_dp else 0,
        image_size=args.image_size,
        base_image_size=args.base_image_size,
        batch_size=args.batch_size,
        data_root=args.data_root,
        stage=args.data_mode,
    )
    
    lightning_model = LightningBaseNet(
        architecture=args.architecture,
        num_classes=args.num_base_classes,
        optimizer_params=config,
        label_smoothing=config["label_smoothing"],
        base_image_size=args.base_image_size,
        enable_dp=args.enable_dp,
        dp_params={
            'noise_multiplier': args.dp_noise_multiplier,
            'max_grad_norm': args.dp_max_grad_norm,
            'target_epsilon': args.dp_target_epsilon,
            'target_delta': args.dp_target_delta,
            'secure_rng': args.dp_secure_rng,
            'epochs': config["epochs"],
        } if args.enable_dp else None,
    )
    
    # For DP training, we need to handle the data loader setup differently
    if args.enable_dp:
        print(f"Differential Privacy enabled:")
        print(f"  Noise multiplier: {args.dp_noise_multiplier}")
        print(f"  Max grad norm: {args.dp_max_grad_norm}")
        print(f"  Target epsilon: {args.dp_target_epsilon}")
        print(f"  Target delta: {args.dp_target_delta}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="ptl/val_acc1",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
        filename="best",
        enable_version_counter=False,
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        filename="last",
        enable_version_counter=False,
    )
    early_stopping_callback = EarlyStopping(
        monitor="ptl/val_acc1",
        patience=10,
        mode="max",
    )


    callbacks = callbacks + [last_checkpoint_callback] + [TQDMProgressBar()] + [checkpoint_callback] # + [early_stopping_callback]

    trainer = pl.Trainer(
        max_epochs=config["epochs"] if not args.DEBUG else 1,
        accelerator="gpu" if not args.DEBUG else "cpu", 
        callbacks=callbacks,
        devices=-1 if not args.DEBUG or args.enable_dp else 1,
        default_root_dir=checkpoint_dir,
        strategy='ddp' if not args.DEBUG else 'ddp',
        gradient_clip_val=config["gradient_clip_val"] if not args.enable_dp else None,  # Disable Lightning's gradient clipping when using DPx
        log_every_n_steps=10,
    )

    torch.set_float32_matmul_precision('medium')
    trainer.logger.log_hyperparams(config)
    if trainer.global_rank == 0:
        print(args.dataset)
    trainer.fit(lightning_model, datamodule=datamodule)
    if trainer.global_rank == 0:
        # reload best network and save just the base model
        try:
            # First try normal loading
            lightning_model = LightningBaseNet.load_from_checkpoint(
                checkpoint_callback.best_model_path
            )
        except RuntimeError as e:
            if "Missing key(s) in state_dict" in str(e) and "model._module." in str(e):
                print("Detected DP-wrapped model state dict. Loading with strict=False and fixing keys...")
                # Load with strict=False to handle key mismatch
                lightning_model = LightningBaseNet.load_from_checkpoint(
                    checkpoint_callback.best_model_path,
                    strict=False
                )
                
                # Load the checkpoint manually and fix the keys
                checkpoint = torch.load(checkpoint_callback.best_model_path, map_location='cpu')
                state_dict = checkpoint['state_dict']
                
                # Create a new state dict with fixed keys (remove _module prefix)
                fixed_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('model._module.'):
                        # Remove the _module prefix
                        new_key = key.replace('model._module.', 'model.')
                        fixed_state_dict[new_key] = value
                    else:
                        fixed_state_dict[key] = value
                
                # Load the fixed state dict
                lightning_model.load_state_dict(fixed_state_dict, strict=False)
                print("Successfully loaded DP-wrapped model with fixed keys")
            else:
                # Re-raise if it's a different error
                raise e
            
        torch.save(lightning_model.model.state_dict(), checkpoint_path)
        print(
            "saved model from {} to {} ".format(
                checkpoint_callback.best_model_path, checkpoint_path
            )
        )
    trainer.strategy.barrier()

if __name__ == "__main__":
    args = argparser()

    config = {
        "lr": args.lr,
        "scheduler": args.scheduler,
        "min_factor": args.min_factor,
        "epochs": args.epochs,
        "opt_type": args.optimizer,
        "weight_decay": args.weight_decay,
        "step_gamma": args.scheduler_step_gamma,
        "step_fraction": args.scheduler_step_fraction,
        "gradient_clip_val": args.grad_clip,
        "label_smoothing": args.label_smoothing,
        "batch_size": args.batch_size,
    }

    train_model(
        config,
        args,
        callbacks=None,
        rerun=args.rerun
    )
