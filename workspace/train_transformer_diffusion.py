import argparse
import os
import time
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import dataset as dataset_module
from model.pdiff import PDiff as DiffusionModel
from model.pdiff import OneDimVAE
from model.diffusion import DDPMSampler, DDIMSampler


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer-based diffusion model with VAE latent")

    # Dataset
    parser.add_argument("--dataset", type=str, default="NumberCkpt_200",
                        help="Dataset class name defined in dataset/__init__.py (e.g., NumberCkpt_200)")
    parser.add_argument("--dim-per-token", type=int, default=64, dest="dim_per_token",
                        help="Token width used to split parameters (divide_slice_length)")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)

    # Training
    parser.add_argument("--seed", type=int, default=430)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vae-steps", type=int, default=500)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="learning_rate")
    parser.add_argument("--vae-learning-rate", type=float, default=2e-5, dest="vae_learning_rate")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--save-minutes", type=float, default=0.0,
                        help="Additionally save checkpoint every M minutes (0 to disable)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoint")

    # Early stopping
    parser.add_argument("--patience", type=int, default=200,
                        help="Number of steps without sufficient loss improvement to stop training")
    parser.add_argument("--min-delta", type=float, default=1e-4, dest="min_delta",
                        help="Minimum loss improvement to reset patience")

    # Diffusion hyperparameters
    parser.add_argument("--model-dim", type=int, default=128, dest="model_dim",
                        help="Latent dimension of VAE (and diffusion sequence length)")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddpm")

    # Transformer denoiser
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-pos-encoding", action="store_true",
                        help="Disable sinusoidal positional encoding")

    # VAE options
    parser.add_argument("--vae-channels", type=str, default="64,128,256,256,32",
                        help="Comma-separated encoder channel sizes for VAE")
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--checkpoint-noise", type=float, default=1e-3,
                        help="Noise added to input params during VAE training")
    parser.add_argument("--latent-noise", type=float, default=1e-1,
                        help="Manual latent std used in VAE reparameterization during training")

    # Sampling during training
    parser.add_argument("--generate-interval", type=int, default=2,
                        help="Sampling interval for saving intermediate generated params (in steps)")

    # W&B
    parser.add_argument("--wandb-project", type=str, default="NN-Diffusion-Transformer")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated tags")
    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="online")

    return parser.parse_args()


def build_components(args: argparse.Namespace):
    DatasetClass = getattr(dataset_module, args.dataset)
    train_set = DatasetClass(dim_per_token=args.dim_per_token, granularity=0, pe_granularity=0, fill_value=0.0)
    if isinstance(train_set.sequence_length, torch.Tensor):
        seq_len = int(train_set.sequence_length.item())
    else:
        seq_len = int(train_set.sequence_length)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=False,
        drop_last=True,
        shuffle=True,
    )

    # Configure diffusion model to use Transformer denoiser over VAE latent
    DiffusionModel.config = {
        "arch": "transformer",
        "model_dim": args.model_dim,
        "time_embedding_dim": args.model_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "use_positional_encoding": (not args.no_pos_encoding),
        "sample_mode": DDPMSampler if args.sampler == "ddpm" else DDIMSampler,
        "beta": (args.beta_start, args.beta_end),
        "T": args.T,
        # Legacy CNN params kept for compatibility; ignored by transformer
        "layer_channels": [1, 64, 128, 256, 512, 256, 128, 64, 1],
        "kernel_size": args.kernel_size,
    }
    diffusion = DiffusionModel(sequence_length=seq_len)

    # VAE operates over checkpoint-tokenized sequence; latent size equals model_dim
    channels = [int(x) for x in args.vae_channels.split(",") if x]
    vae = OneDimVAE(d_model=channels, d_latent=args.model_dim, sequence_length=seq_len, kernel_size=args.kernel_size)

    return train_set, train_loader, diffusion, vae


@torch.no_grad()
def sample_and_decode(model: DiffusionModel, vae: OneDimVAE, device: str):
    model.eval()
    vae.eval()
    mu = model(sample=True)  # [1, d_latent]
    pred = vae.decode(mu)
    return pred


def train(args: argparse.Namespace) -> None:
    # Optional wandb
    use_wandb = args.wandb_mode != "disabled"
    if use_wandb:
        import wandb
        mode = args.wandb_mode
        tags = None if args.wandb_tags is None else [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            group=args.wandb_group,
            tags=tags,
            mode=mode,
            config=vars(args),
        )

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_set, train_loader, model, vae = build_components(args)
    model.to(device)
    vae.to(device)

    # Optimizers and schedulers
    vae_optim = optim.AdamW(vae.parameters(), lr=args.vae_learning_rate, weight_decay=args.weight_decay)
    diff_optim = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    vae_sched = CosineAnnealingLR(vae_optim, T_max=args.vae_steps)
    diff_sched = CosineAnnealingLR(diff_optim, T_max=args.diffusion_steps)

    global_step = 0
    best_loss = float("inf"))
    bad_steps = 0
    last_time_save = time.time()

    # 1) Train VAE
    model.eval()
    vae.train()
    for batch_idx, (param, _) in enumerate(train_loader):
        if batch_idx >= args.vae_steps:
            break
        param = param.flatten(start_dim=1).to(device)
        param = param + torch.randn_like(param) * args.checkpoint_noise
        vae_optim.zero_grad(set_to_none=True)
        loss = vae(x=param, use_var=True, manual_std=args.latent_noise, kld_weight=0.0)
        loss.backward()
        vae_optim.step()
        vae_sched.step()
        if (batch_idx + 1) % args.print_every == 0:
            if use_wandb:
                wandb.log({"vae_loss": float(loss.item())}, step=batch_idx + 1)

    # 2) Train diffusion over VAE latent
    model.train()
    vae.eval()
    for batch_idx, (param, _) in enumerate(train_loader):
        if batch_idx >= args.diffusion_steps:
            break
        param = param.flatten(start_dim=1).to(device)
        with torch.no_grad():
            mu, _ = vae.encode(param)
        diff_optim.zero_grad(set_to_none=True)
        loss = model(x=mu)
        loss.backward()
        diff_optim.step()
        diff_sched.step()

        global_step += 1
        if (global_step % args.print_every) == 0:
            if use_wandb:
                wandb.log({"train_loss": float(loss.item()), "step": global_step})
            # early stopping on training loss plateau
            if best_loss - float(loss.item()) >= args.min_delta:
                best_loss = float(loss.item())
                bad_steps = 0
            else:
                bad_steps += args.print_every
            if bad_steps >= args.patience:
                print(f"Early stopping at step {global_step} (no improvement >= {args.min_delta} for {args.patience} steps)")
                break

        # checkpoint by steps
        if (global_step % args.save_every) == 0:
            save_state(args, model, vae, global_step, use_wandb)
            try_generate(args, model, vae, train_set, use_wandb)

        # checkpoint by time interval
        if args.save_minutes > 0.0 and (time.time() - last_time_save) >= args.save_minutes * 60.0:
            save_state(args, model, vae, global_step, use_wandb)
            last_time_save = time.time()

    # final save
    save_state(args, model, vae, global_step, use_wandb, final=True)
    try_generate(args, model, vae, train_set, use_wandb)

    if use_wandb:
        wandb.finish()


def save_state(args: argparse.Namespace, model: DiffusionModel, vae: OneDimVAE, step: int, use_wandb: bool, final: bool = False) -> None:
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tag = args.wandb_name or f"{args.dataset.lower()}-transformer"
    suffix = "final" if final else f"step{step}"
    path = os.path.join(args.checkpoint_dir, f"{tag}-{suffix}.pth")
    state = {
        "diffusion": model.state_dict(),
        "vae": vae.state_dict(),
        "args": vars(args),
        "step": step,
    }
    torch.save(state, path)
    print(f"Saved checkpoint: {path}")
    if use_wandb:
        import wandb
        artifact = wandb.Artifact(name=f"{tag}-ckpt", type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)


def try_generate(args: argparse.Namespace, model: DiffusionModel, vae: OneDimVAE, train_set, use_wandb: bool) -> None:
    try:
        with torch.no_grad():
            pred = sample_and_decode(model, vae, device=args.device)
            gen_norm = float(pred.abs().mean().item())
        print(f"Generated_norm: {gen_norm:.6f}")
        if use_wandb:
            import wandb
            wandb.log({"generated_norm": gen_norm})
        # Optionally save to dataset generated path for downstream testing
        save_path = train_set.generated_path
        train_set.save_params(pred, save_path=save_path)
    except Exception as e:  # noqa: BLE001
        print(f"Generation skipped due to error: {e}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
