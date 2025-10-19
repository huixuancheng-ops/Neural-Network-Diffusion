import sys, os
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Neural-Network-Diffusion")+1]) if "Neural-Network-Diffusion" in __file__ else os.getcwd()
sys.path.append(root)
os.chdir(root)

import argparse
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from accelerate.utils import DistributedDataParallelKwargs, AutocastKwargs
from accelerate import Accelerator

from model.pdiff import PDiff as Model
from model.pdiff import OneDimVAE as VAE
from model.diffusion import DDPMSampler, DDIMSampler
from torch.utils.data import DataLoader
import dataset as dataset_module

try:
    import wandb  # optional; only used when enabled via CLI
except Exception:  # pragma: no cover
    wandb = None


def set_global_seed(seed: int) -> None:
    """Set global RNG seeds for reproducibility across Python, NumPy, and Torch."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


def resolve_dataset(name: str):
    """Return dataset class from dataset module by class name string.

    Raises:
        ValueError: if the provided dataset name is not found in dataset module.
    """
    cls = getattr(dataset_module, name, None)
    if cls is None:
        available = [k for k in dir(dataset_module) if k[0].isupper()]
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return cls


def build_dataloaders(DatasetCls, batch_size: int, num_workers: int, dim_per_token: int, **dataset_kwargs):
    """Instantiate dataset and wrap with DataLoader for training.

    Returns:
        train_set: dataset instance
        train_loader: DataLoader for training data
        sequence_length: integer length for model's 1D latent vector
    """
    train_set = DatasetCls(dim_per_token=dim_per_token, **dataset_kwargs)
    sequence_length = train_set.sequence_length * dim_per_token
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=True,
        shuffle=True,
    )
    return train_set, train_loader, sequence_length


def build_model_and_optimizers(
    sequence_length: int,
    model_dim: int,
    transformer_config: dict,
    learning_rate: float,
    vae_learning_rate: float,
    weight_decay: float,
    sample_mode: str,
    beta: tuple,
    T: int,
):
    """Construct Model (diffusion + denoiser) and VAE plus their optimizers and schedulers.

    Returns:
        vae, model, vae_optimizer, optimizer, vae_scheduler, scheduler
    """
    # model config
    model_config = {
        "model_dim": model_dim,
        "sample_mode": DDPMSampler if sample_mode.lower() == "ddpm" else DDIMSampler,
        "beta": beta,
        "T": T,
        "transformer_config": transformer_config,
    }
    Model.config = model_config

    model = Model(sequence_length=sequence_length)
    vae = VAE(
        d_model=[64, 128, 256, 256, 32],
        d_latent=model_dim,
        sequence_length=sequence_length,
        kernel_size=7,
    )

    vae_optimizer = optim.AdamW(vae.parameters(), lr=vae_learning_rate, weight_decay=weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    vae_scheduler = CosineAnnealingLR(optimizer=vae_optimizer, T_max=1000000)  # stepped per-iter
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=1000000)  # stepped per-iter

    return vae, model, vae_optimizer, optimizer, vae_scheduler, scheduler


def train_vae(
    accelerator: Accelerator,
    vae: VAE,
    vae_optimizer: optim.Optimizer,
    vae_scheduler: CosineAnnealingLR,
    train_loader: DataLoader,
    max_steps: int,
    print_every: int,
    use_wandb: bool,
):
    """Pre-train the VAE to reconstruct tokenized parameter vectors.

    The VAE is trained with MSE reconstruction loss. KL is optional and disabled here.
    """
    vae.train()
    steps = 0
    running = 0.0
    for batch_idx, (param, _) in enumerate(train_loader):
        if steps >= max_steps:
            break
        vae_optimizer.zero_grad()
        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=True)):
            param = param.flatten(start_dim=1)
            param = param + torch.randn_like(param) * 0.001
            loss = vae(x=param, use_var=True, manual_std=0.1, kld_weight=0.0)
        accelerator.backward(loss)
        vae_optimizer.step()
        if accelerator.is_main_process:
            vae_scheduler.step()
        steps += 1
        running += loss.item()
        if (steps % print_every == 0) and accelerator.is_main_process:
            avg = running / print_every
            print(f"[VAE] step={steps} loss={avg:.6f}")
            running = 0.0
        if use_wandb and accelerator.is_main_process and wandb is not None:
            wandb.log({"vae_loss": loss.item(), "vae_step": steps})


def train_diffusion(
    accelerator: Accelerator,
    model: Model,
    vae: VAE,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    train_loader: DataLoader,
    train_set,
    total_steps: int,
    print_every: int,
    save_every: int,
    checkpoint_dir: str,
    checkpoint_tag: str,
    test_command: str,
    generated_path: str,
    use_wandb: bool,
    log_artifacts: bool,
):
    """Train the diffusion model to predict noise in latent space.

    Periodically saves checkpoints (model + vae), logs training curves to wandb,
    and optionally generates a parameter file followed by running dataset's test script.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.train()
    steps = 0
    running = 0.0

    def save_checkpoint(tag_suffix: str = ""):
        path = os.path.join(checkpoint_dir, f"{checkpoint_tag}{tag_suffix}.pth")
        state = {"diffusion": accelerator.unwrap_model(model).state_dict(), "vae": vae.state_dict()}
        torch.save(state, path)
        if use_wandb and accelerator.is_main_process and wandb is not None and log_artifacts:
            art = wandb.Artifact(name=f"ckpt-{checkpoint_tag}{tag_suffix}", type="model")
            art.add_file(path)
            wandb.log_artifact(art)
        return path

    for batch_idx, (param, _) in enumerate(train_loader):
        if steps >= total_steps:
            break
        optimizer.zero_grad()
        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=True)):
            param = param.flatten(start_dim=1)
            with torch.no_grad():
                mu, _ = vae.encode(param)
            loss = model(x=mu)
        accelerator.backward(loss)
        optimizer.step()
        if accelerator.is_main_process:
            scheduler.step()
        steps += 1
        running += loss.item()

        if (steps % print_every == 0) and accelerator.is_main_process:
            avg = running / print_every
            print(f"[Diffusion] step={steps} loss={avg:.6f}")
            running = 0.0
        if use_wandb and accelerator.is_main_process and wandb is not None:
            wandb.log({"train_loss": loss.item(), "train_step": steps})
        
        if (steps % save_every == 0) and accelerator.is_main_process:
            ckpt_path = save_checkpoint(tag_suffix=f"-s{steps}")
            # Optional: generate and test
            generate(model, vae, generated_path, train_set=train_set, use_wandb=use_wandb)
            if test_command:
                os.system(test_command)
                print()

    if accelerator.is_main_process:
        final_ckpt = save_checkpoint(tag_suffix="-final")
        print(f"Saved final checkpoint to {final_ckpt}")


def generate(model: Model, vae: VAE, save_path: str, train_set, use_wandb: bool = False):
    """Sample from diffusion model in latent space, decode with VAE, and persist parameters."""
    model.eval()
    with torch.no_grad():
        mu = model(sample=True)
        prediction = vae.decode(mu)
        generated_norm = prediction.abs().mean()
    if use_wandb and wandb is not None:
        wandb.log({"generated_norm": generated_norm.item()})
    # Save generated params via dataset-aware postprocess
    dim_per_token = train_set.dim_per_token
    prediction = prediction.view(-1, dim_per_token)
    train_set.save_params(prediction, save_path=save_path)
    model.train()


def main():
    parser = argparse.ArgumentParser(description="Train diffusion with a 4-layer Transformer denoiser")

    # Dataset/training general
    parser.add_argument("--dataset", type=str, required=True, help="Dataset class name from dataset module, e.g. Cifar10_ResNet18")
    parser.add_argument("--seed", type=int, default=430)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dim-per-token", type=int, default=64)

    # Steps and optimization
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--vae-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4, help="Diffusion learning rate")
    parser.add_argument("--vae-lr", type=float, default=2e-5, help="VAE learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)

    # Diffusion process
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--sample-mode", type=str, choices=["ddpm", "ddim"], default="ddpm")

    # Model dimensions
    parser.add_argument("--model-dim", type=int, default=128, help="Latent vector length for diffusion/vae")

    # Transformer denoiser
    parser.add_argument("--tf-embed-dim", type=int, default=128, help="Transformer token feature dimension")
    parser.add_argument("--tf-num-layers", type=int, default=4, help="Number of Transformer encoder layers")
    parser.add_argument("--tf-num-heads", type=int, default=8)
    parser.add_argument("--tf-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--tf-dropout", type=float, default=0.0)

    # Checkpointing and logging
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoint")
    parser.add_argument("--tag", type=str, default=None, help="Run tag. If not set, auto-generated")

    # WandB
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="AR-Param-Generation")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-artifacts", action="store_true", help="Upload checkpoints as artifacts")

    args = parser.parse_args()

    # Seed
    set_global_seed(args.seed)

    # Dataset
    DatasetCls = resolve_dataset(args.dataset)
    train_set, train_loader, sequence_length = build_dataloaders(
        DatasetCls, args.batch_size, args.num_workers, args.dim_per_token,
        granularity=0, pe_granularity=0, fill_value=0.0,
    )

    # Config and models
    transformer_config = dict(
        embed_dim=args.tf_embed_dim,
        num_layers=args.tf_num_layers,
        num_heads=args.tf_num_heads,
        mlp_ratio=args.tf_mlp_ratio,
        dropout=args.tf_dropout,
    )

    vae, model, vae_optimizer, optimizer, vae_scheduler, scheduler = build_model_and_optimizers(
        sequence_length=sequence_length,
        model_dim=args.model_dim,
        transformer_config=transformer_config,
        learning_rate=args.lr,
        vae_learning_rate=args.vae_lr,
        weight_decay=args.weight_decay,
        sample_mode=args.sample_mode,
        beta=(args.beta_start, args.beta_end),
        T=args.T,
    )

    # Accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    vae, model, vae_optimizer, optimizer, train_loader = accelerator.prepare(
        vae, model, vae_optimizer, optimizer, train_loader
    )

    # WandB
    tag = args.tag or f"{args.dataset}_tf{args.tf_num_layers}L_{int(time.time())}"
    if args.wandb and (wandb is not None):
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or tag,
            mode=args.wandb_mode,
            config={
                "dataset": args.dataset,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "total_steps": args.total_steps,
                "vae_steps": args.vae_steps,
                "lr": args.lr,
                "vae_lr": args.vae_lr,
                "weight_decay": args.weight_decay,
                "print_every": args.print_every,
                "save_every": args.save_every,
                "T": args.T,
                "beta": (args.beta_start, args.beta_end),
                "sample_mode": args.sample_mode,
                "model_dim": args.model_dim,
                "transformer_config": transformer_config,
            },
        )

    # Training
    train_vae(
        accelerator=accelerator,
        vae=vae,
        vae_optimizer=vae_optimizer,
        vae_scheduler=vae_scheduler,
        train_loader=train_loader,
        max_steps=args.vae_steps,
        print_every=args.print_every,
        use_wandb=args.wandb and (wandb is not None),
    )

    # Unwrap VAE for later saving consistency
    vae = accelerator.unwrap_model(vae)

    train_diffusion(
        accelerator=accelerator,
        model=model,
        vae=vae,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        train_set=train_set,
        total_steps=args.total_steps,
        print_every=args.print_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_tag=tag,
        test_command=DatasetCls.test_command,
        generated_path=DatasetCls.generated_path,
        use_wandb=args.wandb and (wandb is not None),
        log_artifacts=args.wandb_log_artifacts,
    )

    if args.wandb and (wandb is not None):
        wandb.finish()


if __name__ == "__main__":
    main()
