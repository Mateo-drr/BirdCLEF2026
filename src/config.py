""" Configuration used for the model """
from dataclasses import dataclass
from pathlib import Path
import torch.optim as optim
import torch.nn as nn

cwd = Path(__file__).resolve().parent
@dataclass
class Config:
    # training
    threads: int | None = 4  # None to disable
    cudnn_bench: bool = True
    lr: float = 1e-2
    optimizer = optim.AdamW
    criterion = nn.MSELoss
    scheduler = optim.lr_scheduler.CosineAnnealingLR # None to disable it
    grad_clip: float = 1.0
    num_epochs: int = 12
    batch: int = 64
    num_workers: int = 4 # if 0 will disable prefetch factor
    prefetch_factor: int = 4
    device: str = "cpu"
    half_p: bool = True

    # wandb
    wb: bool = False
    project_name: str = "Sample"

    # others
    basePath: Path = cwd
    modelDir: Path = cwd / "weights"

config = Config()

if __name__ == "__main__":
    from pprint import pprint
    pprint(config.__dict__)