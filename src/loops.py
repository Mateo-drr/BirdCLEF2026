"""
Train and Eval loops
"""
from collections import defaultdict
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import utils

def run_model(model: nn.Module, samples: dict, criterion, config) -> dict:
    """
    Run the model on the given input samples.

    Args:
        model (nn.Module): Model
        samples (dict): Input samples.
        criterion
        config
    Returns:
        dict
    """
    # Move data to device
    data = samples["data"].to(config.device)
    # Run the model
    output = model(data)
    # Calc loss
    loss = criterion(data, output)

    return {
        "loss": loss,
        "data": data,
        "output": output,
    }


def train_loop(
        model: nn.Module,
        train_dl: DataLoader,
        criterion,
        optim: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        wb_metrics: dict,
        config,
        epoch: int,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
):
    """
    Train loop
    Args:
        model (nn.Module): Model
        train_dl (DataLoader): DataLoader
        criterion
        optim (torch.optim.Optimizer): Optimizer
        scaler (torch.amp.GradScaler): GradScaler
        wb_metrics (dict): metrics dictionary to hold epoch results
        config (Any): config
        epoch (int): current epoch
        scheduler (torch.optim.lr_scheduler.LRScheduler): scheduler
    """
    model.train()
    results = defaultdict(list)

    for sample in tqdm(train_dl,desc=f"Epoch {epoch + 1}/{config.num_epochs}"):
        optim.zero_grad()
        if config.half_p:
            with torch.amp.autocast(device_type=config.device):
                outputs = run_model(model, sample, criterion, config)

                scaler.scale(outputs["loss"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optim)
                scaler.update()
        else:
            outputs = run_model(model, sample, criterion, config)
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optim.step()

        # update metrics and loss
        outputs.pop("data")
        outputs.pop("output")
        results = utils.format_metrics(outputs, results)

    scheduler.step() if scheduler is not None else None

    utils.epoch_results(results, wb_metrics, split="train")


def eval_loop(
        model: nn.Module,
        dataloader: DataLoader,
        criterion,
        eval_name: str,
        wb_metrics: dict,
        config,
):
    """
    Validation loop
    Args:
        model (nn.Module): Model
        dataloader (DataLoader): DataLoader
        criterion: Loss function
        eval_name: Evaluation name being run, e.g. valid
        wb_metrics (dict): metrics dictionary to hold epoch results
        config (Any): config
    """
    model.eval()
    results = defaultdict(list)
    with torch.no_grad():
        for sample in tqdm(dataloader, desc=f"{eval_name}"):
            if config.half_p:
                with torch.amp.autocast(device_type=config.device):
                    outputs = run_model(model, sample, criterion, config)
            else:
                outputs = run_model(model, sample, criterion, config)

            # update metrics and loss
            outputs.pop("data")
            outputs.pop("output")
            results = utils.format_metrics(outputs, results)

    utils.epoch_results(results, wb_metrics, split="valid")
