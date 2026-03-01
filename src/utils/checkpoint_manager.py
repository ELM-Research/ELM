import torch
import os
from utils.gpu_manager import is_main


class CheckpointManager:
    def __init__(self, run_dir, args):
        self.run_dir = run_dir
        self.args = args
        self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        self.best_loss = float("inf")
        self.epoch_losses = []
        if is_main():
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch, step, is_best=False, prefix=""):
        if not is_main():
            return
        filename = f"{prefix}epoch_{epoch}_step_{step}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)

        # Handle DDP-wrapped models
        if self.args.distributed:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.optimizer.state_dict(),
            "n_current_steps": optimizer.n_current_steps,
            "best_loss": self.best_loss,
        }
        torch.save(checkpoint, filepath)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{prefix}best.pt")
            torch.save(checkpoint, best_path)

    def resume_checkpoint(self, path, model, optimizer):
        device = next(model.parameters()).device
        ckpt = torch.load(path, map_location=device, weights_only=False)

        # Load model weights
        raw_model = model.module if self.args.distributed else model
        if "model_state_dict" in ckpt:
            raw_model.load_state_dict(ckpt["model_state_dict"])
        elif "state_dict" in ckpt:
            raw_model.load_state_dict(ckpt["state_dict"])
        else:
            raw_model.load_state_dict(ckpt)
            if is_main():
                print("Checkpoint has no epoch info, resuming from epoch 0")
            del ckpt
            return 0

        # Load optimizer state (skip if missing or incompatible)
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                if is_main():
                    print(f"Warning: Could not load optimizer state ({e}), using fresh optimizer")
        elif is_main():
            print("Warning: No optimizer state in checkpoint, using fresh optimizer")

        optimizer.n_current_steps = ckpt.get("n_current_steps", 0)
        self.best_loss = ckpt.get("best_loss", float("inf"))
        start_epoch = ckpt.get("epoch", -1) + 1
        if is_main():
            print(f"Resumed from {path} | epoch {start_epoch} | step {optimizer.n_current_steps}")
        del ckpt
        return start_epoch

    def save_epoch(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.epoch_losses.append(loss)
            return True
        self.epoch_losses.append(loss)
        return False

    def save_step(self, step, total_steps_per_epoch):
        if step == 0:
            return True
        save_interval = max(1, total_steps_per_epoch // 5)
        return step % save_interval == 0

    def stop_early(self):
        if len(self.epoch_losses) < self.args.patience + 1:
            return False
        best_loss = min(self.epoch_losses[: -self.args.patience])
        current_loss = min(self.epoch_losses[-self.args.patience :])
        return current_loss > best_loss - self.args.patience_delta