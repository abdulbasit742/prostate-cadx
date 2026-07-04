import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import time
from lib.logging_setup import logger
from lib.config import config
from lib.db import db
from lib.gpu import gpu_monitor

class Trainer:
    def __init__(self, model, train_loader, val_loader, class_weights=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(config.get("model.checkpoint_dir", "storage/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Enable CUDNN Benchmark
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = config.get("train.cudnn_benchmark", True)
            # Memory layout optimization
            if config.get("train.channels_last", True):
                self.model = self.model.to(memory_format=torch.channels_last)

        # Loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer & Scheduler
        self.lr = config.get("train.lr", 0.0002)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=config.get("train.weight_decay", 0.01))
        
        epochs = config.get("train.epochs", 10)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # AMP Scaler
        self.use_amp = config.get("train.amp", True) and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        self.best_kappa = -1.0
        self.early_stopping_patience = config.get("train.early_stopping_patience", 3)
        self.patience_counter = 0

    def calculate_qwk(self, preds, targets):
        """
        Pure PyTorch / NumPy implementation of Quadratic Weighted Kappa.
        """
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(preds, targets, weights="quadratic")

    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Check for channels last memory layout
                if config.get("train.channels_last", True):
                    images = images.to(memory_format=torch.channels_last)
                    
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0] # Get slide logits if aggregated
                    loss = self.criterion(outputs, targets)
                    
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())
                
        val_loss /= len(self.val_loader.dataset)
        kappa = self.calculate_qwk(all_preds, all_targets)
        return val_loss, kappa

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Check for channels last memory layout
            if config.get("train.channels_last", True):
                images = images.to(memory_format=torch.channels_last)
                
            self.optimizer.zero_grad()
            
            try:
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0] # Get slide logits if aggregated
                    loss = self.criterion(outputs, targets)
                    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                train_loss += loss.item() * images.size(0)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("CUDA OOM detected! Halving batch size and requesting retry.")
                    raise e
                else:
                    raise e
                    
        train_loss /= len(self.train_loader.dataset)
        return train_loss

    def fit(self):
        gpu_monitor.start()
        epochs = config.get("train.epochs", 10)
        logger.info(f"Starting training for {epochs} epochs on {self.device}...")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            try:
                train_loss = self.train_epoch(epoch)
            except RuntimeError as e:
                # If OOM, raise it to allow autotune batch handler to restart
                gpu_monitor.stop()
                raise e
                
            val_loss, kappa = self.evaluate()
            self.scheduler.step()
            
            elapsed = time.time() - start_time
            logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Kappa: {kappa:.4f} | Time: {elapsed:.1f}s")
            
            # Log metrics to DB
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "kappa": kappa,
                "loss": val_loss
            }, checkpoint_path)
            
            db.add_metrics(
                kappa=float(kappa),
                val_loss=float(val_loss),
                batch_size=int(self.train_loader.batch_size),
                epoch=epoch,
                checkpoint_path=str(checkpoint_path)
            )
            
            # Save best checkpoint
            if kappa > self.best_kappa:
                self.best_kappa = kappa
                self.patience_counter = 0
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(self.model.state_dict(), best_path)
                logger.info(f"New best checkpoint saved: Kappa={kappa:.4f}")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break
                
        gpu_monitor.stop()
        logger.info("Training complete.")
        return self.best_kappa
