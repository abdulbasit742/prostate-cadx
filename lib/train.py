import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import time
import torch.nn.functional as F
from lib.logging_setup import logger
from lib.config import config
from lib.db import db
from lib.gpu import gpu_monitor

def info_nce_loss(features_a, features_b, temperature=0.1):
    """
    Computes InfoNCE loss for a batch of positive pairs (features_a, features_b).
    """
    features_a = F.normalize(features_a, dim=1)
    features_b = F.normalize(features_b, dim=1)
    
    batch_size = features_a.size(0)
    device = features_a.device
    
    # Cosine similarities
    sim_a_b = torch.matmul(features_a, features_b.T) / temperature
    sim_a_a = torch.matmul(features_a, features_a.T) / temperature
    sim_b_b = torch.matmul(features_b, features_b.T) / temperature
    
    # Mask out self-similarities
    mask = torch.eye(batch_size, device=device).bool()
    sim_a_a = sim_a_a.masked_fill(mask, -1e4)
    sim_b_b = sim_b_b.masked_fill(mask, -1e4)
    
    targets = torch.arange(batch_size, device=device)
    
    logits_a_b = torch.cat([sim_a_b, sim_a_a], dim=1)
    logits_b_a = torch.cat([sim_a_b.T, sim_b_b], dim=1)
    
    loss_a_b = F.cross_entropy(logits_a_b, targets)
    loss_b_a = F.cross_entropy(logits_b_a, targets)
    
    return (loss_a_b + loss_b_a) / 2.0


def supervised_contrastive_loss(proj_a, proj_b, labels, temperature=0.1):
    """
    Computes Supervised Contrastive Loss (SupCon) for a batch of projections and labels.
    """
    # Normalize features
    proj_a = F.normalize(proj_a, dim=1)
    proj_b = F.normalize(proj_b, dim=1)
    
    # Concatenate features and duplicate labels
    features = torch.cat([proj_a, proj_b], dim=0)
    batch_size = proj_a.shape[0]
    labels = torch.cat([labels, labels], dim=0)
    
    # Similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # For numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Tile mask: mask[i, j] = 1 if labels[i] == labels[j]
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(proj_a.device)
    
    # Self-contrast mask: mask out diagonal entries
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(2 * batch_size).view(-1, 1).to(proj_a.device),
        0
    )
    mask = mask * logits_mask
    
    # Compute log probability
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
    
    # Mean log-likelihood over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
    
    loss = -mean_log_prob_pos.mean()
    return loss


class Trainer:
    def __init__(self, model, train_loader, val_loader, class_weights=None, resume_checkpoint=None):
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

        # Contrastive parameters
        self.contrastive_weight = config.get("train.contrastive_weight", 0.5)
        self.contrastive_temperature = config.get("train.contrastive_temperature", 0.1)

        self.best_kappa = -1.0
        self.early_stopping_patience = config.get("train.early_stopping_patience", 3)
        self.patience_counter = 0
        self.start_epoch = 1

        # Resume from checkpoint if provided
        if resume_checkpoint and Path(resume_checkpoint).exists():
            logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
            ckpt = torch.load(resume_checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            self.start_epoch = ckpt.get("epoch", 1) + 1
            self.best_kappa = ckpt.get("kappa", -1.0)
            logger.info(f"Resumed: starting from epoch {self.start_epoch}, best kappa so far: {self.best_kappa:.4f}")

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
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Check if loader dataset yields dual views
            if getattr(self.train_loader.dataset, "return_dual_views", False):
                images_a, images_b, targets = batch
                images_a, images_b, targets = images_a.to(self.device), images_b.to(self.device), targets.to(self.device)
                
                # Check for channels last memory layout
                if config.get("train.channels_last", True):
                    images_a = images_a.to(memory_format=torch.channels_last)
                    images_b = images_b.to(memory_format=torch.channels_last)
                    
                self.optimizer.zero_grad()
                
                try:
                    with autocast(enabled=self.use_amp):
                        # Forward both views with return_contrastive=True
                        outputs_a, proj_a = self.model(images_a, return_contrastive=True)
                        outputs_b, proj_b = self.model(images_b, return_contrastive=True)
                        
                        # Supervised CE loss (average of both views)
                        loss_sup_a = self.criterion(outputs_a, targets)
                        loss_sup_b = self.criterion(outputs_b, targets)
                        loss_sup = (loss_sup_a + loss_sup_b) / 2.0
                        
                        # Contrastive loss
                        loss_contrastive = supervised_contrastive_loss(proj_a, proj_b, targets, temperature=self.contrastive_temperature)
                        
                        # Joint loss
                        loss = loss_sup + self.contrastive_weight * loss_contrastive
                        
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    train_loss += loss.item() * images_a.size(0)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("CUDA OOM detected! Halving batch size and requesting retry.")
                        raise e
                    else:
                        raise e
            else:
                images, targets = batch
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Check for channels last memory layout
                if config.get("train.channels_last", True):
                    images = images.to(memory_format=torch.channels_last)
                    
                self.optimizer.zero_grad()
                
                try:
                    with autocast(enabled=self.use_amp):
                        outputs = self.model(images)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
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
        logger.info(f"Starting training for {epochs} epochs on {self.device} (from epoch {self.start_epoch})...")
        
        for epoch in range(self.start_epoch, epochs + 1):
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
            backbone = config.get("model.backbone", "resnet50")
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{backbone}_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
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
            
            # Sanity Gate after epoch 3
            if epoch == 3:
                import sys
                import pandas as pd
                if kappa <= 0.001:
                    logger.warning(f"SANITY GATE WARNING at epoch 3. Val QWK is {kappa:.4f} (<= 0.0). Continuing training to allow contrastive representations to align.")
                    db.log_event("WARNING", f"Sanity Gate warning at epoch 3: Val QWK is {kappa:.4f} (<= 0.0). Continuing training.")
                    
                    diag_path = Path("logs/sanity_gate_fail.json")
                    train_dist = pd.Series([item["label"] for item in self.train_loader.dataset.data_list]).value_counts().to_dict()
                    val_dist = pd.Series([item["label"] for item in self.val_loader.dataset.data_list]).value_counts().to_dict()
                    
                    import json
                    with open(diag_path, "w") as df_json:
                        json.dump({
                            "epoch": epoch,
                            "val_qwk": kappa,
                            "val_loss": val_loss,
                            "train_loss": train_loss,
                            "train_label_distribution": {str(k): int(v) for k, v in train_dist.items()},
                            "val_label_distribution": {str(k): int(v) for k, v in val_dist.items()},
                            "message": "Sanity gate warning: validation QWK (Kappa) was <= 0.0 after 3 epochs under contrastive learning."
                        }, df_json, indent=4)
                        
                    logger.info(f"Diagnostic dump saved to {diag_path}")
                else:
                    logger.info(f"SANITY GATE PASSED: validation QWK is {kappa:.4f} (> 0). Continuing training.")
                    db.log_event("INFO", f"Sanity Gate passed: validation QWK is {kappa:.4f} (> 0).")
            
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
