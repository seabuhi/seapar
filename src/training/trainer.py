import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from sklearn.metrics import f1_score, classification_report
from src.utils.logger import setup_logger

logger = setup_logger("trainer")

class FeedbackTrainer:
    """Trainer class for training and evaluating feedback classifier models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        weight_decay: float,
        warmup_ratio: float,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        label_smoothing: float = 0.0,
        early_stopping_patience: int = 3,
        use_amp: bool = True,
        save_dir: str = "outputs",
        fold: int = 0,
        device: Optional[torch.device] = None
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.label_smoothing = label_smoothing
        self.early_stopping_patience = early_stopping_patience
        self.use_amp = use_amp and torch.cuda.is_available()
        self.save_dir = save_dir
        self.fold = fold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup Optimizer and Scheduler
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_checkpoint_path = os.path.join(self.save_dir, f"best_model_fold_{self.fold}.pt")
        
    def _get_optimizer(self) -> torch.optim.Optimizer:
        # Avoid weight decay on bias and LayerNorm parameters
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        
    def _get_scheduler(self):
        total_steps = len(self.train_loader) * self.epochs // self.grad_accum_steps
        num_warmup_steps = int(total_steps * self.warmup_ratio)
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = self.criterion(logits, labels)
                    # Normalize loss for accumulation
                    loss = loss / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = self.criterion(logits, labels)
                loss = loss / self.grad_accum_steps
                loss.backward()
                
            total_loss += loss.item() * self.grad_accum_steps
            
            # Optimizer step
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                self.scheduler.step()
                self.optimizer.zero_grad()
                
        return total_loss / len(self.train_loader)
        
    def evaluate(self) -> Tuple[float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        loss = self.criterion(logits, labels)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = self.criterion(logits, labels)
                    
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                
        val_loss = total_loss / len(self.val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_logits = np.concatenate(all_logits, axis=0)
        
        # Compute Macro F1
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        
        return val_loss, macro_f1, all_labels, all_logits
        
    def fit(self) -> Dict[str, Any]:
        """Runs the training and validation loop."""
        logger.info(f"Starting training on device: {self.device} (AMP={self.use_amp})")
        
        best_macro_f1 = -1.0
        epochs_no_improve = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_macro_f1": []
        }
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_macro_f1, _, _ = self.evaluate()
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_macro_f1"].append(val_macro_f1)
            
            logger.info(
                f"Fold {self.fold} | Epoch {epoch:02d}/{self.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Macro F1: {val_macro_f1:.4f}"
            )
            
            # Save best checkpoint
            if val_macro_f1 > best_macro_f1:
                best_macro_f1 = val_macro_f1
                epochs_no_improve = 0
                logger.info(f"New best validation Macro F1: {best_macro_f1:.4f}. Saving state dict...")
                import io
                buffer = io.BytesIO()
                torch.save(self.model.state_dict(), buffer)
                with open(self.best_checkpoint_path, "wb") as f:
                    f.write(buffer.getvalue())
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break
                    
        # Load best weights back
        logger.info(f"Training completed. Loading best checkpoint: {self.best_checkpoint_path}")
        self.model.load_state_dict(torch.load(self.best_checkpoint_path))
        
        return history
