import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import json

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.preprocessing.text_cleaner import TextCleaner
from src.datasets.feedback_dataset import FeedbackDataset, LABEL_MAP, INV_LABEL_MAP
from src.models.classifier import FeedbackClassifier
from src.training.trainer import FeedbackTrainer

logger = setup_logger("evaluator")

class CrossValidator:
    """Manages Stratified K-Fold cross validation training and evaluation."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.seed = config.get("seed", 42)
        
        # Paths
        self.train_path = config["data"]["train_path"]
        self.save_dir = config["training"]["save_dir"]
        
        # Model hyperparameters
        self.model_name = config["model"]["model_name"]
        self.max_length = config["model"]["max_length"]
        
        # Training hyperparameters
        self.epochs = config["training"]["epochs"]
        self.batch_size = config["training"]["batch_size"]
        self.lr = config["training"]["lr"]
        self.weight_decay = config["training"]["weight_decay"]
        self.warmup_ratio = config["training"]["warmup_ratio"]
        self.grad_accum_steps = config["training"]["grad_accum_steps"]
        self.max_grad_norm = config["training"]["max_grad_norm"]
        self.label_smoothing = config["training"]["label_smoothing"]
        self.early_stopping_patience = config["training"]["early_stopping_patience"]
        self.use_amp = config["training"]["use_amp"]
        
        # Validation hyperparameters
        self.num_folds = config["validation"]["num_folds"]
        self.stratified = config["validation"]["stratified"]
        
        os.makedirs(self.save_dir, exist_ok=True)
        
    def run_cv(self) -> Tuple[float, pd.DataFrame]:
        """Runs the complete cross-validation pipeline.
        
        Returns:
            Tuple containing overall OOF Macro F1 and the OOF predictions DataFrame.
        """
        logger.info(f"Loading training data from {self.train_path}...")
        df = pd.read_csv(self.train_path)
        
        # Convert tags to strings and fill NAs with NaN to pass to Dataset
        feedbacks = df["feedback"].tolist()
        tags = df["tag"].tolist()
        labels = df["label"].tolist()
        
        # Check target distribution
        logger.info(f"Target distribution:\n{df['label'].value_counts(normalize=True)}")
        
        # Initialize KFold split
        set_seed(self.seed)
        if self.stratified:
            kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            splits = list(kf.split(df, labels))
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            splits = list(kf.split(df))
            
        # Placeholders for OOF logits
        oof_logits = np.zeros((len(df), len(LABEL_MAP)))
        oof_preds = np.zeros(len(df))
        
        fold_f1_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"\n================ FOLD {fold} ================")
            
            # Setup Fold Data
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
            # Initialize tokenizer for each fold
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Datasets
            train_dataset = FeedbackDataset(
                feedbacks=train_df["feedback"].tolist(),
                tags=train_df["tag"].tolist(),
                labels=train_df["label"].tolist(),
                tokenizer=tokenizer,
                max_length=self.max_length
            )
            
            val_dataset = FeedbackDataset(
                feedbacks=val_df["feedback"].tolist(),
                tags=val_df["tag"].tolist(),
                labels=val_df["label"].tolist(),
                tokenizer=tokenizer,
                max_length=self.max_length
            )
            
            # Dataloaders
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Initialize model
            model = FeedbackClassifier(
                model_name=self.model_name,
                num_classes=len(LABEL_MAP),
                dropout_prob=0.1
            )
            
            # Initialize Trainer
            trainer = FeedbackTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.epochs,
                lr=self.lr,
                weight_decay=self.weight_decay,
                warmup_ratio=self.warmup_ratio,
                grad_accum_steps=self.grad_accum_steps,
                max_grad_norm=self.max_grad_norm,
                label_smoothing=self.label_smoothing,
                early_stopping_patience=self.early_stopping_patience,
                use_amp=self.use_amp,
                save_dir=self.save_dir,
                fold=fold
            )
            
            # Train
            history = trainer.fit()
            
            # Evaluate on validation set using best weights
            val_loss, fold_macro_f1, val_labels, val_logits = trainer.evaluate()
            fold_f1_scores.append(fold_macro_f1)
            
            logger.info(f"Fold {fold} finished. Validation Macro F1: {fold_macro_f1:.4f}")
            
            # Store OOF logits
            oof_logits[val_idx] = val_logits
            oof_preds[val_idx] = np.argmax(val_logits, axis=1)
            
            # Save fold statistics
            with open(os.path.join(self.save_dir, f"fold_{fold}_history.json"), "w") as f:
                json.dump(history, f, indent=4)
                
            # Free memory
            del model
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Overall OOF metrics
        y_true = np.array([LABEL_MAP[lbl] for lbl in labels])
        overall_macro_f1 = f1_score(y_true, oof_preds, average="macro")
        
        logger.info(f"\n================ OVERALL OOF PERFORMANCE ================")
        logger.info(f"F1 macro scores across folds: {[round(s, 4) for s in fold_f1_scores]}")
        logger.info(f"Mean F1 score: {np.mean(fold_f1_scores):.4f} (std: {np.std(fold_f1_scores):.4f})")
        logger.info(f"Overall OOF Macro F1 Score: {overall_macro_f1:.4f}")
        
        # Save OOF details
        df_oof = df.copy()
        for i, class_name in INV_LABEL_MAP.items():
            df_oof[f"logit_{class_name}"] = oof_logits[:, i]
        df_oof["pred_label"] = [INV_LABEL_MAP[int(p)] for p in oof_preds]
        
        oof_path = os.path.join(self.save_dir, "oof_predictions.csv")
        df_oof.to_csv(oof_path, index=False)
        logger.info(f"OOF predictions saved to {oof_path}")
        
        # Output Classification Report
        report = classification_report(df_oof["label"], df_oof["pred_label"])
        logger.info(f"\nOOF Classification Report:\n{report}")
        
        # Save config along with results for reproducibility
        self.config["results"] = {
            "fold_f1s": fold_f1_scores,
            "mean_f1": float(np.mean(fold_f1_scores)),
            "std_f1": float(np.std(fold_f1_scores)),
            "overall_oof_macro_f1": float(overall_macro_f1)
        }
        with open(os.path.join(self.save_dir, "run_results.json"), "w") as f:
            json.dump(self.config, f, indent=4)
            
        return overall_macro_f1, df_oof
