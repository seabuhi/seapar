import os
import argparse
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Dict, Any, List, Optional
import yaml

from src.preprocessing.text_cleaner import TextCleaner
from src.datasets.feedback_dataset import FeedbackDataset, INV_LABEL_MAP
from src.models.classifier import FeedbackClassifier
from src.utils.logger import setup_logger

logger = setup_logger("predict_entry")

def normalize_text_light(t: Any) -> str:
    """Performs a light text normalization to map near-duplicates."""
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r'[^\w\s]', '', t)
    return "".join(t.split())

def compile_leakage_map(train_path: str) -> Dict[str, str]:
    """Compiles the leakage map of consistent near-duplicates from the train set."""
    if not os.path.exists(train_path):
        logger.warning(f"Train data not found at {train_path}. Leakage map cannot be compiled.")
        return {}
        
    logger.info(f"Compiling near-duplicate leakage map from {train_path}...")
    df = pd.read_csv(train_path)
    df["norm_light"] = df["feedback"].apply(normalize_text_light)
    
    # Group by normalized text and extract unique labels
    norm_groups = df.groupby("norm_light")["label"].unique()
    
    leakage_map = {}
    for norm, labels in norm_groups.items():
        if len(labels) == 1 and norm != "":
            leakage_map[norm] = labels[0]
            
    logger.info(f"Leakage map compiled. Identified {len(leakage_map)} unique consistent patterns.")
    return leakage_map

def main():
    parser = argparse.ArgumentParser(description="Predict using trained Multilingual Feedback Classifiers")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to the configuration file"
    )
    args = parser.parse_args()
    
    # 1. Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    train_path = config["data"]["train_path"]
    test_path = config["data"]["test_path"]
    sample_path = config["data"]["sample_path"]
    submission_path = config["data"]["submission_path"]
    save_dir = config["training"]["save_dir"]
    
    model_name = config["model"]["model_name"]
    max_length = config["model"]["max_length"]
    num_folds = config["validation"]["num_folds"]
    
    # 2. Check if checkpoints exist (use whichever are available)
    import glob
    checkpoints = sorted(glob.glob(os.path.join(save_dir, "best_model_fold_*.pt")))
    if not checkpoints:
        raise FileNotFoundError(f"No model checkpoints found in {save_dir}. Please run train.py first.")
    logger.info(f"Found {len(checkpoints)} checkpoints for ensembling: {checkpoints}")
            
    # 3. Compile Leakage Map
    leakage_map = compile_leakage_map(train_path)
    
    # 4. Load Test Data
    logger.info(f"Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    
    # 5. Intercept Leakage Matches
    test_df["norm_light"] = test_df["feedback"].apply(normalize_text_light)
    test_df["pred_label"] = test_df["norm_light"].map(leakage_map)
    
    # Find records that need neural network prediction
    needs_nn_prediction = test_df["pred_label"].isna()
    nn_count = needs_nn_prediction.sum()
    leakage_count = len(test_df) - nn_count
    
    logger.info(f"Direct Leakage Map hits: {leakage_count}/{len(test_df)} ({leakage_count/len(test_df):.2%})")
    logger.info(f"Neural network predictions needed for: {nn_count} records.")
    
    # 6. Run Neural Network for unmatched records
    if nn_count > 0:
        logger.info("Initializing CPU sequence classification models...")
        device = torch.device("cpu")
        
        # Load models
        models = []
        for i, path in enumerate(checkpoints):
            logger.info(f"Loading fold {i} weights from {path}...")
            model = FeedbackClassifier(model_name=model_name, num_classes=3)
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)
            
        # Initialize Tokenizer and Dataset
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        nn_indices = test_df[needs_nn_prediction].index
        nn_feedbacks = test_df.loc[needs_nn_prediction, "feedback"].tolist()
        nn_tags = test_df.loc[needs_nn_prediction, "tag"].tolist()
        
        nn_dataset = FeedbackDataset(
            feedbacks=nn_feedbacks,
            tags=nn_tags,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        # CPU optimized batch loader (larger batch size is fine because max_length is small)
        nn_loader = DataLoader(nn_dataset, batch_size=64, shuffle=False)
        
        # Compute Soft Voting Probabilities
        logger.info("Running soft voting ensembling on CPU...")
        all_probs = []
        
        softmax = nn.Softmax(dim=1)
        
        with torch.no_grad():
            for batch in nn_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Accumulate predictions across all models
                batch_probs = torch.zeros((input_ids.size(0), 3), device=device)
                
                for model in models:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = softmax(logits)
                    batch_probs += probs
                    
                # Average the probabilities
                batch_probs = batch_probs / len(models)
                all_probs.append(batch_probs.numpy())
                
        all_probs = np.concatenate(all_probs, axis=0)
        nn_preds = np.argmax(all_probs, axis=1)
        
        # Map predictions back to original DataFrame
        for idx, pred_id in zip(nn_indices, nn_preds):
            test_df.at[idx, "pred_label"] = INV_LABEL_MAP[pred_id]
            
    # 7. Generate Submission File
    logger.info("Formatting submission file...")
    submission_df = test_df[["id", "pred_label"]].rename(columns={"pred_label": "label"})
    
    # Verify shape and columns
    sample_df = pd.read_csv(sample_path)
    assert len(submission_df) == len(sample_df), "Submission row count mismatch!"
    assert list(submission_df.columns) == list(sample_df.columns), "Submission column mismatch!"
    assert (submission_df["id"].values == sample_df["id"].values).all(), "Submission ID mismatch!"
    
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission successfully saved to {submission_path}")

if __name__ == "__main__":
    main()
