import os
import argparse
import yaml
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.validation.evaluator import CrossValidator

logger = setup_logger("train_entry")

def main():
    parser = argparse.ArgumentParser(description="Train Multilingual Feedback Classifier")
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
        
    logger.info(f"Loaded config: {config}")
    
    # 2. Set Seed
    seed = config.get("seed", 42)
    set_seed(seed)
    logger.info(f"Reproducibility seed set to: {seed}")
    
    # 3. Run cross-validation
    validator = CrossValidator(config)
    overall_f1, oof_df = validator.run_cv()
    
    logger.info(f"Cross Validation completed. Overall OOF Macro F1: {overall_f1:.4f}")

if __name__ == "__main__":
    main()
