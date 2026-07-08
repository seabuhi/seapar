import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoConfig

class FeedbackClassifier(nn.Module):
    """Transformer classifier module for multilingual feedback.
    
    Wraps standard AutoModelForSequenceClassification, making model selection
    configurable and supporting custom initialization or layers.
    """
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 3,
        dropout_prob: float = 0.1
    ) -> None:
        """
        Args:
            model_name: Pretrained transformer model name/path.
            num_classes: Number of target categories.
            dropout_prob: Dropout probability for hidden layers.
        """
        super().__init__()
        
        # Load configuration
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        
        # Customize dropout in configuration if applicable
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = dropout_prob
        if hasattr(config, "attention_probs_dropout_prob"):
            config.attention_probs_dropout_prob = dropout_prob
            
        # Load the pretrained classification model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Any:
        """Forward pass of the model.
        
        Args:
            input_ids: Token indices tensor.
            attention_mask: Attention mask tensor.
            labels: Ground truth labels (optional).
            
        Returns:
            SequenceClassifierOutput containing loss (if labels provided) and logits.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
