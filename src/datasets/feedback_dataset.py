import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
from transformers import PreTrainedTokenizer
from src.preprocessing.text_cleaner import TextCleaner

# Define class label mappings
LABEL_MAP = {
    "technical_support": 0,
    "customer_support": 1,
    "other": 2
}

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

class FeedbackDataset(Dataset):
    """PyTorch Dataset for feedback classification.
    
    Processes the raw feedback and tag, tokenizes the combined text,
    and returns PyTorch tensors.
    """
    
    def __init__(
        self,
        feedbacks: List[str],
        tags: List[Optional[str]],
        labels: Optional[List[str]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 64,
        cleaner: Optional[TextCleaner] = None,
        prompt_template: str = "tag: {tag} | feedback: {feedback}"
    ) -> None:
        """
        Args:
            feedbacks: List of feedback strings.
            tags: List of tag strings (may contain None/NaN).
            labels: List of label strings (None for inference).
            tokenizer: PreTrainedTokenizer from transformers.
            max_length: Maximum token length.
            cleaner: TextCleaner instance.
            prompt_template: Format string to combine tag and feedback.
        """
        self.feedbacks = feedbacks
        self.tags = tags
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cleaner = cleaner or TextCleaner()
        self.prompt_template = prompt_template
        
        # Prepare inputs
        self.inputs = self._prepare_inputs()
        
    def _prepare_inputs(self) -> List[str]:
        """Cleans, formats, and combines tag and feedback for each sample."""
        combined_texts = []
        for fb, tag in zip(self.feedbacks, self.tags):
            # 1. Clean the feedback
            fb_clean = self.cleaner.clean(str(fb))
            
            # 2. Clean the tag (handle missing tags)
            if not isinstance(tag, str) or str(tag).lower() == "nan" or not tag.strip():
                tag_clean = "[NO_TAG]"
            else:
                tag_clean = tag.strip()
                
            # 3. Combine using prompt template
            combined = self.prompt_template.format(tag=tag_clean, feedback=fb_clean)
            combined_texts.append(combined)
            
        return combined_texts
        
    def __len__(self) -> int:
        return len(self.inputs)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.inputs[idx]
        
        # Tokenize if tokenizer is provided (otherwise return raw text, useful for profiling)
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            item = {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0)
            }
        else:
            item = {"text": text}
            
        if self.labels is not None:
            label_str = self.labels[idx]
            label_id = LABEL_MAP[label_str]
            item["labels"] = torch.tensor(label_id, dtype=torch.long)
            
        return item
