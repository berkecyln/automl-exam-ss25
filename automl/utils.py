# src/utils.py
import torch
import logging
import sys
from pathlib import Path
from torch.utils.data import Dataset

try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def setup_logging(level=logging.INFO, log_file=None):
    """Setup logging configuration.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional log file path
    """
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


class SimpleTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if isinstance(self.tokenizer, dict):
            tokens = text.split()
            ids = [self.tokenizer.get(tok, 1) for tok in tokens[: self.max_length]]
            ids += [0] * (self.max_length - len(ids))
            return torch.tensor(ids), torch.tensor(label)

        elif TRANSFORMERS_AVAILABLE and hasattr(self.tokenizer, "encode_plus"):
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
                "labels": torch.tensor(label),
            }
        else:
            raise ValueError("Tokenizer not defined or unsupported.")
