#!/usr/bin/env python3
"""
T5 LoRA Training Script for Political Bias Detection
=====================================================
Fine-tunes T5-large with LoRA adapters to output structured JSON.

Training format:
  Input:  "classify political bias as json: {article text}"
  Output: '{"dir": {"L": 0.2, "C": 0.6, "R": 0.2}, "deg": {"L": 0.1, "M": 0.8, "H": 0.1}, "reason": "..."}'

Usage:
  python train_bias_detector.py --data train.json --output-dir bias-detector-output
  python train_bias_detector.py --data train.json --output-dir bias-detector-output --epochs 3 --batch-size 8
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_PREFIX = "classify political bias as json: "
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256


# ==============================================================================
# DATASET
# ==============================================================================

class BiasDataset(Dataset):
    """Dataset for bias detection training."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: T5Tokenizer,
        max_input_length: int = MAX_INPUT_LENGTH,
        max_target_length: int = MAX_TARGET_LENGTH,
        input_prefix: str = INPUT_PREFIX
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to training JSON file
            tokenizer: T5 tokenizer
            max_input_length: Maximum input sequence length
            max_target_length: Maximum target sequence length
            input_prefix: Prefix to add to input text
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.input_prefix = input_prefix
        
        # Load data
        logger.info(f"Loading training data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} training samples")
        
        # Validate data format
        self._validate_data()
    
    def _validate_data(self):
        """Validate training data format."""
        required_keys = ['article', 'label']
        label_keys = ['dir', 'deg', 'reason']
        dir_keys = ['L', 'C', 'R']
        deg_keys = ['L', 'M', 'H']
        
        invalid_count = 0
        for i, sample in enumerate(self.data):
            # Check top-level keys
            if not all(k in sample for k in required_keys):
                logger.warning(f"Sample {i}: missing required keys")
                invalid_count += 1
                continue
            
            # Check label structure
            label = sample['label']
            if not all(k in label for k in label_keys):
                logger.warning(f"Sample {i}: missing label keys")
                invalid_count += 1
                continue
            
            # Check dir structure
            if not all(k in label['dir'] for k in dir_keys):
                logger.warning(f"Sample {i}: missing dir keys")
                invalid_count += 1
                continue
            
            # Check deg structure
            if not all(k in label['deg'] for k in deg_keys):
                logger.warning(f"Sample {i}: missing deg keys")
                invalid_count += 1
                continue
        
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        sample = self.data[idx]
        
        # Format input
        article = sample['article']
        input_text = f"{self.input_prefix}{article}"
        
        # Format target as JSON string
        label = sample['label']
        target_text = json.dumps(label, ensure_ascii=False)
        
        # Tokenize input
        input_encodings = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encodings = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels (replace pad token ids with -100 for loss calculation)
        labels = target_encodings['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encodings['input_ids'].squeeze(),
            'attention_mask': input_encodings['attention_mask'].squeeze(),
            'labels': labels
        }


# ==============================================================================
# DATA COLLATOR
# ==============================================================================

class BiasDataCollator:
    """Custom data collator for bias detection."""
    
    def __init__(self, tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def load_model_and_tokenizer(
    model_name: str = "t5-large",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1
) -> tuple:
    """
    Load T5 model with LoRA adapters.
    
    Args:
        model_name: Base T5 model name
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name, verbose=False)
    
    logger.info(f"Loading model: {model_name}")
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        low_cpu_mem_usage=True
    )
    
    # Configure LoRA
    logger.info("Configuring LoRA adapters")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q", "v"],  # Target attention layers
        bias="none"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def train(
    data_path: str,
    output_dir: str,
    model_name: str = "t5-large",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    warmup_steps: int = 500,
    max_input_length: int = MAX_INPUT_LENGTH,
    max_target_length: int = MAX_TARGET_LENGTH,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    device: str = None,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 100,
    resume_from_checkpoint: str = None
):
    """
    Train the bias detection model.
    
    Args:
        data_path: Path to training JSON file
        output_dir: Directory to save model
        model_name: Base T5 model name
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        max_input_length: Maximum input sequence length
        max_target_length: Maximum target sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        device: Device to use (auto-detect if None)
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        logging_steps: Log every N steps
    """
    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    
    # Move model to device
    model = model.to(device)
    
    # Load dataset
    dataset = BiasDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length
    )
    
    # Split into train/eval
    train_size = int(0.95 * len(dataset))
    eval_size = len(dataset) - train_size
    
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Data collator
    data_collator = BiasDataCollator(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=str(output_path / "logs"),
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=(device == "cuda"),
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    logger.info(f"Saving model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Save training config
    config = {
        "model_name": model_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_input_length": max_input_length,
        "max_target_length": max_target_length,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "device": device
    }
    
    with open(output_path / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Training complete!")
    
    return model, tokenizer


def test_model(model, tokenizer, test_texts: List[str], device: str = "cpu"):
    """
    Test the trained model on sample texts.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_texts: List of test texts
        device: Device to use
    """
    logger.info("\n" + "="*60)
    logger.info("TESTING MODEL")
    logger.info("="*60)
    
    model.eval()
    
    for text in test_texts:
        # Format input
        input_text = f"{INPUT_PREFIX}{text}"
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"\nInput: {text[:100]}...")
        logger.info(f"Output: {output_text}")
        
        # Try to parse as JSON
        try:
            parsed = json.loads(output_text)
            logger.info(f"Parsed: {json.dumps(parsed, indent=2)}")
        except json.JSONDecodeError:
            logger.warning("Output is not valid JSON")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train T5 model for political bias detection"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training JSON file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bias-detector-output",
        help="Directory to save model (default: bias-detector-output)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="t5-large",
        help="Base T5 model name (default: t5-large)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)"
    )
    
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test after training"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint directory"
    )
    
    args = parser.parse_args()
    
    # Train
    model, tokenizer = train(
        data_path=args.data,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        device=args.device,
        resume_from_checkpoint=args.resume
    )
    
    # Test
    if args.test:
        test_texts = [
            "The president announced new policies today.",
            "The filthy dictator waged war on innocent civilians.",
            "Congress passed a bipartisan bill with overwhelming support."
        ]
        
        device = args.device or ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
        test_model(model, tokenizer, test_texts, device)


if __name__ == "__main__":
    main()
