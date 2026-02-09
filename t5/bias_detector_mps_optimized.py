#!/usr/bin/env python
"""
Political Bias Detector using T5 with LoRA - OPTIMIZED FOR MAC SILICON (MPS)
==============================================================================
Enhanced with Apple Silicon MPS acceleration and mixed precision training.
"""

import torch
import json
import numpy as np
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from typing import Dict, List, Tuple, Optional
import warnings
import argparse
import sys
from pathlib import Path
warnings.filterwarnings('ignore')


# ==============================================================================
# MPS DEVICE CONFIGURATION
# ==============================================================================

def get_optimal_device() -> torch.device:
    """
    Detect and return the optimal device for Mac Silicon.
    Priority: MPS > CUDA > CPU
    """
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print("✓ MPS (Apple Silicon GPU) detected and enabled")
            return torch.device("mps")
        else:
            print("⚠ MPS available but not built - falling back to CPU")
            return torch.device("cpu")
    elif torch.cuda.is_available():
        print("✓ CUDA GPU detected")
        return torch.device("cuda")
    else:
        print("ℹ Using CPU (no GPU acceleration)")
        return torch.device("cpu")


# ==============================================================================
# PART 1: DATA PREPARATION - WITH NON-FATAL ERROR HANDLING
# ==============================================================================

class BiasDatasetCreator:
    """
    Creates and manages the dataset for political bias detection.
    Supports loading data from JSON files with error logging.
    """
    
    def __init__(self, data_file: str = None):
        """
        Initialize with data from file or built-in samples.
        """
        self.errors = []  # To track malformed entries without crashing
        
        if data_file:
            # Load data from external file
            self.sample_data = self.load_from_file(data_file)
            print(f"Loaded {len(self.sample_data)} valid samples from {data_file}")
        else:
            # Use built-in sample data
            self.sample_data = self._get_builtin_samples()
            print(f"Using {len(self.sample_data)} built-in samples")
    
    def load_from_file(self, filepath: str) -> List[Dict]:
        """
        Load training data from a JSON file. Skips invalid entries.
        """
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON file format in {filepath}: {e.msg}",
                e.doc,
                e.pos
            )
        
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}.")
        
        valid_entries = []
        for i, entry in enumerate(data):
            error_msg = self._check_entry_errors(entry, i)
            if error_msg:
                self.errors.append(error_msg)
                continue  # Skip this entry and keep going
            valid_entries.append(entry)
        
        if not valid_entries:
            raise ValueError("No valid entries found in the provided data file.")
            
        return valid_entries
    
    def _check_entry_errors(self, entry: Dict, index: int) -> str:
        """Validates entry and returns an error string if invalid, else None."""
        try:
            if not isinstance(entry, dict):
                return f"Entry {index}: Expected object, got {type(entry).__name__}"
            
            # Check for article field
            if "article" not in entry:
                return f"Entry {index}: Missing 'article' field"
            if not isinstance(entry["article"], str) or not entry["article"].strip():
                return f"Entry {index}: 'article' must be a non-empty string"
            
            # Check for label field
            if "label" not in entry:
                return f"Entry {index}: Missing 'label' field"
            if not isinstance(entry["label"], dict):
                return f"Entry {index}: 'label' must be an object"
            
            label = entry["label"]
            for field in ["dir", "deg", "reason"]:
                if field not in label:
                    return f"Entry {index}: Missing required label field '{field}'"
            
            return None
        except Exception as e:
            return f"Entry {index}: Unexpected error - {str(e)}"
    
    def _get_builtin_samples(self) -> List[Dict]:
        """Return built-in sample data."""
        return [
            {
                "article": "Former Republican Gov. Chris Christie called Trump's recent rally an absolute disaster.",
                "label": {
                    "dir": {"L": 0.1, "C": 0.2, "R": 0.7},
                    "deg": {"L": 0.3, "M": 0.5, "H": 0.2},
                    "reason": "Negative framing of Republican events suggests right-leaning bias."
                }
            },
            {
                "article": "The Senate voted 52-48 on the infrastructure bill after months of bipartisan negotiations.",
                "label": {
                    "dir": {"L": 0.2, "C": 0.6, "R": 0.2},
                    "deg": {"L": 0.7, "M": 0.25, "H": 0.05},
                    "reason": "Balanced reporting on legislative procedures with neutral language."
                }
            },
            {
                "article": "Progressive activists gathered outside the capitol demanding climate action.",
                "label": {
                    "dir": {"L": 0.6, "C": 0.3, "R": 0.1},
                    "deg": {"L": 0.4, "M": 0.4, "H": 0.2},
                    "reason": "Sympathetic framing of progressive causes suggests left-leaning bias."
                }
            },
            {
                "article": "The Federal Reserve maintained interest rates at current levels during today's meeting.",
                "label": {
                    "dir": {"L": 0.3, "C": 0.5, "R": 0.2},
                    "deg": {"L": 0.8, "M": 0.15, "H": 0.05},
                    "reason": "Neutral economic reporting with minimal political slant."
                }
            }
        ]
    
    def save_sample_file(self, filepath: str):
        """Save the current sample data to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.sample_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.sample_data)} samples to {filepath}")
    
    def format_input(self, article: str) -> str:
        return f"classify political bias as json: {article}"
    
    def format_output(self, label: Dict) -> str:
        return json.dumps(label, ensure_ascii=False, separators=(',', ':'))
    
    def create_dataset(self, data: List[Dict] = None) -> Dataset:
        if data is None:
            data = self.sample_data
        
        inputs = [self.format_input(item["article"]) for item in data]
        outputs = [self.format_output(item["label"]) for item in data]
        
        return Dataset.from_dict({"input_text": inputs, "target_text": outputs})
    
    def train_test_split(self, test_size: float = 0.2) -> Tuple[Dataset, Dataset]:
        dataset = self.create_dataset()
        if len(dataset) < 5:
            test_size = 1 if len(dataset) > 1 else 0
        split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
        return split_dataset["train"], split_dataset["test"]


# ==============================================================================
# PART 2: MODEL SETUP WITH LoRA - MPS OPTIMIZED
# ==============================================================================

class BiasDetectorModel:
    def __init__(
        self, 
        model_name: str = "t5-small", 
        lora_r: int = 16, 
        lora_alpha: int = 32, 
        lora_dropout: float = 0.1,
        device: Optional[torch.device] = None
    ):
        self.model_name = model_name
        self.device = device if device is not None else get_optimal_device()
        
        print(f"Loading tokenizer and base model ({model_name})...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Load base model with optimizations
        self.base_model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32  # MPS works best with float32
        )
        
        # Configure LoRA with MPS-friendly settings
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q", "v"],  # Focus on query and value projections
            bias="none",
            inference_mode=False
        )
        
        print("Applying LoRA configuration...")
        self.model = get_peft_model(self.base_model, lora_config)
        
        # Move model to device
        print(f"Moving model to {self.device}...")
        self.model.to(self.device)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def get_model(self): 
        return self.model
    
    def get_tokenizer(self): 
        return self.tokenizer
    
    def get_device(self):
        return self.device
    
    def save_lora_weights(self, output_dir: str):
        """Save LoRA weights and tokenizer."""
        print(f"Saving model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save device info for later loading
        device_info = {"device_type": str(self.device)}
        with open(Path(output_dir) / "device_info.json", 'w') as f:
            json.dump(device_info, f)

    @classmethod
    def load_lora_model(cls, base_model_name: str, lora_weights_path: str, device: Optional[torch.device] = None):
        """Load a trained LoRA model."""
        if device is None:
            device = get_optimal_device()
        
        print(f"Loading base model: {base_model_name}")
        base_model = T5ForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32
        )
        
        print(f"Loading LoRA weights from: {lora_weights_path}")
        model = PeftModel.from_pretrained(base_model, lora_weights_path)
        model.to(device)
        
        tokenizer = T5Tokenizer.from_pretrained(lora_weights_path)
        
        instance = cls(model_name=base_model_name, device=device)
        instance.model = model
        instance.tokenizer = tokenizer
        return instance


# ==============================================================================
# PART 3: TRAINING CONFIGURATION - MPS OPTIMIZED
# ==============================================================================

class BiasDetectorTrainer:
    def __init__(
        self, 
        model: BiasDetectorModel, 
        train_dataset: Dataset, 
        eval_dataset: Dataset, 
        output_dir: str = "./bias-detector-lora",
        num_epochs: int = 10,
        batch_size: int = 2,
        learning_rate: float = 5e-4,
        gradient_accumulation_steps: int = 1,
        use_mps_optimizations: bool = True
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.device = model.get_device()
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        tokenized_train = self._tokenize(train_dataset)
        tokenized_eval = self._tokenize(eval_dataset)
        
        # Determine if we should use MPS optimizations
        is_mps = self.device.type == "mps"
        
        # MPS-optimized training arguments
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=50,  # Helps with MPS stability
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            gradient_accumulation_steps=gradient_accumulation_steps,
            # MPS-specific optimizations
            fp16=False,  # MPS doesn't support fp16; use default float32
            bf16=False,  # MPS doesn't support bf16
            dataloader_num_workers=0 if is_mps else 2,  # MPS works better with 0 workers
            dataloader_pin_memory=False if is_mps else True,  # Disable for MPS
            use_cpu=False,
            # Memory optimizations
            gradient_checkpointing=False,  # Can enable if OOM occurs
            max_grad_norm=1.0,
            # Evaluation optimizations
            eval_accumulation_steps=1,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            model.get_tokenizer(), 
            padding=True,
            return_tensors="pt"
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=model.get_model(),
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
        )
        
        if is_mps:
            print("\n" + "="*80)
            print("MPS OPTIMIZATION ENABLED")
            print("="*80)
            print("• Using float32 precision (optimal for Apple Silicon)")
            print("• DataLoader workers set to 0 (MPS-optimized)")
            print("• Pin memory disabled (MPS-optimized)")
            print("• Gradient accumulation steps:", gradient_accumulation_steps)
            print("="*80 + "\n")
    
    def _tokenize(self, dataset):
        """Tokenize dataset with proper input/output formatting."""
        def func(ex):
            inputs = self.model.get_tokenizer()(
                ex["input_text"], 
                max_length=512, 
                truncation=True,
                padding=False  # Let data collator handle padding
            )
            labels = self.model.get_tokenizer()(
                ex["target_text"], 
                max_length=300, 
                truncation=True,
                padding=False
            )
            inputs["labels"] = labels["input_ids"]
            return inputs
        
        return dataset.map(
            func, 
            batched=True, 
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
    
    def train(self):
        """Train the model with progress reporting."""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        
        # Train
        train_result = self.trainer.train()
        
        # Print training summary
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Final training loss: {train_result.training_loss:.4f}")
        print("="*80 + "\n")
        
        # Save model
        self.model.save_lora_weights(self.output_dir)
        
        return self.model


# ==============================================================================
# PART 4: INFERENCE - MPS OPTIMIZED
# ==============================================================================

class BiasPredictor:
    def __init__(self, model: BiasDetectorModel, max_length: int = 300):
        self.model = model.get_model()
        self.tokenizer = model.get_tokenizer()
        self.device = model.get_device()
        self.max_length = max_length
        
        # Set model to eval mode
        self.model.eval()
        
        print(f"Predictor initialized on {self.device}")
    
    def predict(self, article: str, return_raw: bool = False, use_beam_search: bool = True) -> Dict:
        """
        Predict political bias for a given article.
        
        Args:
            article: The article text to analyze
            return_raw: If True, include raw model output
            use_beam_search: If True, use beam search (slower, more accurate).
                           If False, use greedy decoding (faster, less accurate)
        
        Returns:
            Dictionary with bias predictions
        """
        input_text = f"classify political bias as json: {article}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            if use_beam_search:
                # Original behavior: beam search (matches original script exactly)
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length
                )
            else:
                # MPS-optimized: greedy decoding (faster)
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=1,
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse JSON
        try:
            result = json.loads(generated_text)
            if return_raw:
                result["_raw_output"] = generated_text
            return result
        except json.JSONDecodeError:
            return {
                "error": "JSON Parse Error",
                "raw": generated_text,
                "article": article[:100] + "..." if len(article) > 100 else article
            }
    
    def predict_batch(self, articles: List[str], use_beam_search: bool = True) -> List[Dict]:
        """Predict bias for multiple articles."""
        return [self.predict(article, use_beam_search=use_beam_search) for article in articles]


# ==============================================================================
# PART 5: CLI & MAIN
# ==============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Political Bias Detector - Optimized for Mac Silicon (MPS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python bias_detector_mps_optimized.py
  
  # Train with custom data
  python bias_detector_mps_optimized.py --data my_data.json --epochs 15
  
  # Use larger model
  python bias_detector_mps_optimized.py --model-name t5-base --batch-size 1
  
  # Predict only (requires trained model)
  python bias_detector_mps_optimized.py --predict-only --model-path ./bias-detector-output
  
  # Export sample data template
  python bias_detector_mps_optimized.py --export-sample sample_data.json
        """
    )
    
    # Data arguments
    parser.add_argument('--data', type=str, default=None,
                        help='Path to JSON training data file')
    parser.add_argument('--export-sample', type=str, default=None,
                        help='Export sample data template to specified file')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, default='t5-small',
                        choices=['t5-small', 't5-base', 't5-large'],
                        help='T5 model variant (default: t5-small)')
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA rank parameter (default: 16)')
    parser.add_argument('--lora-alpha', type=int, default=32,
                        help='LoRA alpha parameter (default: 32)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Training batch size (default: 2)')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4)')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--output-dir', type=str, default='./bias-detector-output',
                        help='Output directory for model (default: ./bias-detector-output)')
    
    # Inference arguments
    parser.add_argument('--predict-only', action='store_true',
                        help='Skip training and only run prediction')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model (required for --predict-only)')
    parser.add_argument('--test-article', type=str, default=None,
                        help='Article text to classify (for testing)')
    parser.add_argument('--fast-predict', action='store_true',
                        help='Use greedy decoding for faster predictions (less accurate)')
    
    # Device arguments
    parser.add_argument('--cpu-only', action='store_true',
                        help='Force CPU usage (disable MPS/CUDA)')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Handle sample export
    if args.export_sample:
        print(f"Exporting sample data template to: {args.export_sample}")
        BiasDatasetCreator().save_sample_file(args.export_sample)
        print("✓ Sample data exported successfully!")
        return
    
    # Determine device
    if args.cpu_only:
        device = torch.device("cpu")
        print("ℹ Forcing CPU usage (MPS/CUDA disabled)")
    else:
        device = get_optimal_device()
    
    # STEP 1: PREPARE DATA
    print("\n" + "="*80)
    print("POLITICAL BIAS DETECTOR - MPS OPTIMIZED")
    print("="*80)
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    try:
        creator = BiasDatasetCreator(data_file=args.data)
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        sys.exit(1)
    
    train_ds, eval_ds = creator.train_test_split()
    print(f"Training samples: {len(train_ds)}")
    print(f"Evaluation samples: {len(eval_ds)}")
    
    # STEP 2 & 3: MODEL & TRAINING
    if not args.predict_only:
        print("\n[Step 2/3] Initializing Model with LoRA...")
        model = BiasDetectorModel(
            model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            device=device
        )
        
        print("\n[Step 3/3] Training...")
        trainer = BiasDetectorTrainer(
            model, 
            train_ds, 
            eval_ds, 
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation
        )
        model = trainer.train()
    else:
        print("\n[Step 2] Loading existing model...")
        if not args.model_path:
            print("❌ Error: --model-path required for --predict-only mode")
            sys.exit(1)
        model = BiasDetectorModel.load_lora_model(
            args.model_name, 
            args.model_path,
            device=device
        )
    
    # STEP 4: PREDICTION
    print("\n[Step 4] Running Predictions...")
    predictor = BiasPredictor(model)
    
    # Determine prediction mode
    use_beam_search = not args.fast_predict
    if args.fast_predict:
        print("ℹ Using fast prediction mode (greedy decoding)")
    else:
        print("ℹ Using beam search (matches original behavior)")
    
    # Test with provided article or default
    test_text = args.test_article if args.test_article else \
                "The legislature passed a routine budget bill today with overwhelming bipartisan support."
    
    print(f"\nTest Article: {test_text}\n")
    prediction = predictor.predict(test_text, return_raw=False, use_beam_search=use_beam_search)
    print("Prediction Result:")
    print(json.dumps(prediction, indent=2))
    
    # Run additional sample predictions
    print("\n" + "-"*80)
    print("Additional Sample Predictions:")
    print("-"*80)
    
    sample_articles = [
        "Conservative lawmakers blocked the new environmental regulations.",
        "Progressive groups celebrated the healthcare expansion.",
        "The central bank adjusted monetary policy based on economic indicators."
    ]
    
    for i, article in enumerate(sample_articles, 1):
        pred = predictor.predict(article, use_beam_search=use_beam_search)
        print(f"\n{i}. {article}")
        print(f"   Result: {json.dumps(pred, indent=6)}")
    
    # FINAL REPORT: Data integrity
    if creator.errors:
        print("\n" + "!"*80)
        print(f"DATA INTEGRITY REPORT: {len(creator.errors)} entries skipped")
        print("!"*80)
        for err in creator.errors:
            print(f"  ⚠ {err}")
        print("!"*80 + "\n")
    
    print("\n" + "="*80)
    print("PROCESS COMPLETE")
    print("="*80)
    print(f"Model location: {args.output_dir if not args.predict_only else args.model_path}")
    print(f"Device used: {device}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
