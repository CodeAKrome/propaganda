"""
Political Bias Detector using T5 with LoRA - WITH CLI SUPPORT & NON-FATAL ERROR HANDLING
========================================================================================
Now supports skipping malformed data entries and reporting them at the end.
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
from typing import Dict, List, Tuple
import warnings
import argparse
import sys
from pathlib import Path
warnings.filterwarnings('ignore')


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
# PART 2: MODEL SETUP WITH LoRA
# ==============================================================================

class BiasDetectorModel:
    def __init__(self, model_name: str = "t5-small", lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.base_model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q", "v"],
            bias="none"
        )
        self.model = get_peft_model(self.base_model, lora_config)
    
    def get_model(self): return self.model
    def get_tokenizer(self): return self.tokenizer
    
    def save_lora_weights(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def load_lora_model(cls, base_model_name: str, lora_weights_path: str):
        base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, lora_weights_path)
        tokenizer = T5Tokenizer.from_pretrained(lora_weights_path)
        instance = cls(model_name=base_model_name)
        instance.model = model
        instance.tokenizer = tokenizer
        return instance


# ==============================================================================
# PART 3: TRAINING CONFIGURATION
# ==============================================================================

class BiasDetectorTrainer:
    def __init__(self, model, train_dataset, eval_dataset, output_dir="./bias-detector-lora", num_epochs=10, batch_size=2, learning_rate=5e-4):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        
        tokenized_train = self._tokenize(train_dataset)
        tokenized_eval = self._tokenize(eval_dataset)
        
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none",
            fp16=torch.cuda.is_available()
        )
        
        self.trainer = Trainer(
            model=model.get_model(),
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=DataCollatorForSeq2Seq(model.get_tokenizer(), padding=True)
        )
    
    def _tokenize(self, dataset):
        def func(ex):
            inputs = self.model.get_tokenizer()(ex["input_text"], max_length=512, truncation=True)
            labels = self.model.get_tokenizer()(ex["target_text"], max_length=300, truncation=True)
            inputs["labels"] = labels["input_ids"]
            return inputs
        return dataset.map(func, batched=True, remove_columns=dataset.column_names)
    
    def train(self):
        self.trainer.train()
        self.model.save_lora_weights(self.output_dir)
        return self.model


# ==============================================================================
# PART 4: INFERENCE
# ==============================================================================

class BiasPredictor:
    def __init__(self, model):
        self.model = model.get_model()
        self.tokenizer = model.get_tokenizer()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def predict(self, article: str) -> Dict:
        input_text = f"classify political bias as json: {article}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=300)
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            return json.loads(generated_text)
        except:
            return {"error": "JSON Parse Error", "raw": generated_text}


# ==============================================================================
# PART 5: CLI & MAIN
# ==============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Political Bias Detector")
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--model-name', type=str, default='t5-small')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--output-dir', type=str, default='./bias-detector-output')
    parser.add_argument('--predict-only', action='store_true')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--export-sample', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if args.export_sample:
        BiasDatasetCreator().save_sample_file(args.export_sample)
        return

    # STEP 1: PREPARE DATA (Non-fatal collection)
    print("\n" + "="*80)
    print("Political Bias Detector - Pipeline")
    print("="*80)
    
    try:
        creator = BiasDatasetCreator(data_file=args.data)
    except Exception as e:
        print(f"Critical Error: {e}")
        sys.exit(1)
        
    train_ds, eval_ds = creator.train_test_split()
    
    # STEP 2 & 3: MODEL & TRAINING
    if not args.predict_only:
        print("\n[Step 2/3] Initializing and Training...")
        model = BiasDetectorModel(model_name=args.model_name)
        trainer = BiasDetectorTrainer(model, train_ds, eval_ds, output_dir=args.output_dir, num_epochs=args.epochs, batch_size=args.batch_size)
        model = trainer.train()
    else:
        print("\n[Step 2] Loading existing model...")
        if not args.model_path:
            print("Error: --model-path required for --predict-only mode")
            sys.exit(1)
        model = BiasDetectorModel.load_lora_model(args.model_name, args.model_path)

    # STEP 4: PREDICTION
    predictor = BiasPredictor(model)
    test_text = "The legislature passed a routine budget bill today."
    print(f"\nSample Prediction:\n{json.dumps(predictor.predict(test_text), indent=2)}")

    # FINAL REPORT: List all skipped errors at the very end
    if creator.errors:
        print("\n" + "!"*80)
        print(f"DATA INTEGRITY REPORT: {len(creator.errors)} entries skipped during loading")
        print("!"*80)
        for err in creator.errors:
            print(f"  -> {err}")
        print("!"*80 + "\n")

    print(f"Process complete. Model handled by: {args.output_dir if not args.predict_only else args.model_path}")

if __name__ == "__main__":
    main()