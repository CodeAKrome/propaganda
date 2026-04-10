#!/usr/bin/env python3

"""
LoRA Fine-tuning Script for Local Models
=========================================
Fine-tunes Llama, Qwen, GLM, T5 with LoRA adapters.

Supports:
- Llama 2/3 (via transformers + peft)
- Qwen 2.5
- GLM-4
- T5
- Any HuggingFace causal LM

Usage:
    python train_lora.py --data train.json --model meta-llama/Llama-3.2-1B --output lora-output
    python train_lora.py --data train.json --model Qwen/Qwen2-1.5B --epochs 3 --batch-size 4
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import pymongo


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "rssnews")
MONGO_TELEMETRY_COLL = "lora_training_telemetry"


def get_telemetry_collection():
    client = pymongo.MongoClient(MONGO_URI)
    return client[MONGO_DB][MONGO_TELEMETRY_COLL]


class MongoDBTelemetryCallback(TrainerCallback):
    def __init__(self, run_id: str, coll):
        self.run_id = run_id
        self.coll = coll
        self.start_time = datetime.utcnow()
        logger.info(f"[TELEMETRY] Initialized for run_id: {run_id}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        try:
            entry = {
                "run_id": self.run_id,
                "timestamp": datetime.utcnow(),
                "step": state.global_step,
                "epoch": state.epoch,
                "metrics": dict(logs),
            }
            self.coll.insert_one(entry)
        except Exception as e:
            logger.warning(f"[TELEMETRY] Log failed: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        try:
            entry = {
                "run_id": self.run_id,
                "timestamp": datetime.utcnow(),
                "event": "training_complete",
                "total_steps": state.global_step,
                "duration_seconds": (
                    datetime.utcnow() - self.start_time
                ).total_seconds(),
            }
            self.coll.insert_one(entry)
        except Exception as e:
            logger.warning(f"[TELEMETRY] End failed: {e}")


@dataclass
class ModelConfig:
    name: str
    task_type: str
    model_class: Any
    tokenizer_class: Any
    is_seq2seq: bool = False


MODEL_CONFIGS = {
    "llama": ModelConfig(
        name="Llama",
        task_type=TaskType.CAUSAL_LM,
        model_class=AutoModelForCausalLM,
        tokenizer_class=AutoTokenizer,
        is_seq2seq=False,
    ),
    "qwen": ModelConfig(
        name="Qwen",
        task_type=TaskType.CAUSAL_LM,
        model_class=AutoModelForCausalLM,
        tokenizer_class=AutoTokenizer,
        is_seq2seq=False,
    ),
    "glm": ModelConfig(
        name="GLM",
        task_type=TaskType.CAUSAL_LM,
        model_class=AutoModelForCausalLM,
        tokenizer_class=AutoTokenizer,
        is_seq2seq=False,
    ),
    "t5": ModelConfig(
        name="T5",
        task_type=TaskType.SEQ_CLS,
        model_class=AutoModelForSeq2SeqLM,
        tokenizer_class=AutoTokenizer,
        is_seq2seq=True,
    ),
    "default": ModelConfig(
        name="Default",
        task_type=TaskType.CAUSAL_LM,
        model_class=AutoModelForCausalLM,
        tokenizer_class=AutoTokenizer,
        is_seq2seq=False,
    ),
}


class LoRADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        is_seq2seq: bool = False,
    ):
        with open(data_path, "r") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_seq2seq = is_seq2seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.is_seq2seq:
            input_text = item.get("input", item.get("text", ""))
            output_text = item.get("output", "")
        else:
            if "messages" in item:
                prompt = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in item["messages"]]
                )
                input_text = prompt
                output_text = (
                    item["messages"][-1]["content"] if item["messages"] else ""
                )
            elif "instruction" in item:
                input_text = (
                    f"Instruction: {item['instruction']}\nInput: {item['input']}"
                )
                output_text = item.get("output", "")
            else:
                input_text = item.get("text", "")
                output_text = json.dumps(item.get("bias", {})) if "bias" in item else ""

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        if self.is_seq2seq:
            output_enc = self.tokenizer(
                output_text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": input_enc["input_ids"].squeeze(),
                "attention_mask": input_enc["attention_mask"].squeeze(),
                "labels": output_enc["input_ids"].squeeze(),
            }
        else:
            full_text = input_text + "\n" + output_text
            full_enc = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": full_enc["input_ids"].squeeze(),
                "attention_mask": full_enc["attention_mask"].squeeze(),
                "labels": full_enc["input_ids"].squeeze(),
            }


def get_model_and_tokenizer(model_name: str, model_type: str, use_mps: bool = True):
    device = (
        "mps"
        if use_mps and torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["default"])

    tokenizer = config.tokenizer_class.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = config.model_class.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
    )

    return model, tokenizer, config, device


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=task_type,
        bias="none",
        inference_mode=False,
    )


def train(
    data_path: str,
    model_name: str,
    output_dir: str,
    model_type: str = "llama",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    max_length: int = 2048,
    warmup_steps: int = 100,
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_mps: bool = True,
    save_steps: int = 100,
    logging_steps: int = 25,
    eval_steps: int = 200,
    save_total_limit: int = 3,
):
    logger.info(f"Loading data from {data_path}")
    logger.info(f"Model: {model_name}, Type: {model_type}")

    model, tokenizer, config, device = get_model_and_tokenizer(
        model_name, model_type, use_mps
    )

    lora_config = create_lora_config(
        r=lora_r, lora_alpha=lora_alpha, task_type=config.task_type
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = LoRADataset(data_path, tokenizer, max_length, config.is_seq2seq)

    data_collator = (
        DataCollatorForSeq2Seq(tokenizer, model=model)
        if config.is_seq2seq
        else DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        bf16=not (device == "cpu"),
        fp16=device == "cuda",
        dataloader_num_workers=0,
        report_to=["none"],
        logging_dir=f"{output_dir}/logs",
        remove_unused_columns=False,
    )

    coll = get_telemetry_collection()
    run_id = (
        f"lora_{model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    callbacks = [MongoDBTelemetryCallback(run_id, coll)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Training complete!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Fine-tune local models with LoRA")

    parser.add_argument("--data", required=True, help="Training data JSON file")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--model-type",
        default="llama",
        choices=["llama", "qwen", "glm", "t5", "default"],
        help="Model architecture",
    )

    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max-length", type=int, default=2048, help="Max sequence length"
    )
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")

    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")

    parser.add_argument(
        "--no-mps", action="store_true", help="Disable MPS acceleration"
    )
    parser.add_argument(
        "--save-steps", type=int, default=100, help="Save every N steps"
    )
    parser.add_argument(
        "--logging-steps", type=int, default=25, help="Log every N steps"
    )
    parser.add_argument(
        "--eval-steps", type=int, default=200, help="Eval every N steps"
    )

    args = parser.parse_args()

    train(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_mps=not args.no_mps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
    )


if __name__ == "__main__":
    main()
