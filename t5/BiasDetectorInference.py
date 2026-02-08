import sys
import json
import torch
import fire
import logging
import warnings
import os

# Suppress library noise
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel

class BiasDetectorInference:
    """
    Inference class for political bias detection. 
    Outputs strictly valid, unescaped JSON objects.
    """
    def __init__(self, model_path: str = './bias-detector-output', base_model_name: str = "t5-small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = T5Tokenizer.from_pretrained(base_model_name, verbose=False)
        base_model = T5ForConditionalGeneration.from_pretrained(
            base_model_name, 
            low_cpu_mem_usage=True
        )
        
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, query: str) -> None:
        # Handle CLI hyphen/bool input
        if query == "-" or isinstance(query, bool):
            input_text = sys.stdin.read()
        else:
            input_text = str(query)

        if not input_text.strip():
            return

        formatted_input = f"classify political bias as json: {input_text}"
        
        inputs = self.tokenizer(
            formatted_input, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
        
        raw_result = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # --- CLEANING & UNESCAPING LOGIC ---
        try:
            # 1. If the model output is wrapped in literal quotes (e.g. "{"dir"..."}"), 
            # we load it to unescape the string first.
            if raw_result.startswith('"') and raw_result.endswith('"'):
                decoded_string = json.loads(raw_result)
            else:
                decoded_string = raw_result

            # 2. If the string is missing outer braces, add them
            if not decoded_string.startswith('{'):
                decoded_string = "{" + decoded_string + "}"
            
            # 3. Parse as actual JSON object to validate and normalize
            final_json = json.loads(decoded_string)
            
            # 4. Print clean, unescaped JSON to stdout
            sys.stdout.write(json.dumps(final_json) + '\n')
            
        except Exception:
            # Fallback: manually clean common escape errors if json.loads fails
            cleaned = raw_result.replace('\\"', '"').strip('"')
            if not cleaned.startswith('{'):
                cleaned = "{" + cleaned + "}"
            sys.stdout.write(cleaned + '\n')
        
        sys.stdout.flush()

if __name__ == "__main__":
    fire.Fire(BiasDetectorInference)