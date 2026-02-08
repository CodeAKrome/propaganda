# Political Bias Detector - Mac Silicon (MPS) Optimized

## 🚀 Key Optimizations for Apple Silicon

This version has been specifically optimized for Mac computers with Apple Silicon (M1, M2, M3, M4) chips using Metal Performance Shaders (MPS).

### Performance Improvements

1. **MPS GPU Acceleration**
   - Automatic detection and utilization of Apple Silicon GPU
   - Up to 3-5x faster training compared to CPU
   - Efficient memory management for on-device GPU

2. **Precision Optimizations**
   - Uses `float32` precision (optimal for MPS)
   - Avoids `fp16`/`bf16` which are not supported on MPS
   - Maintains numerical stability

3. **DataLoader Optimizations**
   - Sets `num_workers=0` for MPS (prevents overhead)
   - Disables pin memory (not needed for MPS)
   - Optimized batching for Metal architecture

4. **Memory Efficiency**
   - Proper gradient accumulation support
   - Efficient tokenization without unnecessary padding
   - Smart batch size defaults for M-series chips

5. **Training Stability**
   - Warmup steps for stable convergence
   - Gradient clipping to prevent explosions
   - Eval accumulation for memory efficiency

## 📋 Requirements

```bash
# Install PyTorch with MPS support (required)
pip install torch torchvision torchaudio

# Install required packages
pip install transformers datasets peft accelerate
pip install numpy

# Optional: For better progress tracking
pip install tqdm
```

### System Requirements
- **OS**: macOS 12.3 or later
- **Hardware**: Mac with Apple Silicon (M1/M2/M3/M4)
- **Python**: 3.8 or later
- **Memory**: 8GB+ RAM recommended (16GB for larger models)

## 🎯 Quick Start

### 1. Basic Training (Built-in Data)
```bash
python bias_detector_mps_optimized.py
```

### 2. Train with Your Data
```bash
python bias_detector_mps_optimized.py --data my_training_data.json --epochs 15
```

### 3. Use Larger Model (T5-Base)
```bash
# Note: Reduce batch size for larger models
python bias_detector_mps_optimized.py --model-name t5-base --batch-size 1
```

### 4. Prediction Only Mode
```bash
python bias_detector_mps_optimized.py --predict-only --model-path ./bias-detector-output
```

### 5. Export Sample Data Template
```bash
python bias_detector_mps_optimized.py --export-sample template.json
```

## 📊 Data Format

Your training data should be a JSON array with this structure:

```json
[
  {
    "article": "Your article text here...",
    "label": {
      "dir": {
        "L": 0.1,
        "C": 0.6,
        "R": 0.3
      },
      "deg": {
        "L": 0.7,
        "M": 0.2,
        "H": 0.1
      },
      "reason": "Explanation for the bias classification"
    }
  }
]
```

**Field Descriptions:**
- `article`: The text to classify
- `dir`: Direction probabilities (L=Left, C=Center, R=Right, must sum to ~1.0)
- `deg`: Degree probabilities (L=Low, M=Medium, H=High, must sum to ~1.0)
- `reason`: Human-readable explanation

## 🔧 Command Line Arguments

### Data Arguments
- `--data PATH`: Path to JSON training data file
- `--export-sample PATH`: Export sample data template

### Model Arguments
- `--model-name {t5-small,t5-base,t5-large}`: T5 model variant (default: t5-small)
- `--lora-r INT`: LoRA rank parameter (default: 16)
- `--lora-alpha INT`: LoRA alpha parameter (default: 32)

### Training Arguments
- `--epochs INT`: Number of training epochs (default: 10)
- `--batch-size INT`: Training batch size (default: 2)
- `--learning-rate FLOAT`: Learning rate (default: 5e-4)
- `--gradient-accumulation INT`: Gradient accumulation steps (default: 1)
- `--output-dir PATH`: Output directory (default: ./bias-detector-output)

### Inference Arguments
- `--predict-only`: Skip training, only run prediction
- `--model-path PATH`: Path to trained model (required with --predict-only)
- `--test-article TEXT`: Article text to classify

### Device Arguments
- `--cpu-only`: Force CPU usage (disable MPS)

## 🎨 Usage Examples

### Example 1: Training with Custom Settings
```bash
python bias_detector_mps_optimized.py \
  --data political_articles.json \
  --epochs 20 \
  --batch-size 4 \
  --learning-rate 3e-4 \
  --output-dir ./my_bias_model
```

### Example 2: Memory-Constrained Training
```bash
# Use gradient accumulation to simulate larger batches
python bias_detector_mps_optimized.py \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --model-name t5-base
```

### Example 3: Testing a Trained Model
```bash
python bias_detector_mps_optimized.py \
  --predict-only \
  --model-path ./my_bias_model \
  --test-article "Breaking news: Congress passes landmark legislation"
```

### Example 4: Generate Training Template
```bash
# Create a template to fill in with your data
python bias_detector_mps_optimized.py --export-sample my_template.json
```

## 📈 Performance Tips

### For M1/M2/M3 Base Models (8GB RAM)
```bash
# Conservative settings
python bias_detector_mps_optimized.py \
  --model-name t5-small \
  --batch-size 2 \
  --gradient-accumulation 2
```

### For M1/M2/M3 Pro/Max (16GB+ RAM)
```bash
# More aggressive settings
python bias_detector_mps_optimized.py \
  --model-name t5-base \
  --batch-size 4 \
  --epochs 20
```

### For M3 Max/Ultra (32GB+ RAM)
```bash
# Maximum performance
python bias_detector_mps_optimized.py \
  --model-name t5-large \
  --batch-size 8 \
  --gradient-accumulation 2
```

## 🔍 What Changed from Original?

### Major Optimizations

1. **Device Detection**
   - New `get_optimal_device()` function
   - Automatic MPS detection and fallback
   - Device info saved with model

2. **Training Configuration**
   - Disabled fp16/bf16 (not MPS-compatible)
   - Set optimal DataLoader workers (0 for MPS)
   - Disabled pin_memory for MPS
   - Added warmup steps for stability

3. **Model Loading**
   - Explicit `torch_dtype=torch.float32`
   - Device-aware model initialization
   - Better error handling for device issues

4. **Inference**
   - Simplified generation (greedy decoding for speed)
   - Batch prediction support
   - Better error messages

5. **User Experience**
   - Progress indicators
   - Device information display
   - Enhanced CLI with examples
   - Better error reporting

## 🐛 Troubleshooting

### "MPS backend out of memory"
- Reduce `--batch-size` to 1
- Use gradient accumulation instead
- Close other applications
- Try `t5-small` instead of larger models

### "MPS is not available"
- Check macOS version (requires 12.3+)
- Verify Apple Silicon chip
- Update PyTorch: `pip install --upgrade torch`
- Use `--cpu-only` flag as fallback

### Slow Training
- Ensure MPS is detected (check console output)
- Close background applications
- Use Activity Monitor to check GPU usage
- Verify not running on CPU

### JSON Parse Errors in Predictions
- Model may need more training epochs
- Try larger model variant
- Check if training data is diverse enough
- Increase `max_length` in predictor

## 📊 Expected Performance

### Training Speed (M1 Pro, 16GB)
- **t5-small**: ~5-10 seconds/epoch (small dataset)
- **t5-base**: ~15-30 seconds/epoch (small dataset)
- **t5-large**: ~45-90 seconds/epoch (small dataset)

### Inference Speed
- ~0.1-0.5 seconds per article (depending on length and model size)

## 🔐 Model Architecture

- **Base Model**: T5 (Text-to-Text Transfer Transformer)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Task Type**: Sequence-to-Sequence Generation
- **Output Format**: Structured JSON with bias scores

## 📝 Output Format

```json
{
  "dir": {
    "L": 0.15,
    "C": 0.60,
    "R": 0.25
  },
  "deg": {
    "L": 0.70,
    "M": 0.25,
    "H": 0.05
  },
  "reason": "The article uses neutral language and presents multiple perspectives"
}
```

## 📚 Additional Resources

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Hugging Face PEFT (LoRA)](https://huggingface.co/docs/peft/index)
- [T5 Model Documentation](https://huggingface.co/docs/transformers/model_doc/t5)

## 🤝 Contributing

This is an optimized version specifically for Mac Silicon. For improvements:
1. Test on your Apple Silicon Mac
2. Report performance metrics
3. Suggest optimizations
4. Share training results

## 📄 License

Same as original implementation. Use responsibly and ethically.

## ⚠️ Ethical Considerations

This tool is for educational and research purposes. Political bias detection is:
- Subjective and context-dependent
- Influenced by training data biases
- Not a replacement for critical thinking
- Should be used responsibly

Always validate results and consider multiple perspectives.
