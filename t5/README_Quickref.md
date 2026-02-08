# Quick Reference - MPS Optimized Bias Detector

## 🚀 Common Commands

### First Time Setup
```bash
# Install dependencies
pip install torch torchvision torchaudio transformers datasets peft accelerate numpy

# Test installation
python bias_detector_mps_optimized.py --help
```

### Training Workflows

#### Start with Defaults
```bash
python bias_detector_mps_optimized.py
```
- Uses built-in sample data
- T5-small model
- 10 epochs
- Batch size 2
- Output: `./bias-detector-output/`

#### Train with Your Data
```bash
python bias_detector_mps_optimized.py --data your_data.json
```

#### Quick Training (Testing)
```bash
python bias_detector_mps_optimized.py --epochs 3 --batch-size 4
```

#### Production Training
```bash
python bias_detector_mps_optimized.py \
  --data large_dataset.json \
  --epochs 20 \
  --batch-size 4 \
  --model-name t5-base \
  --output-dir ./production-model
```

### Prediction Workflows

#### Predict with Default Text
```bash
python bias_detector_mps_optimized.py --predict-only --model-path ./bias-detector-output
```

#### Predict Custom Article
```bash
python bias_detector_mps_optimized.py \
  --predict-only \
  --model-path ./my-model \
  --test-article "Your article text here..."
```

## 📊 Model Size Guide

| Model    | Parameters | RAM Needed | Speed     | Batch Size |
|----------|-----------|------------|-----------|------------|
| t5-small | 60M       | 4GB+       | Fast      | 4-8        |
| t5-base  | 220M      | 8GB+       | Medium    | 2-4        |
| t5-large | 770M      | 16GB+      | Slow      | 1-2        |

## 🎯 Mac Model Recommendations

### M1/M2 (8GB)
```bash
--model-name t5-small --batch-size 2 --gradient-accumulation 2
```

### M1/M2 Pro (16GB)
```bash
--model-name t5-base --batch-size 4
```

### M1/M2 Max (32GB+)
```bash
--model-name t5-base --batch-size 8
# or
--model-name t5-large --batch-size 2
```

### M3/M4 Series
```bash
# Add 50% more batch size than equivalent M1/M2
```

## 🔧 Troubleshooting Quick Fixes

### Out of Memory
```bash
# Reduce batch size
--batch-size 1 --gradient-accumulation 4

# Or force CPU
--cpu-only
```

### Slow Training
```bash
# Check MPS is detected - look for:
# "✓ MPS (Apple Silicon GPU) detected and enabled"

# If using CPU, remove any --cpu-only flag
```

### Poor Predictions
```bash
# Train longer
--epochs 20

# Use more data
--data larger_dataset.json

# Try larger model
--model-name t5-base
```

## 📁 File Structure After Training

```
bias-detector-output/
├── adapter_config.json      # LoRA configuration
├── adapter_model.bin         # Trained LoRA weights
├── device_info.json          # Device used for training
├── special_tokens_map.json   # Tokenizer config
├── tokenizer_config.json     # Tokenizer config
├── tokenizer.json            # Tokenizer vocab
└── logs/                     # Training logs
```

## 📝 Data Validation Checklist

Before training, ensure your JSON data has:
- ✅ Valid JSON array structure
- ✅ Each entry has "article" (string)
- ✅ Each entry has "label" (object)
- ✅ Label has "dir" with L, C, R probabilities
- ✅ Label has "deg" with L, M, H probabilities
- ✅ Label has "reason" (string)
- ✅ Probabilities sum to ~1.0

## 🔍 Verification Commands

### Check PyTorch MPS
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

### Check Model Size
```bash
du -sh ./bias-detector-output/
```

### View Training Logs
```bash
cat ./bias-detector-output/logs/*/events.out.*
# or use tensorboard if installed:
# tensorboard --logdir ./bias-detector-output/logs
```

## ⚡ Performance Optimization Tips

1. **Use appropriate batch size**
   - Start with 2, increase until OOM
   - Use gradient accumulation if needed

2. **Monitor Activity Monitor**
   - GPU should show activity during training
   - Memory pressure should be green/yellow

3. **Close other apps**
   - Free up RAM for larger batches
   - Reduce GPU contention

4. **Use latest PyTorch**
   ```bash
   pip install --upgrade torch
   ```

5. **Keep macOS updated**
   - Better MPS performance in newer versions

## 🎓 Example Training Session

```bash
# 1. Create data template
python bias_detector_mps_optimized.py --export-sample my_data.json

# 2. Edit my_data.json with your articles

# 3. Train
python bias_detector_mps_optimized.py \
  --data my_data.json \
  --epochs 15 \
  --batch-size 4 \
  --output-dir ./my-trained-model

# 4. Test
python bias_detector_mps_optimized.py \
  --predict-only \
  --model-path ./my-trained-model \
  --test-article "Test article text..."
```

## 📞 Getting Help

1. Run with `--help` for all options
2. Check console output for MPS detection
3. Monitor Activity Monitor during training
4. Review README_MPS.md for detailed docs

## 🔗 Quick Links

- Original code: `bias_detector.py`
- MPS optimized: `bias_detector_mps_optimized.py`
- Full docs: `README_MPS.md`
- This guide: `QUICK_REFERENCE.md`
