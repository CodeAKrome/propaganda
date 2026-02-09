# Understanding Your Training Data and Expected Outputs

## Your Data Format (Confirmed Compatible ✓)

Based on the sample from your `train.json`, your data is structured correctly:

```json
{
  "article": "Long article text here...",
  "label": {
    "dir": {
      "L": 0.2,    // Left bias probability
      "C": 0.6,    // Center bias probability  
      "R": 0.2     // Right bias probability
    },
    "deg": {
      "L": 0.1,    // Low degree
      "M": 0.8,    // Medium degree
      "H": 0.1     // High degree
    },
    "reason": "Explanation of the bias classification..."
  }
}
```

## What the Model Will Output

After training on your data, the model should produce outputs in the **exact same JSON format**:

```json
{
  "dir": {"L": 0.2, "C": 0.6, "R": 0.2},
  "deg": {"L": 0.1, "M": 0.8, "H": 0.1},
  "reason": "Generated explanation based on the article content"
}
```

## Why You Might See Different Outputs

### 1. **Beam Search vs Greedy Decoding** (NOW FIXED)

The original issue was:
- **Original script**: Used beam search (explores multiple possibilities)
- **MPS script v1.0**: Used greedy decoding (faster but different)
- **MPS script v1.1 (CURRENT)**: Uses beam search by default ✓

**Solution**: The current version matches the original behavior exactly.

### 2. **Model Training Differences**

Even with identical code, outputs can vary due to:
- Random initialization
- Training hardware (CPU vs GPU vs MPS)
- PyTorch version differences
- Numerical precision variations

### 3. **Expected Variation Examples**

Even a well-trained model might produce slightly different probabilities:

**Training Label:**
```json
{
  "dir": {"L": 0.2, "C": 0.6, "R": 0.2},
  "deg": {"L": 0.7, "M": 0.25, "H": 0.05}
}
```

**Model Output (Normal Variation):**
```json
{
  "dir": {"L": 0.18, "C": 0.62, "R": 0.20},
  "deg": {"L": 0.68, "M": 0.27, "H": 0.05}
}
```
✓ This is expected - small variations are normal

**Model Output (Good but Different):**
```json
{
  "dir": {"L": 0.25, "C": 0.55, "R": 0.20},
  "deg": {"L": 0.65, "M": 0.30, "H": 0.05}
}
```
✓ Still reasonable - shows the model learned the pattern

**Model Output (Problematic):**
```json
{
  "dir": {"L": 0.9, "C": 0.05, "R": 0.05},
  "deg": {"L": 0.1, "M": 0.1, "H": 0.8}
}
```
✗ Very different - indicates training issues

## Sample From Your Data

### Entry #1: RFK Jr. Article
**Expected Classification:**
- Direction: Mostly center (0.6) with some left/right balance
- Degree: Moderate (0.8)
- Reason: Negative descriptors but includes opposing viewpoints

### Entry #2: Housing Crisis Article  
**Expected Classification:**
- Direction: Right-leaning (0.7)
- Degree: High (0.6)
- Reason: Promotes conservative approach, cites AEI, praises Trump

### Entry #3: Trans Athletes Article
**Expected Classification:**
- Direction: Right-leaning (0.7)  
- Degree: High (0.6)
- Reason: Uses loaded language, quotes only conservative sources

## Ensuring Consistent Outputs

### Use the Same Prediction Mode

**For Production/Comparison** (Recommended):
```bash
python bias_detector_mps_optimized.py \
  --predict-only \
  --model-path ./your-trained-model
  # Uses beam search by default
```

**For Speed Testing** (3-5x faster):
```bash
python bias_detector_mps_optimized.py \
  --predict-only \
  --model-path ./your-trained-model \
  --fast-predict
  # Uses greedy decoding - may differ
```

### Training Recommendations for Your Data

Based on your sample, here are optimal training settings:

```bash
# For M1/M2 (8GB RAM)
python bias_detector_mps_optimized.py \
  --data train.json \
  --model-name t5-small \
  --epochs 15 \
  --batch-size 2 \
  --output-dir ./my-bias-model

# For M1/M2 Pro/Max (16GB+ RAM)
python bias_detector_mps_optimized.py \
  --data train.json \
  --model-name t5-base \
  --epochs 20 \
  --batch-size 4 \
  --output-dir ./my-bias-model

# For best accuracy (if you have time)
python bias_detector_mps_optimized.py \
  --data train.json \
  --model-name t5-base \
  --epochs 30 \
  --batch-size 4 \
  --learning-rate 3e-4 \
  --output-dir ./my-bias-model
```

## Validating Your Full Dataset

Before training, validate your complete `train.json`:

```bash
python validate_training_data.py train.json
```

This will check:
- ✓ JSON structure is valid
- ✓ All required fields are present
- ✓ Probabilities sum to ~1.0
- ✓ Data types are correct

## Testing After Training

### Step 1: Train the Model
```bash
python bias_detector_mps_optimized.py \
  --data train.json \
  --epochs 15 \
  --output-dir ./test-model
```

### Step 2: Test with Known Article
```bash
python bias_detector_mps_optimized.py \
  --predict-only \
  --model-path ./test-model \
  --test-article "Your test article here..."
```

### Step 3: Compare Output Format

The output should match this structure:
```json
{
  "dir": {
    "L": <number 0-1>,
    "C": <number 0-1>,
    "R": <number 0-1>
  },
  "deg": {
    "L": <number 0-1>,
    "M": <number 0-1>,
    "H": <number 0-1>
  },
  "reason": "Text explanation..."
}
```

If you see:
```json
{
  "error": "JSON Parse Error",
  "raw": "Some malformed output"
}
```

This means:
- Model needs more training
- Training data may have formatting issues
- Model may be too small for the complexity

## Common Issues and Solutions

### Issue 1: Model outputs JSON but with wrong keys
**Cause**: Model memorized structure but not the exact keys  
**Solution**: Train longer or add more examples

### Issue 2: Model outputs text instead of JSON
**Cause**: Not enough training or data format issues  
**Solution**: 
- Increase epochs (try 20-30)
- Validate training data format
- Use larger model (t5-base instead of t5-small)

### Issue 3: Probabilities don't sum to 1.0
**Cause**: Model learned patterns but not exact constraints  
**Solution**: This is actually normal - post-process to normalize if needed

### Issue 4: Different outputs each run
**Cause**: Different prediction modes or random sampling  
**Solution**: 
- Use beam search (default) for consistency
- Avoid `--fast-predict` for reproducible results

## Your Next Steps

1. **Validate your full train.json**:
   ```bash
   python validate_training_data.py train.json
   ```

2. **Train with your data**:
   ```bash
   python bias_detector_mps_optimized.py --data train.json --epochs 15
   ```

3. **Test predictions**:
   ```bash
   python bias_detector_mps_optimized.py \
     --predict-only \
     --model-path ./bias-detector-output \
     --test-article "Test article from your data..."
   ```

4. **Compare with original script** (if needed):
   - Both should use beam search by default
   - Outputs should match when using same model
   - Small numerical differences (<0.05) are normal

## Key Takeaway

✓ Your `train.json` format is **perfect** - no changes needed  
✓ The MPS-optimized script **now matches** the original behavior  
✓ Use **default settings** for predictions that match the original script  
✓ Use `--fast-predict` only when you need speed over exact reproducibility

The model will produce valid JSON outputs in the same format as your training labels. Minor probability variations are normal and expected.
