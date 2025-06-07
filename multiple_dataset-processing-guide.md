# 📁 Batch Processing Guide for Multiple Datasets

## 🚀 Quick Setup for Your Use Case

### 1. Directory Structure Setup
```bash
# Create workspace
python setup_sam.py --create-dirs-only

# Your structure will look like:
sam_workspace/
├── data/
│   ├── raw/                    # Put ALL your datasets here!
│   │   ├── reasoning_data.jsonl
│   │   ├── code_examples.jsonl
│   │   ├── conversations.json
│   │   ├── text_corpus.txt
│   │   ├── scientific_papers.csv
│   │   └── ... (200+ files)
│   ├── processed/              # Processed output goes here
│   └── vocabulary/
└── configs/
    └── datasets_config.json   # Batch processing config
```

### 2. Process ALL Files at Once
```bash
# Option 1: Process entire raw directory (easiest!)
python setup_sam.py --datasets-dir data/raw/ --reasoning-format --create-vocab

# Option 2: Use config file (more control)
python setup_sam.py --datasets-config configs/datasets_config.json --create-vocab
```

### 3. Configuration for 200-300M Parameters

Create `configs/large_model_config.json`:
```json
{
  "model": {
    "initial_hidden_dim": 1024,
    "initial_num_layers": 12,
    "max_hidden_dim": 2048,
    "max_num_layers": 20,
    "concept_memory_size": 200000,
    "growth_factor": 1.3,
    "neurochemical_enabled": true,
    "biological_computing": true,
    "emergent_representations": true,
    "multi_level_evolution": true,
    "distributed_cognition": true
  },
  "training": {
    "batch_size": 8,
    "learning_rate": 2e-5,
    "max_steps": 200000,
    "mixed_precision": true,
    "gradient_checkpointing": true
  },
  "evolution": {
    "evolve_every": 1000,
    "dream_cycle_minutes": 1.0
  }
}
```

## 📝 Dataset Configuration Examples

### Config for Mixed Formats (`configs/datasets_config.json`)
```json
{
  "datasets": [
    {
      "path": "data/raw/reasoning_conversations.jsonl",
      "format": "jsonl",
      "type": "reasoning",
      "options": {
        "preserve_reasoning_structure": true
      }
    },
    {
      "path": "data/raw/code_examples.jsonl", 
      "format": "jsonl",
      "type": "code",
      "options": {
        "preserve_code_structure": true
      }
    },
    {
      "path": "data/raw/scientific_papers.csv",
      "format": "csv", 
      "type": "standard",
      "options": {
        "text_field": "abstract",
        "chunk_size": 2000
      }
    },
    {
      "path": "data/raw/conversations.json",
      "format": "json",
      "type": "standard"
    },
    {
      "path": "data/raw/text_corpus.txt",
      "format": "txt",
      "type": "standard", 
      "options": {
        "chunk_size": 1500,
        "overlap": 200
      }
    }
  ]
}
```

### Your Specific Reasoning Format
For your format with `messages`, `role`, `content`, `<think></think>`, `<answer></answer>`:

```json
{
  "datasets": [
    {
      "path": "data/raw/your_reasoning_data.jsonl",
      "format": "jsonl",
      "type": "reasoning",
      "options": {
        "preserve_reasoning_structure": true,
        "extract_think_answer": true
      }
    }
  ]
}
```

The system will automatically:
- Detect the `messages` array format
- Extract `role` and `content` 
- Preserve `<think></think>` and `<answer></answer>` structure
- Store metadata from `info` fields
- Maintain `reference_answer` and `test_case` data

## 🔧 Processing Commands

### Basic Batch Processing
```bash
# Process all files in raw directory
python setup_sam.py --datasets-dir data/raw/ --create-vocab

# With reasoning format detection
python setup_sam.py --datasets-dir data/raw/ --reasoning-format --create-vocab

# Parallel processing (faster)
python setup_sam.py --datasets-dir data/raw/ --parallel-workers 8 --create-vocab
```

### Advanced Processing
```bash
# Use specific config
python setup_sam.py --datasets-config configs/datasets_config.json

# Custom output location
python setup_sam.py --datasets-dir data/raw/ --output data/processed_custom/

# Process only specific formats
find data/raw/ -name "*.jsonl" | head -50 > temp_list.txt
# Then use config file with paths from temp_list.txt
```

## 📊 Expected Output

After processing, you'll get:
```
data/processed/
├── reasoning_your_reasoning_data.jsonl      # Your thinking/reasoning data
├── processed_code_examples.jsonl           # Code examples  
├── processed_conversations.jsonl           # Conversation data
├── processed_scientific_papers.jsonl       # Scientific text
├── processed_text_corpus.jsonl            # General text corpus
├── combined_vocabulary.txt                  # Unified vocabulary
└── processing_summary.json                 # Statistics
```

## 🧠 Training on All Data

### Start Training
```bash
# Train on all processed data
python run.py --mode train \
  --data data/processed/ \
  --config configs/large_model_config.json \
  --wandb

# With distributed training (if multiple GPUs)
python run.py --mode train \
  --data data/processed/ \
  --config configs/large_model_config.json \
  --distributed --world-size 4 --rank 0
```

### Monitor Training
```bash
# Check progress
python run.py --mode interact --model models/checkpoints/sam_latest

# Run evaluation
python run.py --mode evaluate --model models/checkpoints/sam_latest --eval-data data/processed/
```

## 🎯 Optimizations for Large-Scale Processing

### Memory Optimization
```json
{
  "training": {
    "batch_size": 4,              // Reduce if OOM
    "gradient_checkpointing": true,
    "mixed_precision": true,
    "offload_optimizer": true     // For very large models
  }
}
```

### Processing Optimization
```bash
# Use more workers for large datasets
python setup_sam.py --datasets-dir data/raw/ --parallel-workers 16

# Process in chunks for very large datasets
split -l 100000 huge_dataset.jsonl chunk_
for chunk in chunk_*; do
  python setup_sam.py --dataset $chunk --format jsonl
done
```

## 🔍 Monitoring Processing

### Check Progress
```bash
# Monitor processing
tail -f logs/setup.log

# Check processed examples
ls -la data/processed/
grep "processed_examples" data/processed/*.info.json
```

### Validate Data Quality
```python
# Quick validation script
import json

def check_processed_data(file_path):
    with open(file_path, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    print(f"File: {file_path}")
    print(f"Examples: {len(examples)}")
    print(f"Avg length: {sum(len(ex['text']) for ex in examples) / len(examples):.0f}")
    print(f"Sample: {examples[0]['text'][:100]}...")
    
    # Check reasoning format
    reasoning_examples = [ex for ex in examples if '<think>' in ex['text']]
    print(f"Reasoning examples: {len(reasoning_examples)}")

# Check all processed files
import glob
for file in glob.glob("data/processed/*.jsonl"):
    check_processed_data(file)
    print("-" * 50)
```

## 🚨 Common Issues & Solutions

### Large File Processing
```bash
# If files are too large, split them first
split -l 50000 large_file.jsonl split_
python setup_sam.py --datasets-dir . --pattern "split_*"
```

### Memory Issues During Processing
```bash
# Reduce chunk size
python setup_sam.py --datasets-dir data/raw/ --chunk-size 500

# Process files one by one
for file in data/raw/*.jsonl; do
  python setup_sam.py --dataset "$file" --format jsonl
done
```

### Format Detection Issues
```bash
# Force specific format
python setup_sam.py --dataset data.jsonl --format jsonl --reasoning-format

# Check file contents first
head -n 5 your_file.jsonl
```

## 🎉 Success Checklist

After processing, you should have:
- ✅ All files processed without errors
- ✅ Combined vocabulary file created  
- ✅ Processing summary with statistics
- ✅ Reasoning structure preserved in processed files
- ✅ Metadata maintained for all examples
- ✅ Ready for training with 200-300M parameter model

Your SAM will learn from ALL formats: reasoning, code, conversations, papers, and general text - creating a truly comprehensive AI! 🧠✨
