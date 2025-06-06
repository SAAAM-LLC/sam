# Setup environment
python setup_sam.py --dataset your_data.jsonl --format jsonl

# Train model  
python run.py --mode train --data data/processed/ --config configs/train_config.json

# Interactive use
python run.py --mode interact --model models/checkpoints/sam_latest
