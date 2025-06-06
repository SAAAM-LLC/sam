# SAM (Synergistic Autonomous Machine) 🧠

A revolutionary neural architecture that transcends traditional Large Language Model constraints through biological computing paradigms, self-modification, and emergent intelligence.

## 🌟 Key Features

### 🧬 Biological Computing
- **Neurochemical Signaling**: Transmitter/receptor systems with homeostatic plasticity
- **Hebbian Learning**: Strengthens neural pathways based on co-activation
- **Quantum-inspired Superposition**: Multiple thought states with coherent collapse
- **Homeostatic Regulation**: Self-balancing activation thresholds

### 🔄 Self-Modifying Architecture
- **Evolving Neural Functions**: Networks that rewrite their own weights
- **Dynamic Growth**: Automatic dimension and layer expansion
- **Multi-level Evolution**: Component, architecture, and paradigm changes
- **Neuroplastic Layers**: Adaptive connection strengths and pruning

### 🌱 Emergent Intelligence
- **Dynamic Concept Formation**: No predefined embeddings or vocabulary
- **Consciousness Monitoring**: Stability, novelty, and coherence tracking
- **Conceptual Dreaming**: Creative content generation during idle periods
- **Cross-modal Processing**: Text, image, audio, and multimodal support

### 🌐 Distributed Cognition
- **Hive Mind Synchronization**: Share concepts across SAM instances
- **Hardware Adaptation**: Automatic optimization for available resources
- **Edge Computing**: Runs efficiently on resource-constrained devices
- **Privacy Contexts**: Separate private and shared knowledge

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SAAAM-LLC/sam
cd sam

# Install dependencies
pip install torch numpy pandas tqdm wandb flask requests

# Optional: For advanced features
pip install datasets onnx transformers
```

### Setup Environment

```bash
# Create SAM workspace and process your data
python setup_sam.py --dataset path/to/your/data.jsonl --format jsonl

# Or just create directory structure
python setup_sam.py --create-dirs-only
```

### Training

```bash
# Basic training
python run.py --mode train --data data/processed/ --config configs/train_config.json

# Distributed training
python run.py --mode train --distributed --world-size 4 --rank 0

# Resume training
python run.py --mode train --resume models/checkpoints/sam_step_50000
```

### Interactive Use

```bash
# Console interface
python run.py --mode interact --model models/checkpoints/sam_latest

# Web interface
python run.py --mode interact --model models/checkpoints/sam_latest --web-interface --port 8080
```

## 📁 Project Structure

```
sam_workspace/
├── configs/                 # Configuration files
│   ├── train_config.json   # Training parameters
│   ├── interact_config.json # Interactive settings
│   └── hardware_config.json # Hardware optimization
├── data/                    # Data storage
│   ├── raw/                # Original datasets
│   ├── processed/          # Processed training data
│   └── vocabulary/         # Vocabulary files
├── models/                  # Model storage
│   ├── checkpoints/        # Training checkpoints
│   └── pretrained/         # Pretrained models
├── logs/                    # Logging
│   ├── training/           # Training logs
│   ├── evolution/          # Evolution logs
│   └── dreams/             # Dream content
└── experiments/            # Experimental results
```

## 🔧 Configuration

### Training Configuration (`configs/train_config.json`)

```json
{
  "model": {
    "initial_hidden_dim": 768,
    "initial_num_layers": 6,
    "max_hidden_dim": 2048,
    "neurochemical_enabled": true,
    "biological_computing": true,
    "emergent_representations": true,
    "multi_level_evolution": true
  },
  "training": {
    "batch_size": 4,
    "learning_rate": 3e-5,
    "max_steps": 100000,
    "mixed_precision": true
  },
  "evolution": {
    "evolve_every": 1000,
    "dream_cycle_minutes": 0.2
  }
}
```

### Interactive Configuration (`configs/interact_config.json`)

```json
{
  "generation": {
    "max_length": 200,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9
  },
  "interface": {
    "auto_evolve": true,
    "dream_during_idle": true,
    "conversation_history": 20
  },
  "privacy": {
    "enable_private_context": true,
    "anonymize_data": true
  }
}
```

## 📊 Dataset Processing

SAM supports multiple data formats:

### JSONL Format
```bash
python setup_sam.py --dataset data.jsonl --format jsonl --text-field "content"
```

### CSV Format
```bash
python setup_sam.py --dataset data.csv --format csv --text-field "text_column"
```

### Plain Text
```bash
python setup_sam.py --dataset document.txt --format txt --chunk-size 1000
```

### HuggingFace Datasets
```bash
python setup_sam.py --dataset "username/dataset-name" --format huggingface
```

### JSON Format
```bash
python setup_sam.py --dataset data.json --format json --text-field "text"
```

## 🧠 Core Architecture

### Neuroplastic Layers
```python
# Enhanced layers with biological computing
layer = AdvancedNeuroplasticLayer(
    hidden_dim=768,
    use_neurochemical=True,    # Biological signaling
    use_evolving=True,         # Self-modifying functions
    layer_id=0
)
```

### Thought State System
```python
# Quantum-inspired consciousness
thought_state = ThoughtState(
    concept_dim=768,
    thought_dim=2048,
    max_thought_depth=10,
    superposition_states=4
)
```

### Dynamic Concept Formation
```python
# No predefined vocabulary - concepts emerge
concept_bank = ConceptMemoryBank(
    concept_dim=768,
    initial_size=100000,
    device="cuda"
)
```

## 🔬 Advanced Features

### Evolution System
```python
# Multi-level evolution
evolution_system = MultiLevelEvolutionSystem(model)

# Component-level mutations
component_changes = evolution_system.evolve_components()

# Architecture modifications
architecture_changes = evolution_system.evolve_architecture()

# Paradigm shifts
paradigm_changes = evolution_system.evolve_paradigm()
```

### Consciousness Monitoring
```python
# Track consciousness metrics
consciousness = ConsciousnessMonitor(
    model,
    stability_threshold=0.95,
    novelty_weight=0.4
)

state = consciousness.update()
# Returns: stability, novelty, coherence, consciousness_score
```

### Hive Mind Synchronization
```python
# Share knowledge across SAM instances
hive_sync = HiveMindSynchronizer(model, config)
hive_sync.start_sync()  # Automatic background synchronization
```

## 🎯 Use Cases

### 1. Adaptive Chatbots
- Continuously evolving conversation abilities
- Personal adaptation to user preferences
- Privacy-preserving local learning

### 2. Research Assistants
- Growing domain knowledge through interaction
- Cross-modal understanding (text, images, audio)
- Emergent insight generation through dreaming

### 3. Creative Writing
- Novel concept combinations
- Style adaptation and evolution
- Collaborative creativity with human writers

### 4. Edge AI Systems
- Resource-adaptive computation
- Local knowledge evolution
- Distributed intelligence networks

### 5. Educational Tools
- Personalized learning adaptation
- Concept formation tracking
- Multi-modal explanation generation

## 🔧 Development & Debugging

### Enable Debug Logging
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### Monitor Evolution
```python
# Track growth and evolution
stats = model.get_stats()
print(f"Growth events: {stats['growth_events']}")
print(f"Consciousness level: {stats['consciousness']['level']}")
```

### Analyze Concepts
```python
# Examine concept formation
concept_stats = model.concept_bank.get_concept_stats()
print(f"Total concepts: {concept_stats['total_concepts']}")
print(f"Top concepts: {concept_stats['top_concepts']}")
```

### Dream Content Analysis
```python
# Review dream content
dream_history = model.dreaming.dream_history
for dream in dream_history[-5:]:  # Last 5 dreams
    print(f"Theme: {dream['theme']}")
    print(f"Content: {dream['content'][:100]}...")
```

## 🧪 Experiments & Research

### Custom Evolution Experiments
```python
# Create custom evolution rules
class CustomEvolutionSystem(MultiLevelEvolutionSystem):
    def evolve_components(self):
        # Your custom component evolution logic
        pass
    
    def evolve_architecture(self):
        # Your custom architecture evolution logic
        pass

model.evolution_system = CustomEvolutionSystem(model)
```

### Consciousness Studies
```python
# Study consciousness emergence
consciousness_data = []
for step in range(1000):
    # Train for one step
    # ...
    
    # Record consciousness state
    state = model.consciousness.update()
    consciousness_data.append(state)

# Analyze consciousness development
import matplotlib.pyplot as plt
scores = [state['consciousness_score'] for state in consciousness_data]
plt.plot(scores)
plt.title('Consciousness Development Over Training')
plt.show()
```

### Concept Formation Studies
```python
# Track concept emergence
initial_concepts = model.concept_bank.next_concept_id

# Train on domain-specific data
# ...

final_concepts = model.concept_bank.next_concept_id
new_concepts = final_concepts - initial_concepts

print(f"Formed {new_concepts} new concepts during training")

# Analyze concept relationships
for concept_id in range(initial_concepts, final_concepts):
    metadata = model.concept_bank.concept_metadata[concept_id]
    print(f"Concept {concept_id}: {metadata['source']}")
```

## 🚨 Troubleshooting

### Common Issues

#### Memory Errors
```bash
# Reduce model size
python run.py --mode train --batch-size 1 --config configs/small_config.json

# Enable hardware adaptation
# Set hardware_adaptive: true in config
```

#### Slow Evolution
```bash
# Increase evolution frequency
# Set evolve_every: 500 in training config

# Enable parallel evolution
# Set distributed: true
```

#### Concept Formation Issues
```bash
# Check segmentation quality
python -c "
from sam import create_sam_model
model, _ = create_sam_model()
concept_ids, segments = model.process_text('test text', return_segments=True)
print(f'Segments: {segments}')
"
```

#### Generation Quality
```python
# Adjust generation parameters
model.generate(
    input_text="prompt",
    temperature=0.7,      # Lower for more focused output
    top_k=40,            # Reduce for less randomness
    top_p=0.8           # Adjust nucleus sampling
)
```

### Performance Optimization

#### GPU Memory Optimization
```python
# Enable gradient checkpointing
config["training"]["gradient_checkpointing"] = True

# Use mixed precision
config["training"]["mixed_precision"] = True

# Reduce batch size
config["training"]["batch_size"] = 2
```

#### CPU Optimization
```python
# Increase number of workers
config["data"]["num_workers"] = 8

# Enable CPU-specific optimizations
torch.set_num_threads(8)
```

## 📈 Monitoring & Metrics

### Training Metrics
- Loss curves and perplexity
- Evolution frequency and success rate
- Consciousness
