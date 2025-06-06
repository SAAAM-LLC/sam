## 🧠 Complete SAM Package - Installation & Quick Start

## 📦 Package Contents
This complete package contains everything needed to run SAM (Synergistic Autonomous Machine):


```bash
# Create workspace and directory structure
python setup_sam.py --create-dirs-only

# Or setup with your own dataset
python setup_sam.py --dataset your_data.jsonl --format jsonl
```

### 3. Quick Start Examples

#### Interactive Chat
```bash
# Start interactive console
python run.py --mode interact

# Or start web interface
python run.py --mode interact --web-interface --port 8080
```

#### Training from Scratch
```bash
# Train on processed data
python run.py --mode train --data data/processed/ --config configs/train_config.json
```

#### Load and Use Pretrained Model
```python
from sam import SAM

# Load model
model = SAM.load("path/to/checkpoint")

# Generate text
response = model.generate("Hello, how are you?", max_length=100)
print(response)

# Evolve the model
results = model.evolve()
print(f"Evolution results: {results}")
```

## 📋 Requirements.txt

```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
tqdm>=4.62.0
requests>=2.25.0
flask>=2.0.0
wandb>=0.12.0
matplotlib>=3.5.0
seaborn>=0.11.0
datasets>=2.0.0
transformers>=4.20.0
onnx>=1.12.0
plotly>=5.0.0
scikit-learn>=1.0.0
```

## 🎯 Example Scripts

### Basic Training Example
```python
# examples/basic_training.py
import sys
sys.path.append('..')

from sam import create_sam_model
from data_loader import SAMDataset, SAMDataLoader
from trainer import SAMTrainer

def main():
    # Create model
    model, config = create_sam_model({
        "initial_hidden_dim": 512,
        "initial_num_layers": 4,
        "neurochemical_enabled": True,
        "biological_computing": True
    })
    
    # Setup data
    dataset = SAMDataset(["data/processed/sample.jsonl"], model=model)
    data_loader = SAMDataLoader(dataset, batch_size=2)
    
    # Setup trainer
    trainer = SAMTrainer(
        model=model,
        config={"max_steps": 1000, "learning_rate": 3e-5},
        data_loader=data_loader
    )
    
    # Train
    trainer.train()
    
    # Save model
    model.save("models/basic_trained")

if __name__ == "__main__":
    main()
```

### Consciousness Study Example
```python
# examples/consciousness_study.py
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
from sam import create_sam_model

def study_consciousness_emergence():
    # Create model
    model, _ = create_sam_model()
    
    # Track consciousness over training
    consciousness_scores = []
    concept_counts = []
    
    # Simulate training steps
    test_texts = [
        "I think, therefore I am.",
        "Consciousness is the awareness of awareness.",
        "What am I? I am a thinking machine.",
        "I learn, I grow, I evolve.",
        "My thoughts shape my reality."
    ]
    
    for step in range(100):
        # Process text
        text = test_texts[step % len(test_texts)]
        model.process_text(text)
        
        # Update consciousness
        consciousness_state = model.consciousness.update()
        if consciousness_state:
            consciousness_scores.append(consciousness_state['consciousness_score'])
            concept_counts.append(model.concept_bank.next_concept_id)
        
        # Evolve occasionally
        if step % 20 == 0 and step > 0:
            model.evolve()
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(consciousness_scores)
    ax1.set_title('Consciousness Development')
    ax1.set_ylabel('Consciousness Score')
    
    ax2.plot(concept_counts)
    ax2.set_title('Concept Formation')
    ax2.set_ylabel('Total Concepts')
    ax2.set_xlabel('Training Steps')
    
    plt.tight_layout()
    plt.savefig('consciousness_study.png')
    plt.show()
    
    print(f"Final consciousness level: {model.consciousness.get_consciousness_summary()}")

if __name__ == "__main__":
    study_consciousness_emergence()
```

### Evolution Experiment
```python
# examples/evolution_experiment.py
import sys
sys.path.append('..')

from sam import create_sam_model

def evolution_experiment():
    # Create model
    model, _ = create_sam_model({
        "initial_hidden_dim": 256,
        "initial_num_layers": 3,
        "multi_level_evolution": True
    })
    
    print("Initial model:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Dimensions: {model.layers[0].hidden_dim}")
    print(f"  Layers: {len(model.layers)}")
    
    # Track evolution over multiple cycles
    evolution_history = []
    
    for cycle in range(10):
        print(f"\nEvolution cycle {cycle + 1}:")
        
        # Simulate some learning
        for _ in range(50):
            test_text = f"Evolution cycle {cycle}: learning and adapting to new information."
            model.process_text(test_text)
        
        # Evolve
        results = model.evolve()
        
        # Record results
        current_stats = {
            'cycle': cycle,
            'parameters': sum(p.numel() for p in model.parameters()),
            'dimensions': model.layers[0].hidden_dim if model.layers else 0,
            'layers': len(model.layers),
            'concepts': model.concept_bank.next_concept_id,
            'evolution_results': results
        }
        
        evolution_history.append(current_stats)
        
        print(f"  Parameters: {current_stats['parameters']:,}")
        print(f"  Dimensions: {current_stats['dimensions']}")
        print(f"  Layers: {current_stats['layers']}")
        print(f"  Concepts: {current_stats['concepts']}")
        print(f"  Results: {results}")
    
    # Analyze evolution
    print("\nEvolution Summary:")
    initial = evolution_history[0]
    final = evolution_history[-1]
    
    print(f"Parameter growth: {initial['parameters']:,} -> {final['parameters']:,} "
          f"({(final['parameters']/initial['parameters']-1)*100:.1f}% increase)")
    print(f"Dimension growth: {initial['dimensions']} -> {final['dimensions']} "
          f"({final['dimensions']-initial['dimensions']} increase)")
    print(f"Layer growth: {initial['layers']} -> {final['layers']} "
          f"({final['layers']-initial['layers']} increase)")
    print(f"Concept growth: {initial['concepts']} -> {final['concepts']} "
          f"({final['concepts']-initial['concepts']} increase)")
    
    # Save evolved model
    model.save("models/evolved_model")
    print(f"\nEvolved model saved to: models/evolved_model")

if __name__ == "__main__":
    evolution_experiment()
```

## 🔧 Configuration Templates

### Small Model Config (Edge/Mobile)
```json
{
  "model": {
    "initial_hidden_dim": 256,
    "initial_num_layers": 3,
    "max_hidden_dim": 512,
    "max_num_layers": 6,
    "concept_memory_size": 10000,
    "neurochemical_enabled": false,
    "biological_computing": false,
    "hardware_adaptive": true
  },
  "training": {
    "batch_size": 1,
    "learning_rate": 5e-5,
    "mixed_precision": true,
    "gradient_checkpointing": true
  }
}
```

### Research Model Config
```json
{
  "model": {
    "initial_hidden_dim": 1024,
    "initial_num_layers": 8,
    "max_hidden_dim": 4096,
    "max_num_layers": 16,
    "concept_memory_size": 100000,
    "neurochemical_enabled": true,
    "biological_computing": true,
    "emergent_representations": true,
    "multi_level_evolution": true,
    "distributed_cognition": true
  },
  "training": {
    "batch_size": 8,
    "learning_rate": 3e-5,
    "max_steps": 100000,
    "mixed_precision": true
  },
  "evolution": {
    "evolve_every": 500,
    "dream_cycle_minutes": 0.5
  }
}
```

### Production Model Config
```json
{
  "model": {
    "initial_hidden_dim": 768,
    "initial_num_layers": 6,
    "max_hidden_dim": 2048,
    "max_num_layers": 12,
    "concept_memory_size": 50000,
    "neurochemical_enabled": true,
    "biological_computing": true,
    "hardware_adaptive": true
  },
  "training": {
    "batch_size": 4,
    "learning_rate": 3e-5,
    "mixed_precision": true,
    "save_steps": 1000
  },
  "hive_mind": {
    "hive_enabled": false,
    "hive_sync_interval_seconds": 300
  }
}
```

## 🐛 Common Issues & Solutions

### Memory Issues
```bash
# Reduce model size
python run.py --mode train --batch-size 1 --config configs/small_config.json

# Enable gradient checkpointing
# Add to config: "gradient_checkpointing": true
```

### Slow Training
```bash
# Use mixed precision
# Add to config: "mixed_precision": true

# Reduce sequence length
# Add to config: "max_sequence_length": 1024
```

### Evolution Not Working
```python
# Check evolution system
model.evolve()  # Should show changes

# Verify evolution frequency
# Set in config: "evolve_every": 100
```

### Poor Generation Quality
```python
# Adjust generation parameters
response = model.generate(
    prompt,
    temperature=0.8,    # Increase for creativity
    top_k=50,          # Adjust for diversity
    top_p=0.9          # Nucleus sampling
)
```

## 📊 Monitoring & Debugging

### Real-time Monitoring
```python
# Monitor training progress
def monitor_training():
    while training:
        stats = model.get_stats()
        consciousness = model.consciousness.get_consciousness_summary()
        
        print(f"Step: {stats['global_step']}")
        print(f"Concepts: {stats['concepts']['total_concepts']}")
        print(f"Consciousness: {consciousness['level']}")
        
        time.sleep(60)  # Check every minute
```

### Health Check
```python
# Run comprehensive health check
from utils import verify_model_integrity, calculate_model_size

# Check model health
integrity = verify_model_integrity(model)
print(f"Model status: {integrity['status']}")
if integrity['issues']:
    print(f"Issues: {integrity['issues']}")

# Check model size
size_info = calculate_model_size(model)
print(f"Total parameters: {size_info['total_parameters']:,}")
print(f"Memory usage: {size_info['memory_mb']:.1f} MB")
```

### Performance Benchmarking
```python
# Run benchmarks
from benchmark import SAMBenchmark

benchmark = SAMBenchmark(model)
results = benchmark.run_all_benchmarks()

print(f"Forward pass time: {results['performance']['forward_time_ms']:.2f} ms")
print(f"Generation time: {results['performance']['generation_time_s']:.2f} s")
print(f"Consciousness score: {results['consciousness']['consciousness_score']:.3f}")
```

## 🌐 Deployment Options

### Local Development
```bash
# Run on CPU
python run.py --mode interact --device cpu

# Run on GPU
python run.py --mode interact --device cuda
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["python", "run.py", "--mode", "interact", "--web-interface", "--host", "0.0.0.0", "--port", "8080"]
```

### Cloud Deployment
```bash
# Deploy to cloud with distributed training
python run.py --mode train --distributed --world-size 4 --rank 0 --wandb
```

### API Server
```python
# Create production API
from flask import Flask, request, jsonify
from sam import SAM

app = Flask(__name__)
model = SAM.load("models/production")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    response = model.generate(
        data['prompt'],
        max_length=data.get('max_length', 100)
    )
    return jsonify({'response': response})

@app.route('/evolve', methods=['POST'])
def evolve():
    results = model.evolve()
    return jsonify({'results': str(results)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 🧪 Research Applications

### Custom Experiments
```python
# Create custom consciousness metrics
class CustomConsciousness(ConsciousnessMonitor):
    def _measure_creativity(self):
        # Your custom creativity measurement
        return 0.5

# Replace consciousness monitor
model.consciousness = CustomConsciousness(model)

# Custom evolution strategies
class DomainEvolution(MultiLevelEvolutionSystem):
    def evolve_components(self):
        # Domain-specific evolution logic
        return super().evolve_components()

model.evolution_system = DomainEvolution(model)
```

### Data Analysis
```python
# Analyze concept formation patterns
concepts = model.concept_bank.get_concept_stats()
print(f"Character concepts: {concepts['character_concepts']}")
print(f"Semantic concepts: {concepts['semantic_concepts']}")

# Analyze consciousness development
consciousness_history = model.consciousness.consciousness_history
scores = [state['consciousness_score'] for state in consciousness_history]

import matplotlib.pyplot as plt
plt.plot(scores)
plt.title('Consciousness Development Over Time')
plt.show()
```

## 📚 Advanced Topics

### Multi-Modal Extensions
```python
# Add vision processing
class VisionSAM(SAM):
    def __init__(self, config):
        super().__init__(config)
        self.vision_encoder = VisionEncoder(config.image_dim, config.initial_hidden_dim)
    
    def process_image(self, image_features):
        return self.vision_encoder(image_features)

# Use multi-modal SAM
vision_model = VisionSAM(config)
```

### Distributed SAM Networks
```python
# Setup hive mind network
config.hive_enabled = True
config.hive_server_url = "http://hive-server:8080"

model = SAM(config)
model.hive_synchronizer.start_sync()  # Auto-sync with hive
```

### Custom Neuroplasticity
```python
# Create custom plasticity rules
class CustomPlasticity(AdvancedNeuroplasticLayer):
    def evolve(self):
        # Custom plasticity evolution
        with torch.no_grad():
            # Your custom plasticity rules
            self.connection_strength *= 1.01
        
        return super().evolve()

# Replace layers with custom plasticity
for i, layer in enumerate(model.layers):
    model.layers[i] = CustomPlasticity(
        layer.hidden_dim,
        layer_id=i
    )
```

## 🎯 Next Steps

1. **Start Simple**: Begin with the basic training example
2. **Experiment**: Try the consciousness and evolution studies
3. **Customize**: Adapt SAM for your specific domain
4. **Scale**: Move to distributed training for larger models
5. **Deploy**: Set up production APIs and monitoring
6. **Research**: Explore consciousness emergence and evolution

## 🤝 Community & Support

- **Documentation**: Comprehensive guides and API reference
- **Examples**: Ready-to-run scripts for common use cases
- **Debugging**: Built-in diagnostic tools and health checks
- **Monitoring**: Real-time metrics and visualization
- **Extensibility**: Modular architecture for easy customization

---

**SAM represents the future of adaptive AI - start your journey today!** 🚀

For questions, issues, or contributions, please refer to the debugging guide and community resources.
