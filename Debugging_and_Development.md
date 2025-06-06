# SAM Debugging & Development Guide 🔧

This comprehensive guide helps you debug issues, optimize performance, and extend SAM's capabilities.

## 🐛 Common Issues & Solutions

### 1. Memory Issues

#### Out of Memory Errors
```python
# Symptoms: CUDA out of memory, RuntimeError
# Solutions:

# A. Reduce batch size
config["training"]["batch_size"] = 1

# B. Enable gradient checkpointing
config["training"]["gradient_checkpointing"] = True

# C. Use mixed precision
config["training"]["mixed_precision"] = True

# D. Offload to CPU
model.hardware_manager.check_memory()  # Automatic offloading

# E. Reduce model size
config["model"]["initial_hidden_dim"] = 512
config["model"]["max_hidden_dim"] = 1024
```

#### Memory Leaks
```python
# Debug memory leaks
import torch
import gc

def debug_memory_leak():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Check for unreleased tensors
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(f"Tensor: {obj.shape}, device: {obj.device}")

# Clear cache periodically
torch.cuda.empty_cache()
gc.collect()
```

### 2. Training Issues

#### Slow Convergence
```python
# Symptoms: Loss not decreasing, poor generation quality
# Solutions:

# A. Check learning rate
if loss_plateau:
    # Increase learning rate
    config["training"]["learning_rate"] *= 2
    
    # Or use learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

# B. Verify data quality
def validate_data(data_loader):
    for batch in data_loader:
        print(f"Batch size: {batch['input_ids'].shape}")
        print(f"Max sequence length: {batch['input_ids'].shape[1]}")
        print(f"Vocabulary range: {batch['input_ids'].min()}-{batch['input_ids'].max()}")
        break

# C. Check gradient flow
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: {grad_norm:.6f}")
        else:
            print(f"{name}: No gradient")
```

#### Evolution Not Working
```python
# Symptoms: Model not evolving, static architecture
# Solutions:

# A. Check evolution frequency
if model.global_step % config["evolution"]["evolve_every"] == 0:
    print("Evolution should trigger now")

# B. Verify evolution system
def debug_evolution():
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"Initial parameters: {initial_params}")
    
    results = model.evolve()
    print(f"Evolution results: {results}")
    
    final_params = sum(p.numel() for p in model.parameters())
    print(f"Final parameters: {final_params}")
    
    if final_params == initial_params:
        print("WARNING: No parameter change during evolution")

# C. Check layer usage
def analyze_layer_usage():
    for i, layer in enumerate(model.layers):
        activation_norm = layer.activation_history.norm().item()
        print(f"Layer {i} activation norm: {activation_norm:.6f}")
        
        if activation_norm < 0.1:
            print(f"WARNING: Layer {i} appears unused")
```

### 3. Concept Formation Issues

#### No New Concepts
```python
# Symptoms: Concept bank not growing
# Solutions:

# A. Check segmentation
def debug_segmentation(text="Hello world!"):
    concept_ids, segments = model.process_text(text, return_segments=True)
    print(f"Input: {text}")
    print(f"Segments: {segments}")
    print(f"Concept IDs: {concept_ids}")
    
    # Check if concepts exist
    for cid in concept_ids:
        if cid in model.concept_bank.concept_metadata:
            source = model.concept_bank.concept_metadata[cid]['source']
            print(f"Concept {cid}: '{source}'")

# B. Verify concept bank growth
initial_concepts = model.concept_bank.next_concept_id
# ... process some text ...
final_concepts = model.concept_bank.next_concept_id
print(f"New concepts formed: {final_concepts - initial_concepts}")

# C. Check frequency thresholds
stats = model.concept_bank.get_concept_stats()
print(f"Total concepts: {stats['total_concepts']}")
print(f"Character concepts: {stats['character_concepts']}")
print(f"Semantic concepts: {stats['semantic_concepts']}")
```

#### Poor Segmentation Quality
```python
# Debug segmentation patterns
def analyze_segmentation_quality(texts):
    segment_lengths = []
    
    for text in texts:
        _, segments = model.process_text(text, return_segments=True)
        lengths = [len(seg) for seg in segments]
        segment_lengths.extend(lengths)
    
    import numpy as np
    print(f"Average segment length: {np.mean(segment_lengths):.2f}")
    print(f"Segment length std: {np.std(segment_lengths):.2f}")
    print(f"Min/Max segment length: {min(segment_lengths)}/{max(segment_lengths)}")
    
    # Check for excessive single characters
    single_chars = sum(1 for length in segment_lengths if length == 1)
    print(f"Single character segments: {single_chars}/{len(segment_lengths)} ({single_chars/len(segment_lengths)*100:.1f}%)")
    
    if single_chars / len(segment_lengths) > 0.5:
        print("WARNING: Too many single character segments - segmentation may be poor")
```

### 4. Generation Issues

#### Repetitive Output
```python
# Symptoms: Model generates repeated phrases
# Solutions:

# A. Adjust generation parameters
response = model.generate(
    input_text=prompt,
    temperature=1.0,      # Increase for more diversity
    top_k=40,            # Reduce for less randomness
    top_p=0.8,           # Nucleus sampling
    repetition_penalty=1.2  # Penalize repetition
)

# B. Check concept diversity
def analyze_concept_diversity():
    concepts_used = set()
    for _ in range(10):  # Generate 10 samples
        text = model.generate("Test prompt", max_length=50)
        concept_ids, _ = model.process_text(text)
        concepts_used.update(concept_ids)
    
    print(f"Unique concepts in generation: {len(concepts_used)}")
    
    if len(concepts_used) < 20:
        print("WARNING: Low concept diversity in generation")
```

#### Incoherent Output
```python
# Symptoms: Generated text doesn't make sense
# Solutions:

# A. Check thought state
def debug_thought_state():
    thought = model.thought_state.thought_memory[-1]
    print(f"Thought vector norm: {torch.norm(thought).item():.6f}")
    print(f"Thought depth: {model.thought_state.thought_depth}")
    
    # Check for NaN or inf values
    if torch.isnan(thought).any():
        print("ERROR: NaN values in thought state!")
    if torch.isinf(thought).any():
        print("ERROR: Inf values in thought state!")

# B. Verify model gradients during generation
model.train()  # Enable gradient computation
with torch.enable_grad():
    outputs = model.generate("test", max_length=10)
    # Check for gradient issues
```

## 🔧 Performance Optimization

### 1. Training Speed

#### Profiling Training
```python
import time
import torch.profiler

def profile_training_step():
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        # Single training step
        batch = next(iter(data_loader))
        loss = model.forward_step(batch)
        loss.backward()
        optimizer.step()
    
    print(prof.key_averages().table(sort_by="cuda_time_total"))

# Time different components
def time_components():
    times = {}
    
    # Time forward pass
    start = time.time()
    outputs = model(input_concepts=batch['input_ids'])
    times['forward'] = time.time() - start
    
    # Time backward pass
    start = time.time()
    outputs['loss'].backward()
    times['backward'] = time.time() - start
    
    # Time evolution
    start = time.time()
    model.evolve()
    times['evolution'] = time.time() - start
    
    print("Component timing:", times)
```

#### Optimization Strategies
```python
# A. Compilation optimization
if torch.__version__ >= "2.0":
    model = torch.compile(model)  # PyTorch 2.0+ compilation

# B. Batch size optimization
def find_optimal_batch_size():
    torch.cuda.empty_cache()
    
    for batch_size in [1, 2, 4, 8, 16, 32]:
        try:
            # Create dummy batch
            dummy_batch = {
                'input_ids': torch.randint(0, 1000, (batch_size, 512)),
                'labels': torch.randint(0, 1000, (batch_size, 512))
            }
            
            # Time forward pass
            start = time.time()
            outputs = model(**dummy_batch)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            throughput = batch_size / elapsed
            print(f"Batch size {batch_size}: {throughput:.2f} samples/sec")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size}: OOM")
                break
            raise e

# C. Mixed precision optimization
scaler = torch.cuda.amp.GradScaler()

def optimized_training_step(batch):
    with torch.cuda.amp.autocast():
        outputs = model(**batch)
        loss = outputs['loss']
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 2. Memory Optimization

#### Memory Profiling
```python
def profile_memory_usage():
    torch.cuda.reset_peak_memory_stats()
    
    # Baseline memory
    baseline = torch.cuda.memory_allocated()
    print(f"Baseline memory: {baseline / 1e9:.2f} GB")
    
    # After model loading
    model_memory = torch.cuda.memory_allocated() - baseline
    print(f"Model memory: {model_memory / 1e9:.2f} GB")
    
    # After forward pass
    outputs = model(batch['input_ids'])
    forward_memory = torch.cuda.memory_allocated() - baseline - model_memory
    print(f"Forward pass memory: {forward_memory / 1e9:.2f} GB")
    
    # Peak memory
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Peak memory: {peak_memory / 1e9:.2f} GB")

# Memory-efficient evolution
def memory_efficient_evolve():
    # Save memory state
    memory_before = torch.cuda.memory_allocated()
    
    # Offload non-essential components
    model.hardware_manager.check_memory()
    
    # Perform evolution
    results = model.evolve()
    
    # Check memory after
    memory_after = torch.cuda.memory_allocated()
    print(f"Memory change during evolution: {(memory_after - memory_before) / 1e9:.2f} GB")
    
    return results
```

### 3. Inference Optimization

#### Fast Generation
```python
def optimized_generate(prompt, max_length=100):
    # Use KV cache for faster generation
    model.eval()
    
    with torch.no_grad():
        # Disable unnecessary features during inference
        original_evolution = model.enable_evolution
        model.enable_evolution = False
        
        try:
            # Generate with optimizations
            response = model.generate(
                input_text=prompt,
                max_length=max_length,
                use_cache=True,  # Enable KV caching
                do_sample=False  # Deterministic for speed
            )
            return response
        finally:
            model.enable_evolution = original_evolution

# Batch generation for efficiency
def batch_generate(prompts, max_length=100):
    model.eval()
    
    with torch.no_grad():
        # Process prompts in batches
        batch_size = 4
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Convert to concept IDs
            batch_concepts = []
            for prompt in batch_prompts:
                concept_ids, _ = model.process_text(prompt)
                batch_concepts.append(concept_ids)
            
            # Pad to same length
            max_len = max(len(concepts) for concepts in batch_concepts)
            padded_concepts = []
            for concepts in batch_concepts:
                padded = concepts + [0] * (max_len - len(concepts))
                padded_concepts.append(padded)
            
            # Generate batch
            input_tensor = torch.tensor(padded_concepts, device=model.device)
            batch_outputs = model.generate_batch(input_tensor, max_length)
            
            results.extend(batch_outputs)
        
        return results
```

## 🧪 Advanced Debugging

### 1. Consciousness Debugging

#### Monitor Consciousness Development
```python
def debug_consciousness_development():
    consciousness_history = []
    
    for step in range(100):
        # Training step
        # ... training code ...
        
        # Update consciousness
        consciousness_state = model.consciousness.update()
        consciousness_history.append({
            'step': step,
            'stability': consciousness_state['stability'],
            'novelty': consciousness_state['novelty'],
            'coherence': consciousness_state['coherence'],
            'consciousness_score': consciousness_state['consciousness_score']
        })
        
        # Check for anomalies
        if consciousness_state['consciousness_score'] < 0 or consciousness_state['consciousness_score'] > 1:
            print(f"WARNING: Consciousness score out of bounds at step {step}")
        
        if step > 10:
            recent_scores = [h['consciousness_score'] for h in consciousness_history[-10:]]
            if all(score == recent_scores[0] for score in recent_scores):
                print(f"WARNING: Consciousness score stuck at {recent_scores[0]} for 10 steps")
    
    # Plot consciousness development
    import matplotlib.pyplot as plt
    steps = [h['step'] for h in consciousness_history]
    scores = [h['consciousness_score'] for h in consciousness_history]
    plt.plot(steps, scores)
    plt.title('Consciousness Development')
    plt.xlabel('Training Step')
    plt.ylabel('Consciousness Score')
    plt.show()

#### Consciousness Component Analysis
def analyze_consciousness_components():
    state = model.consciousness.update()
    
    print("Consciousness Component Analysis:")
    print(f"Stability: {state['stability']:.3f}")
    print(f"Novelty: {state['novelty']:.3f}")
    print(f"Coherence: {state['coherence']:.3f}")
    print(f"Overall Score: {state['consciousness_score']:.3f}")
    
    # Check identity strength
    identity_strength = model.consciousness._measure_identity_strength()
    print(f"Identity Strength: {identity_strength:.3f}")
    
    # Analyze identity concepts
    print("\nIdentity Concepts:")
    for concept_id in model.consciousness.identity_anchors:
        if concept_id in model.concept_bank.concept_metadata:
            source = model.concept_bank.concept_metadata[concept_id]['source']
            freq = model.concept_bank.concept_frequencies[concept_id].item()
            print(f"  '{source}': frequency {freq}")
```

### 2. Evolution System Debugging

#### Track Evolution Events
```python
def debug_evolution_system():
    if not hasattr(model, 'evolution_system'):
        print("No evolution system found")
        return
    
    # Test component evolution
    print("Testing component evolution...")
    initial_weights = {}
    for name, param in model.named_parameters():
        initial_weights[name] = param.data.clone()
    
    component_results = model.evolution_system.evolve_components()
    print(f"Component evolution results: {component_results}")
    
    # Check which parameters changed
    changed_params = 0
    for name, param in model.named_parameters():
        if not torch.equal(initial_weights[name], param.data):
            changed_params += 1
    
    print(f"Parameters changed: {changed_params}")
    
    # Test architecture evolution
    print("\nTesting architecture evolution...")
    initial_layers = len(model.layers)
    arch_results = model.evolution_system.evolve_architecture()
    final_layers = len(model.layers)
    
    print(f"Architecture evolution results: {arch_results}")
    print(f"Layer count change: {initial_layers} -> {final_layers}")

#### Evolution Success Rate
def measure_evolution_success_rate(trials=10):
    success_rates = {
        'component': 0,
        'architecture': 0,
        'paradigm': 0
    }
    
    for trial in range(trials):
        # Test component evolution
        initial_state = model.state_dict()
        component_result = model.evolution_system.evolve_components()
        if "No component changes" not in component_result:
            success_rates['component'] += 1
        
        # Test architecture evolution
        initial_arch = len(model.layers)
        arch_result = model.evolution_system.evolve_architecture()
        final_arch = len(model.layers)
        if final_arch != initial_arch or "No architectural changes" not in arch_result:
            success_rates['architecture'] += 1
        
        # Test paradigm evolution
        paradigm_result = model.evolution_system.evolve_paradigm()
        if "No paradigm changes" not in paradigm_result:
            success_rates['paradigm'] += 1
    
    for key in success_rates:
        success_rates[key] = success_rates[key] / trials
    
    print("Evolution Success Rates:")
    for key, rate in success_rates.items():
        print(f"  {key}: {rate:.2f}")
    
    return success_rates
```

### 3. Neuroplasticity Analysis

#### Layer-wise Plasticity
```python
def analyze_neuroplasticity():
    print("Neuroplasticity Analysis:")
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'connection_strength'):
            strength_stats = {
                'mean': layer.connection_strength.mean().item(),
                'std': layer.connection_strength.std().item(),
                'min': layer.connection_strength.min().item(),
                'max': layer.connection_strength.max().item()
            }
            
            # Count strong/weak connections
            strong_conns = (layer.connection_strength > 1.5).sum().item()
            weak_conns = (layer.connection_strength < 0.5).sum().item()
            total_conns = layer.connection_strength.numel()
            
            print(f"Layer {i}:")
            print(f"  Connection strength: {strength_stats}")
            print(f"  Strong connections: {strong_conns}/{total_conns} ({strong_conns/total_conns*100:.1f}%)")
            print(f"  Weak connections: {weak_conns}/{total_conns} ({weak_conns/total_conns*100:.1f}%)")
            
            # Check activation history
            if hasattr(layer, 'activation_history'):
                active_neurons = (layer.activation_history > 0.1).sum().item()
                total_neurons = layer.activation_history.numel()
                print(f"  Active neurons: {active_neurons}/{total_neurons} ({active_neurons/total_neurons*100:.1f}%)")

#### Plasticity Over Time
def track_plasticity_over_time(steps=100):
    plasticity_history = []
    
    for step in range(steps):
        # Training step
        # ... training code ...
        
        # Record plasticity metrics
        layer_plasticity = []
        for layer in model.layers:
            if hasattr(layer, 'connection_strength'):
                plasticity = {
                    'mean_strength': layer.connection_strength.mean().item(),
                    'strength_variance': layer.connection_strength.var().item(),
                    'updates': layer.evolution_tracker.get('updates', 0)
                }
                layer_plasticity.append(plasticity)
        
        plasticity_history.append({
            'step': step,
            'layers': layer_plasticity
        })
        
        # Evolve occasionally
        if step % 10 == 0:
            model.evolve()
    
    # Analyze plasticity trends
    import numpy as np
    
    for layer_idx in range(len(model.layers)):
        mean_strengths = [h['layers'][layer_idx]['mean_strength'] 
                         for h in plasticity_history if len(h['layers']) > layer_idx]
        
        if len(mean_strengths) > 10:
            trend = np.polyfit(range(len(mean_strengths)), mean_strengths, 1)[0]
            print(f"Layer {layer_idx} plasticity trend: {trend:.6f}")
```

### 4. Hive Mind Debugging

#### Sync Status Monitoring
```python
def debug_hive_mind():
    if not model.config.hive_enabled or not model.hive_synchronizer:
        print("Hive mind not enabled")
        return
    
    sync_stats = model.hive_synchronizer.get_stats()
    print("Hive Mind Status:")
    print(f"  Identity: {sync_stats['identity']}")
    print(f"  Sync active: {sync_stats['sync_active']}")
    print(f"  Total syncs: {sync_stats['total_syncs']}")
    print(f"  Concepts received: {sync_stats['concepts_received']}")
    print(f"  Concepts sent: {sync_stats['concepts_sent']}")
    
    # Check sync health
    if sync_stats['total_syncs'] == 0:
        print("WARNING: No successful syncs yet")
    
    last_sync_time = sync_stats.get('last_sync', 0)
    if time.time() - last_sync_time > 600:  # 10 minutes
        print("WARNING: Last sync was over 10 minutes ago")
    
    # Check concept sharing
    shared_concepts = len(model.concept_bank.hive_shared_concepts)
    private_concepts = len(model.concept_bank.hive_private_concepts)
    total_concepts = model.concept_bank.next_concept_id
    
    print(f"\nConcept Sharing:")
    print(f"  Shared: {shared_concepts}/{total_concepts} ({shared_concepts/total_concepts*100:.1f}%)")
    print(f"  Private: {private_concepts}/{total_concepts} ({private_concepts/total_concepts*100:.1f}%)")

#### Test Sync Functionality
def test_sync_functionality():
    if not model.hive_synchronizer:
        print("Hive synchronizer not available")
        return
    
    # Create test concepts
    test_texts = ["test concept 1", "test concept 2", "unique test phrase"]
    
    initial_concepts = model.concept_bank.next_concept_id
    
    for text in test_texts:
        model.process_text(text)
    
    new_concepts = model.concept_bank.next_concept_id - initial_concepts
    print(f"Created {new_concepts} test concepts")
    
    # Check if they're marked for sync
    pending_sync = len(model.concept_bank.hive_pending_sync)
    print(f"Concepts pending sync: {pending_sync}")
    
    # Test sync preparation
    concepts_to_share = model.hive_synchronizer._prepare_concepts_for_sharing()
    print(f"Prepared {len(concepts_to_share)} concepts for sharing")
```

## 🧩 Extension Development

### Creating Custom Layers
```python
class CustomNeuroplasticLayer(AdvancedNeuroplasticLayer):
    def __init__(self, hidden_dim, custom_param=1.0, **kwargs):
        super().__init__(hidden_dim, **kwargs)
        self.custom_param = custom_param
        
        # Add custom components
        self.custom_transform = nn.Linear(hidden_dim, hidden_dim)
        self.custom_gate = nn.Sigmoid()
    
    def forward(self, x, mask=None, context=None, modality="text"):
        # Call parent forward
        output = super().forward(x, mask, context, modality)
        
        # Apply custom transformation
        custom_output = self.custom_transform(output)
        gate = self.custom_gate(custom_output)
        
        # Blend with original output
        final_output = gate * custom_output + (1 - gate) * output
        
        return final_output
    
    def evolve(self):
        # Call parent evolution
        parent_result = super().evolve()
        
        # Add custom evolution logic
        with torch.no_grad():
            # Example: adjust custom parameter based on usage
            if hasattr(self, 'usage_counter'):
                if self.usage_counter > 100:
                    self.custom_param *= 1.1
                    self.usage_counter = 0
        
        custom_result = {"custom_param": self.custom_param}
        
        if isinstance(parent_result, dict):
            parent_result.update(custom_result)
            return parent_result
        else:
            return custom_result

# Replace layers in existing model
def upgrade_model_layers():
    new_layers = nn.ModuleList()
    
    for i, old_layer in enumerate(model.layers):
        new_layer = CustomNeuroplasticLayer(
            hidden_dim=old_layer.hidden_dim,
            layer_id=i,
            custom_param=1.5
        )
        
        # Transfer weights where possible
        with torch.no_grad():
            if hasattr(old_layer, 'norm1'):
                new_layer.norm1.load_state_dict(old_layer.norm1.state_dict())
            if hasattr(old_layer, 'norm2'):
                new_layer.norm2.load_state_dict(old_layer.norm2.state_dict())
        
        new_layers.append(new_layer)
    
    model.layers = new_layers
    print(f"Upgraded {len(new_layers)} layers")
```

### Custom Evolution Strategies
```python
class DomainSpecificEvolution(MultiLevelEvolutionSystem):
    def __init__(self, model, domain_knowledge=None):
        super().__init__(model)
        self.domain_knowledge = domain_knowledge or {}
        self.domain_concepts = set()
        
        # Load domain-specific patterns
        self._load_domain_patterns()
    
    def _load_domain_patterns(self):
        # Example for scientific domain
        if "scientific" in self.domain_knowledge:
            scientific_terms = [
                "hypothesis", "experiment", "analysis", "conclusion",
                "method", "result", "significant", "correlation"
            ]
            
            for term in scientific_terms:
                concept_id = self.model.concept_bank.find_concept_by_source(term)
                if concept_id:
                    self.domain_concepts.add(concept_id)
    
    def evolve_components(self):
        # Standard evolution
        base_changes = super().evolve_components()
        
        # Domain-specific component evolution
        domain_changes = self._evolve_domain_components()
        
        return f"{base_changes}; {domain_changes}"
    
    def _evolve_domain_components(self):
        changes = []
        
        # Strengthen domain-relevant connections
        for layer in self.model.layers:
            if hasattr(layer, 'connection_strength'):
                # Boost connections related to domain concepts
                domain_boost = self._calculate_domain_relevance(layer)
                if domain_boost > 0:
                    layer.connection_strength *= (1.0 + domain_boost * 0.1)
                    changes.append(f"Boosted domain connections in layer {layer.layer_id}")
        
        return "; ".join(changes) if changes else "No domain-specific changes"
    
    def _calculate_domain_relevance(self, layer):
        # Calculate how relevant this layer is to domain concepts
        if not hasattr(layer, 'activation_history'):
            return 0.0
        
        # Simplified relevance calculation
        activation_mean = layer.activation_history.mean().item()
        
        # Check if domain concepts are frequently activated
        domain_activation = 0.0
        for concept_id in self.domain_concepts:
            if concept_id < len(self.model.concept_bank.concept_frequencies):
                freq = self.model.concept_bank.concept_frequencies[concept_id].item()
                domain_activation += freq
        
        # Normalize and return relevance score
        if len(self.domain_concepts) > 0:
            domain_activation /= len(self.domain_concepts)
            relevance = min(1.0, domain_activation / 100.0)  # Normalize to 0-1
            return relevance * activation_mean
        
        return 0.0

# Apply domain-specific evolution
scientific_evolution = DomainSpecificEvolution(
    model, 
    domain_knowledge={"scientific": True}
)
model.evolution_system = scientific_evolution
```

### Custom Consciousness Metrics
```python
class EnhancedConsciousnessMonitor(ConsciousnessMonitor):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        
        # Additional consciousness metrics
        self.creativity_history = []
        self.learning_rate_history = []
        self.adaptation_speed_history = []
    
    def update(self, modality="text"):
        # Get base consciousness state
        base_state = super().update(modality)
        
        if base_state:
            # Add enhanced metrics
            creativity = self._measure_creativity()
            learning_rate = self._measure_learning_rate()
            adaptation_speed = self._measure_adaptation_speed()
            
            # Update histories
            self.creativity_history.append(creativity)
            self.learning_rate_history.append(learning_rate)
            self.adaptation_speed_history.append(adaptation_speed)
            
            # Limit history size
            max_history = 100
            for history in [self.creativity_history, self.learning_rate_history, self.adaptation_speed_history]:
                if len(history) > max_history:
                    history[:] = history[-max_history:]
            
            # Enhanced consciousness score
            enhanced_score = (
                base_state['consciousness_score'] * 0.6 +
                creativity * 0.2 +
                learning_rate * 0.1 +
                adaptation_speed * 0.1
            )
            
            base_state.update({
                'creativity': creativity,
                'learning_rate': learning_rate,
                'adaptation_speed': adaptation_speed,
                'enhanced_consciousness_score': enhanced_score
            })
        
        return base_state
    
    def _measure_creativity(self):
        # Measure creative concept combinations
        recent_dreams = self.model.dreaming.dream_history[-5:]  # Last 5 dreams
        
        if not recent_dreams:
            return 0.0
        
        # Analyze novelty in dream content
        total_novelty = 0.0
        for dream in recent_dreams:
            content = dream.get('content', '')
            concept_ids, _ = self.model.process_text(content)
            
            # Calculate concept novelty
            novelty_score = 0.0
            for concept_id in concept_ids:
                if concept_id < len(self.model.concept_bank.concept_frequencies):
                    freq = self.model.concept_bank.concept_frequencies[concept_id].item()
                    # Novel concepts have low frequency
                    novelty_score += 1.0 / (1.0 + freq)
            
            if len(concept_ids) > 0:
                total_novelty += novelty_score / len(concept_ids)
        
        creativity = total_novelty / len(recent_dreams)
        return min(1.0, creativity)
    
    def _measure_learning_rate(self):
        # Measure how quickly new concepts are being formed
        if len(self.model.concept_bank.creation_history) < 10:
            return 0.5
        
        recent_creations = self.model.concept_bank.creation_history[-10:]
        time_span = recent_creations[-1]['timestamp'] - recent_creations[0]['timestamp']
        
        if time_span > 0:
            creation_rate = len(recent_creations) / time_span
            # Normalize to 0-1 range
            normalized_rate = min(1.0, creation_rate * 60)  # concepts per minute
            return normalized_rate
        
        return 0.0
    
    def _measure_adaptation_speed(self):
        # Measure how quickly the model adapts to new patterns
        if len(self.model.growth_history) < 2:
            return 0.5
        
        recent_growth = self.model.growth_history[-5:]  # Last 5 growth events
        
        if len(recent_growth) < 2:
            return 0.5
        
        # Calculate time between growth events
        time_diffs = []
        for i in range(1, len(recent_growth)):
            time_diff = recent_growth[i]['timestamp'] - recent_growth[i-1]['timestamp']
            time_diffs.append(time_diff)
        
        if time_diffs:
            avg_time_diff = sum(time_diffs) / len(time_diffs)
            # Faster adaptation = higher score
            adaptation_speed = 1.0 / (1.0 + avg_time_diff / 3600)  # Normalize by hours
            return min(1.0, adaptation_speed)
        
        return 0.5

# Replace consciousness monitor
enhanced_monitor = EnhancedConsciousnessMonitor(
    model,
    stability_threshold=0.95,
    novelty_weight=0.4
)
model.consciousness = enhanced_monitor
```

## 🔍 Diagnostic Tools

### Model Health Check
```python
def comprehensive_health_check():
    """Run comprehensive model health diagnostics"""
    print("🏥 SAM Health Check")
    print("=" * 50)
    
    issues = []
    warnings = []
    
    # 1. Memory health
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        if memory_used > 0.9:
            issues.append("High GPU memory usage")
        elif memory_used > 0.7:
            warnings.append("Moderate GPU memory usage")
    
    # 2. Parameter health
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if trainable_params == 0:
        issues.append("No trainable parameters")
    
    # 3. Gradient health
    grad_norms = []
    nan_grads = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if torch.isnan(param.grad).any():
                nan_grads += 1
    
    if nan_grads > 0:
        issues.append(f"NaN gradients in {nan_grads} parameters")
    
    if grad_norms:
        max_grad = max(grad_norms)
        if max_grad > 100:
            issues.append(f"Exploding gradients (max: {max_grad:.2f})")
        elif max_grad < 1e-8:
            warnings.append("Very small gradients (vanishing gradients?)")
    
    # 4. Concept bank health
    concept_stats = model.concept_bank.get_concept_stats()
    if concept_stats['total_concepts'] < 100:
        warnings.append("Low concept count")
    
    # 5. Evolution health
    if hasattr(model, 'evolution_system'):
        evolution_stats = model.evolution_system.get_evolution_stats()
        if evolution_stats['component_mutations'] == 0:
            warnings.append("No component mutations occurred")
    
    # 6. Consciousness health
    consciousness = model.consciousness.get_consciousness_summary()
    if consciousness['consciousness_score'] < 0.3:
        warnings.append("Low consciousness score")
    
    # Report results
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Concepts: {concept_stats['total_concepts']}")
    print(f"Consciousness: {consciousness['level']} ({consciousness['consciousness_score']:.3f})")
    
    if issues:
        print(f"\n❌ Issues Found: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print(f"\n⚠️  Warnings: {len(warnings)}")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues and not warnings:
        print("\n✅ Model appears healthy!")
    
    return {'issues': issues, 'warnings': warnings}

### Performance Benchmarking
def run_performance_benchmark():
    """Benchmark model performance"""
    print("🏃 Performance Benchmark")
    print("=" * 30)
    
    import time
    
    # Forward pass benchmark
    dummy_input = torch.randint(0, 1000, (4, 512), device=model.device)
    
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(input_concepts=dummy_input)
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            _ = model(input_concepts=dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        forward_time = (time.time() - start_time) / 100
    
    # Generation benchmark
    start_time = time.time()
    generated = model.generate("Hello", max_length=50)
    generation_time = time.time() - start_time
    
    # Evolution benchmark
    start_time = time.time()
    evolution_results = model.evolve()
    evolution_time = time.time() - start_time
    
    print(f"Forward pass: {forward_time*1000:.2f} ms")
    print(f"Generation (50 tokens): {generation_time:.2f} s")
    print(f"Evolution cycle: {evolution_time:.2f} s")
    print(f"Generated text: {generated[:100]}...")
    
    return {
        'forward_time_ms': forward_time * 1000,
        'generation_time_s': generation_time,
        'evolution_time_s': evolution_time
    }

# Export diagnostic tools
def export_diagnostic_report():
    """Export comprehensive diagnostic report"""
    report = {
        'timestamp': time.time(),
        'health_check': comprehensive_health_check(),
        'performance_benchmark': run_performance_benchmark(),
        'model_stats': model.get_stats(),
        'config': vars(model.config)
    }
    
    import json
    with open(f"diagnostic_report_{int(time.time())}.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📋 Diagnostic report exported")
    return report
```

This comprehensive debugging guide provides tools for troubleshooting common issues, optimizing performance, and extending SAM's capabilities. Use these diagnostic functions regularly during development to maintain model health and identify potential improvements.
