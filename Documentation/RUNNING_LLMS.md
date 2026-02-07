# 🧟 GPU Necromancer - Running LLMs Complete Guide

## Quick Answer

To run LLMs with GPU Necromancer:

```python
from necromancer_universal_agent import UniversalNecromancerAgent
from necromancer_universal_agent import ModelRequest

# 1. Create agent
agent = UniversalNecromancerAgent()

# 2. Create request
request = ModelRequest(
    model_name="mistral-7b",
    max_tokens=512,
    temperature=0.7
)

# 3. Run
result = agent.run_with_healing(request)
print(result['output'])
```

---

## 📋 Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements_universal.txt

# Essential:
pip install pynvml psutil numpy llama-cpp-python

# For ONNX support:
pip install onnxruntime

# For Intel GPU support:
pip install openvino  # Optional
```

### 2. Download a Model

Models need to be in **GGUF format** for llama-cpp:

```bash
# Option A: Use HuggingFace (fastest)
pip install huggingface-hub

python -c "
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id='TheBloke/Mistral-7B-Instruct-v0.1-GGUF',
    filename='mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    cache_dir='./models'
)
print(f'Downloaded to: {model_path}')
"

# Option B: Manual Download
# Visit: https://huggingface.co/TheBloke
# Download a GGUF model (Q4_K_M quantization recommended)
# Save to: ./models/

# Option C: Use Ollama (simplest)
ollama pull mistral:7b
```

### 3. Find Your Model Path
```bash
ls -lh ./models/
# Example output:
# mistral-7b-instruct-v0.1.Q4_K_M.gguf  (4.4GB)
```

---

## 🎯 Complete Example - Running Inference

### Basic Chat
```python
from necromancer_universal_agent import UniversalNecromancerAgent, ModelRequest

# Initialize
agent = UniversalNecromancerAgent()
agent.display_strategy()

# Define your model and request
request = ModelRequest(
    model_name="mistral-7b",
    task="chat",
    max_tokens=256,
    temperature=0.7
)

# Run
result = agent.run_with_healing(request)
print(f"Output: {result['output']}")
```

### Text Completion
```python
request = ModelRequest(
    model_name="mistral-7b",
    task="completion",
    max_tokens=512,
    temperature=0.5,
    top_p=0.95
)

result = agent.run_with_healing(request)
print(result['output'])
```

---

## 🛠️ Advanced Configuration

### Custom Model Path
```python
from necromancer_universal_agent import UniversalNecromancerAgent

agent = UniversalNecromancerAgent()

# Tell agent where your model is
model_path = "/path/to/mistral-7b.gguf"

# Load model
model = agent.inference_engine.load_model(
    model_path=model_path,
    quantization_bits=agent.strategy.max_quantization_bits
)

print(f"✅ Loaded model with {agent.strategy.max_quantization_bits}-bit quantization")
```

### Specific GPU Selection
```python
# Use GPU at index 1
agent = UniversalNecromancerAgent(device_id=1)

# Prefer Intel GPU
agent = UniversalNecromancerAgent(vendor_preference="intel_integrated")

# Prefer CPU
agent = UniversalNecromancerAgent(vendor_preference="cpu")
```

### Custom Inference Parameters
```python
request = ModelRequest(
    model_name="mistral-7b",
    task="chat",
    context_length=8192,  # Long context
    max_tokens=2048,      # Long output
    temperature=0.1,      # Deterministic
    top_p=0.9
)
```

---

## 📊 Recommended Model Sizes

Based on your GPU VRAM:

| VRAM | Model | Quantization | Speed |
|------|-------|--------------|-------|
| **3-4GB** | 3B (phi, llama) | 4-bit | Fast |
| **6-8GB** | 7B (mistral, llama) | 4-5 bit | Good |
| **12-16GB** | 13B (neural-chat) | 5-6 bit | Good |
| **20GB+** | 30B+ (solar, yi) | 8-bit | Okay |
| **CPU** | Any size | 4-bit quant | Slow |

**Always use quantized models (Q4_K_M recommended)**

---

## 🚀 Step-by-Step: From Zero to Running

### Step 1: Install
```bash
pip install -r requirements_universal.txt
```

### Step 2: Download Model
```bash
mkdir -p ./models

python << 'PYTHON'
from huggingface_hub import hf_hub_download
import os

# Download Mistral 7B (4.4GB)
model_path = hf_hub_download(
    repo_id='TheBloke/Mistral-7B-Instruct-v0.1-GGUF',
    filename='mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    cache_dir='./models'
)
print(f"✅ Model downloaded: {model_path}")
PYTHON
```

### Step 3: Check GPU
```bash
python necromancer_universal_agent.py --list-devices
```

Output example:
```
[0] NVIDIA RTX 4090
    Vendor: nvidia
    Memory: 24.0GB
    Compute: 330 TFLOPS

[1] Intel UHD 770
    Vendor: intel_integrated
    Memory: 8.0GB
    Compute: 10 TFLOPS

[2] CPU (16 cores)
    Vendor: cpu
    Memory: 64.0GB
    Compute: 51 TFLOPS
```

### Step 4: Test Strategy
```bash
python necromancer_universal_agent.py --show-strategy
```

Output example:
```
GPU SPECIFICATIONS:
   Name: NVIDIA RTX 4090
   Vendor: nvidia
   Memory: 24.0GB
   Performance: 330 TFLOPS (FP32)

RECOMMENDED SETTINGS:
   Backend: llama-cpp
   Quantization: 16-bit
   Max Model: ~70B parameters
   Attention: flash_attention_2
```

### Step 5: Run Inference
```python
from necromancer_universal_agent import UniversalNecromancerAgent, ModelRequest

# Create agent
agent = UniversalNecromancerAgent()

# Load model
MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
model = agent.inference_engine.load_model(
    model_path=MODEL_PATH,
    quantization_bits=4
)

# Define request
request = ModelRequest(
    model_name="mistral-7b",
    max_tokens=512,
    temperature=0.7
)

# Prepare config
config = agent.prepare_model(request)

# Execute
result = agent.execute_model(config)
print(result['output'])
```

---

## 🤖 Complete Inference Script

Create `run_llm.py`:

```python
#!/usr/bin/env python3
"""
GPU Necromancer - Complete LLM Inference Script
"""

import argparse
import sys
from necromancer_universal_agent import UniversalNecromancerAgent, ModelRequest

def main():
    parser = argparse.ArgumentParser(description="Run LLM with GPU Necromancer")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--tokens", type=int, default=256, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--device", type=int, default=None, help="GPU device ID")
    parser.add_argument("--vendor", default=None, help="Preferred vendor")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🧟 GPU NECROMANCER - LLM INFERENCE")
    print("="*80 + "\n")
    
    # Create agent with GPU selection
    print("📊 Initializing GPU Necromancer...")
    agent = UniversalNecromancerAgent(
        device_id=args.device,
        vendor_preference=args.vendor
    )
    
    # Display strategy
    print(f"\n✅ Using: {agent.gpu.name}")
    print(f"   Strategy: {agent.strategy.recommended_backend}")
    print(f"   Quantization: {agent.strategy.max_quantization_bits}-bit")
    
    # Load model
    print(f"\n📥 Loading model: {args.model}")
    try:
        model = agent.inference_engine.load_model(
            model_path=args.model,
            quantization_bits=agent.strategy.max_quantization_bits
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1
    
    # Run inference
    print(f"\n💬 Running inference...")
    print(f"   Prompt: {args.prompt[:50]}...")
    print(f"   Max tokens: {args.tokens}")
    
    # Create request
    request = ModelRequest(
        model_name="inference",
        max_tokens=args.tokens,
        temperature=args.temperature
    )
    
    # Execute
    try:
        config = agent.prepare_model(request)
        result = agent.execute_model(config)
        
        print("\n" + "="*80)
        print("🧟 OUTPUT")
        print("="*80)
        print(result['output'])
        print("="*80 + "\n")
        
        return 0
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

Run it:
```bash
python run_llm.py \
    --model ./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf \
    --prompt "What is machine learning?" \
    --tokens 256
```

---

## 🎨 Interactive Chat Script

Create `chat.py`:

```python
#!/usr/bin/env python3
"""
Interactive chat with GPU Necromancer
"""

from necromancer_universal_agent import UniversalNecromancerAgent, ModelRequest
import sys

def main():
    MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    
    print("\n" + "="*80)
    print("🧟 GPU NECROMANCER - INTERACTIVE CHAT")
    print("="*80)
    print("Type 'quit' to exit\n")
    
    # Initialize
    agent = UniversalNecromancerAgent()
    print(f"Using: {agent.gpu.name}\n")
    
    # Load model once
    try:
        model = agent.inference_engine.load_model(
            model_path=MODEL_PATH,
            quantization_bits=agent.strategy.max_quantization_bits
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 1
    
    # Chat loop
    conversation_history = []
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt.lower() == 'quit':
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            # Build context from history
            context = "\n".join(conversation_history[-5:])  # Last 5 messages
            full_prompt = f"{context}\n{prompt}" if context else prompt
            
            # Create request
            request = ModelRequest(
                model_name="chat",
                max_tokens=256,
                temperature=0.7
            )
            
            # Execute
            config = agent.prepare_model(request)
            result = agent.execute_model(config)
            
            response = result['output']
            print(f"\nAssistant: {response}\n")
            
            # Track history
            conversation_history.append(f"You: {prompt}")
            conversation_history.append(f"Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python chat.py
```

---

## 🔧 Model Configuration Tips

### For Integrated GPU (Intel UHD 770)
```python
agent = UniversalNecromancerAgent(vendor_preference="intel_integrated")

# Strategy will automatically:
# - Use unified memory
# - Apply 4-bit quantization
# - Limit to smaller models
# - Warn about bandwidth

request = ModelRequest(
    model_name="mistral-3b",  # Smaller model
    max_tokens=128,           # Shorter output
)
```

### For CPU-Only
```python
agent = UniversalNecromancerAgent(vendor_preference="cpu")

# Can use larger models due to lots of RAM
# But will be SLOW (5-20 tokens/sec)
request = ModelRequest(
    model_name="mistral-7b",
    max_tokens=512,
)
```

### For High-End GPU (RTX 4090)
```python
agent = UniversalNecromancerAgent()  # Auto-selects RTX 4090

# Can run large models with good precision
request = ModelRequest(
    model_name="llama2-70b",
    context_length=8192,
    max_tokens=2048,
)
```

---

## 📥 Getting Models

### Option 1: Hugging Face Hub (Easiest)
```bash
pip install huggingface-hub

python << 'PYTHON'
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id='TheBloke/Mistral-7B-Instruct-v0.1-GGUF',
    filename='mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    cache_dir='./models'
)
print(f"Downloaded: {model_path}")
PYTHON
```

### Option 2: Ollama (Simplest UI)
```bash
# Install: https://ollama.ai

ollama pull mistral:7b
ollama pull llama2:7b
ollama pull neural-chat:7b

# Models saved to:
# ~/.ollama/models/
```

### Option 3: Manual Download
1. Visit https://huggingface.co/TheBloke
2. Find a GGUF model (Q4_K_M quantization)
3. Download to `./models/`

### Recommended Models

| Model | Size | VRAM | Tokens/Sec | Link |
|-------|------|------|------------|------|
| Phi 2.7B | 1.6GB | 3GB | Fast | TheBloke/phi-2-GGUF |
| Mistral 7B | 4.4GB | 6GB | Good | TheBloke/Mistral-7B-Instruct-GGUF |
| Neural Chat 7B | 4.1GB | 6GB | Good | TheBloke/neural-chat-7B-GGUF |
| Llama 2 13B | 7.4GB | 10GB | Okay | TheBloke/Llama-2-13B-chat-GGUF |

---

## ⚡ Performance Optimization

### Speed Up Inference
```python
request = ModelRequest(
    model_name="mistral-7b",
    temperature=0.1,        # Faster (less sampling)
    max_tokens=128,         # Shorter output
)

# Use smaller quantization (faster but less accurate)
model = agent.inference_engine.load_model(
    model_path="model.gguf",
    quantization_bits=4  # 4-bit fastest
)
```

### Better Quality
```python
request = ModelRequest(
    model_name="mistral-7b",
    temperature=0.7,        # More creative
    max_tokens=512,         # Longer output
    top_p=0.95,            # Diverse sampling
)

# Use higher quantization (slower but better quality)
model = agent.inference_engine.load_model(
    model_path="model.gguf",
    quantization_bits=8  # 8-bit better quality
)
```

---

## 🐛 Troubleshooting

### "Model not found"
```bash
# Check model path
ls -lh ./models/

# Make sure you downloaded a GGUF model
# Not PyTorch (.pt) or Safetensors (.safetensors)
```

### "Out of memory"
```python
# Use smaller model or more quantization
request = ModelRequest(
    model_name="smaller-model",
    max_tokens=128,  # Shorter output
)

# Or use 4-bit quantization
model = agent.inference_engine.load_model(
    model_path="model.gguf",
    quantization_bits=4
)
```

### "Low performance on integrated GPU"
```python
# This is expected! Integrated GPUs have low bandwidth
# Try using smaller model or CPU
agent = UniversalNecromancerAgent(vendor_preference="cpu")

# Or accept slow speed
request = ModelRequest(model_name="phi-2.7b", max_tokens=64)
```

### "CUDA/Driver error"
```bash
# Update NVIDIA drivers
nvidia-driver-update  # Or similar for your system

# Reinstall pynvml
pip install --upgrade pynvml

# Check NVIDIA is working
nvidia-smi
```

---

## 📊 Example: Full Working Script

```python
#!/usr/bin/env python3
"""
Complete working example - runs LLM on any GPU
"""

from necromancer_universal_agent import UniversalNecromancerAgent, ModelRequest

# Configuration
MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
PROMPT = "Explain quantum computing in simple terms."

def main():
    # 1. Initialize GPU Necromancer
    print("🧟 Initializing GPU Necromancer...")
    agent = UniversalNecromancerAgent()
    
    # 2. Display GPU info
    print(f"\n📊 GPU Info:")
    print(f"   Name: {agent.gpu.name}")
    print(f"   Vendor: {agent.gpu.vendor.value}")
    print(f"   Memory: {agent.gpu.total_memory_gb:.1f}GB")
    
    # 3. Display strategy
    print(f"\n⚙️ Strategy:")
    print(f"   Backend: {agent.strategy.recommended_backend}")
    print(f"   Quantization: {agent.strategy.max_quantization_bits}-bit")
    print(f"   Max Model: {agent.strategy.estimated_max_model_size_b:.0f}B")
    
    # 4. Load model
    print(f"\n📥 Loading model: {MODEL_PATH}")
    model = agent.inference_engine.load_model(
        model_path=MODEL_PATH,
        quantization_bits=agent.strategy.max_quantization_bits
    )
    print("✅ Model loaded!")
    
    # 5. Run inference
    print(f"\n💬 Running inference...")
    print(f"   Prompt: {PROMPT}")
    
    request = ModelRequest(
        model_name="mistral-7b",
        max_tokens=256,
        temperature=0.7
    )
    
    config = agent.prepare_model(request)
    result = agent.execute_model(config)
    
    # 6. Display output
    print(f"\n✅ Output:")
    print(f"   {result['output']}")

if __name__ == "__main__":
    main()
```

---

## 🎯 Summary: How to Run LLMs

1. **Install**: `pip install -r requirements_universal.txt`
2. **Download Model**: Use HuggingFace Hub or Ollama
3. **Initialize Agent**: `agent = UniversalNecromancerAgent()`
4. **Load Model**: `model = agent.inference_engine.load_model(model_path)`
5. **Run**: Create `ModelRequest` and execute
6. **Get Output**: Read from `result['output']`

That's it! 🚀

---

