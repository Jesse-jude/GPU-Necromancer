#!/usr/bin/env python3
"""
🧟 GPU Necromancer - Simple LLM Runner
Simplest possible way to run an LLM with GPU Necromancer
"""

from necromancer_universal_agent import UniversalNecromancerAgent, ModelRequest

def main():
    """Complete working example"""
    
    # Configuration
    MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    PROMPT = "What is artificial intelligence?"
    
    print("\n" + "="*80)
    print(" GPU NECROMANCER - SIMPLE LLM RUNNER")
    print("="*80 + "\n")
    
    # 1. Initialize - auto-detects best GPU
    print(" Detecting GPU...")
    agent = UniversalNecromancerAgent()
    print(f" Using: {agent.gpu.name}")
    print(f"   Vendor: {agent.gpu.vendor.value}")
    print(f"   Memory: {agent.gpu.total_memory_gb:.1f}GB")
    
    # 2. Show strategy
    print(f"\n⚙️ Strategy:")
    print(f"   Backend: {agent.strategy.recommended_backend}")
    print(f"   Quantization: {agent.strategy.max_quantization_bits}-bit")
    print(f"   Max Model: {agent.strategy.estimated_max_model_size_b:.0f}B parameters")
    
    # 3. Load model
    print(f"\n Loading model: {MODEL_PATH}")
    try:
        model = agent.inference_engine.load_model(
            model_path=MODEL_PATH,
            quantization_bits=agent.strategy.max_quantization_bits
        )
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print(f" Model not found: {MODEL_PATH}")
        print(f" Download it first:")
        print(f"   pip install huggingface-hub")
        print(f"   python -c \"")
        print(f"from huggingface_hub import hf_hub_download")
        print(f"hf_hub_download(")
        print(f"    repo_id='TheBloke/Mistral-7B-Instruct-v0.1-GGUF',")
        print(f"    filename='mistral-7b-instruct-v0.1.Q4_K_M.gguf',")
        print(f"    cache_dir='./models'")
        print(f")")
        print(f"   \"")
        return 1
    except Exception as e:
        print(f" Error loading model: {e}")
        return 1
    
    # 4. Prepare request
    print(f"\n Prompt: {PROMPT}")
    request = ModelRequest(
        model_name="mistral-7b",
        task="chat",
        max_tokens=256,
        temperature=0.7
    )
    
    # 5. Execute
    print(f"   Generating response...")
    try:
        config = agent.prepare_model(request)
        result = agent.execute_model(config)
        
        # 6. Display output
        print(f"\n" + "="*80)
        print(" OUTPUT")
        print("="*80)
        print(f"\n{result['output']}\n")
        print("="*80 + "\n")
        
        return 0
    
    except Exception as e:
        print(f" Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
