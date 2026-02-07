#!/usr/bin/env python3
"""
GPU Necromancer - Universal Quick Start Examples
Shows how to use the universal multi-GPU support system
"""

import logging
from necromancer_universal_agent import UniversalNecromancerAgent, ModelRequest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_detection():
    """Example 1: Basic GPU Detection"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic GPU Detection")
    print("="*80)
    
    from necromancer_universal_detectors import UnifiedGPUDetector
    
    # Create detector
    detector = UnifiedGPUDetector()
    
    # Detect all GPUs
    devices = detector.detect_all_devices()
    
    print(f"\nDetected {len(devices)} device(s):")
    for i, device in enumerate(devices):
        print(f"\n[{i}] {device.name}")
        print(f"    Vendor: {device.vendor.value}")
        print(f"    Memory: {device.total_memory_gb:.1f}GB")
        print(f"    Compute Units: {device.max_compute_units}")


def example_2_strategy_generation():
    """Example 2: Automatic Strategy Generation"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Automatic Strategy Generation")
    print("="*80)
    
    from necromancer_universal_detectors import UnifiedGPUDetector
    from necromancer_universal_strategies import StrategyGenerator
    
    # Get optimal device
    detector = UnifiedGPUDetector()
    gpu = detector.get_optimal_device()
    
    print(f"\nSelected GPU: {gpu.name}")
    
    # Generate strategy
    strategy = StrategyGenerator.generate_strategy(gpu)
    
    print(f"\nStrategy Generated:")
    print(f"  Backend: {strategy.recommended_backend}")
    print(f"  Compute API: {strategy.recommended_compute_api}")
    print(f"  Attention: {strategy.attention_mechanism}")
    print(f"  Quantization: {strategy.max_quantization_bits}-bit")
    print(f"  Max Model: ~{strategy.estimated_max_model_size_b:.0f}B")
    
    if strategy.optimizations:
        print(f"\n  Optimizations:")
        for opt in strategy.optimizations:
            print(f"    ✓ {opt}")
    
    if strategy.limitations:
        print(f"\n  Limitations:")
        for lim in strategy.limitations:
            print(f"    • {lim}")


def example_3_agent_creation():
    """Example 3: Create Universal Agent"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Create Universal Agent")
    print("="*80)
    
    # Create agent - automatically detects best GPU
    agent = UniversalNecromancerAgent()
    
    # Display strategy
    agent.display_strategy()
    
    # Get status
    status = agent.get_status_report()
    print(f"\nAgent Status:")
    print(f"  GPU: {status['gpu']}")
    print(f"  Vendor: {status['vendor']}")
    print(f"  Executions: {status['total_executions']}")


def example_4_model_preparation():
    """Example 4: Prepare Model Request"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Prepare Model Request")
    print("="*80)
    
    agent = UniversalNecromancerAgent()
    
    # Create model request
    request = ModelRequest(
        model_name="mistral-7b",
        task="chat",
        context_length=4096,
        max_tokens=512,
        temperature=0.7,
    )
    
    print(f"\nModel Request:")
    print(f"  Model: {request.model_name}")
    print(f"  Task: {request.task}")
    print(f"  Context: {request.context_length}")
    
    # Prepare configuration
    config = agent.prepare_model(request)
    
    print(f"\nPrepared Config:")
    print(f"  Quantization: {config['quantization_bits']}-bit")
    print(f"  Attention: {config['attention_mechanism']}")
    print(f"  Batch Size: {config['batch_size']}")


def example_5_vendor_preference():
    """Example 5: Specify Vendor Preference"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Vendor Preference")
    print("="*80)
    
    from necromancer_universal_detectors import UnifiedGPUDetector
    
    detector = UnifiedGPUDetector()
    devices = detector.detect_all_devices()
    
    print(f"\nAvailable vendors:")
    for device in devices:
        print(f"  - {device.vendor.value} ({device.name})")
    
    # Try to prefer CPU (always available)
    print(f"\nCreating agent with CPU preference...")
    agent = UniversalNecromancerAgent(vendor_preference="cpu")
    print(f"✅ Created with {agent.gpu.name}")


def example_6_all_devices():
    """Example 6: List All Detected Devices"""
    print("\n" + "="*80)
    print("EXAMPLE 6: List All Devices")
    print("="*80)
    
    agent = UniversalNecromancerAgent()
    agent.list_all_devices()


def example_7_inference_backend_selection():
    """Example 7: Inference Backend Selection"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Inference Backend Selection")
    print("="*80)
    
    from necromancer_universal_detectors import UnifiedGPUDetector
    from necromancer_universal_backends import UnifiedInferenceEngine
    
    detector = UnifiedGPUDetector()
    gpu = detector.get_optimal_device()
    
    print(f"\nGPU: {gpu.name}")
    
    # Create inference engine
    engine = UnifiedInferenceEngine(gpu)
    
    try:
        backend = engine.select_backend()
        print(f"Selected Backend: {backend.name}")
        print(f"Available Quantizations: {backend.supported_quantizations}")
    except Exception as e:
        print(f"No backends available: {e}")


def run_all_examples():
    """Run all examples"""
    try:
        example_1_basic_detection()
    except Exception as e:
        print(f"Example 1 error: {e}")
    
    try:
        example_2_strategy_generation()
    except Exception as e:
        print(f"Example 2 error: {e}")
    
    try:
        example_3_agent_creation()
    except Exception as e:
        print(f"Example 3 error: {e}")
    
    try:
        example_4_model_preparation()
    except Exception as e:
        print(f"Example 4 error: {e}")
    
    try:
        example_5_vendor_preference()
    except Exception as e:
        print(f"Example 5 error: {e}")
    
    try:
        example_6_all_devices()
    except Exception as e:
        print(f"Example 6 error: {e}")
    
    try:
        example_7_inference_backend_selection()
    except Exception as e:
        print(f"Example 7 error: {e}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧟 GPU NECROMANCER - UNIVERSAL QUICK START EXAMPLES")
    print("="*80)
    
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_1_basic_detection()
        elif example_num == "2":
            example_2_strategy_generation()
        elif example_num == "3":
            example_3_agent_creation()
        elif example_num == "4":
            example_4_model_preparation()
        elif example_num == "5":
            example_5_vendor_preference()
        elif example_num == "6":
            example_6_all_devices()
        elif example_num == "7":
            example_7_inference_backend_selection()
        else:
            print(f"Unknown example: {example_num}")
    else:
        run_all_examples()
    
    print("\n" + "="*80)
    print("✅ Examples completed!")
    print("="*80 + "\n")

