"""
GPU Necromancer - Universal Strategy Generator
Vendor-agnostic strategy generation for all GPU types
"""

import logging
from dataclasses import asdict
from typing import Optional

from necromancer_core_enums import GPUVendor, ComputeAPI
from necromancer_core_specs import UniversalGPUSpecs, UniversalNecromancerStrategy

logger = logging.getLogger(__name__)


class StrategyGenerator:
    """Generate optimal strategy for any GPU"""
    
    @staticmethod
    def generate_strategy(gpu_specs: UniversalGPUSpecs) -> UniversalNecromancerStrategy:
        """Generate optimal strategy based on GPU vendor and specs"""
        
        logger.info(f"Generating strategy for {gpu_specs.name} ({gpu_specs.vendor.value})")
        
        if gpu_specs.vendor == GPUVendor.NVIDIA:
            return _generate_nvidia_strategy(gpu_specs)
        elif gpu_specs.vendor == GPUVendor.INTEL_INTEGRATED:
            return _generate_intel_integrated_strategy(gpu_specs)
        elif gpu_specs.vendor == GPUVendor.INTEL_ARC:
            return _generate_intel_arc_strategy(gpu_specs)
        elif gpu_specs.vendor == GPUVendor.AMD:
            return _generate_amd_strategy(gpu_specs)
        elif gpu_specs.vendor == GPUVendor.CPU:
            return _generate_cpu_strategy(gpu_specs)
        else:
            logger.warning(f"Unknown vendor {gpu_specs.vendor}, using CPU strategy")
            return _generate_cpu_strategy(gpu_specs)


def _generate_nvidia_strategy(gpu_specs: UniversalGPUSpecs) -> UniversalNecromancerStrategy:
    """Strategy for NVIDIA GPUs"""
    
    compute_score = gpu_specs.compute_capability[0] + (gpu_specs.compute_capability[1] / 10)
    available_vram = gpu_specs.available_memory_gb * 0.80
    
    # Attention mechanism
    if compute_score >= 8.0:
        attention = "flash_attention_2"
    elif compute_score >= 7.5:
        attention = "sdpa"
    elif compute_score >= 6.0:
        attention = "xformers"
    else:
        attention = "eager"
    
    # CUDA version
    if compute_score < 6.0:
        cuda_version = "11.8"
    elif compute_score < 8.0:
        cuda_version = "12.1"
    else:
        cuda_version = "12.4"
    
    # Quantization based on VRAM
    if available_vram >= 20:
        max_bits = 16
        max_model = 13
    elif available_vram >= 12:
        max_bits = 8
        max_model = 13
    elif available_vram >= 8:
        max_bits = 6
        max_model = 8
    elif available_vram >= 6:
        max_bits = 5
        max_model = 7
    elif available_vram >= 4:
        max_bits = 4
        max_model = 7
    else:
        max_bits = 4
        max_model = 3
    
    # Context length
    if available_vram >= 16:
        max_context = 8192
    elif available_vram >= 12:
        max_context = 4096
    elif available_vram >= 8:
        max_context = 2048
    else:
        max_context = 1024
    
    # Batch size
    if available_vram >= 20:
        max_batch = 8
    elif available_vram >= 12:
        max_batch = 4
    elif available_vram >= 8:
        max_batch = 2
    else:
        max_batch = 1
    
    # KV cache
    if available_vram >= 12:
        kv_cache = "float16"
    elif available_vram >= 8:
        kv_cache = "int8"
    else:
        kv_cache = "int4"
    
    limitations = []
    optimizations = []
    warnings = []
    
    if compute_score < 6.0:
        limitations.append("Legacy GPU - forced to eager mode attention")
        warnings.append("Performance will be significantly slower than modern GPUs")
    
    if compute_score >= 8.0:
        optimizations.append("Flash Attention 2 supported - maximum performance")
    elif compute_score >= 7.5:
        optimizations.append("SDPA (Scaled Dot-Product Attention) available")
    elif compute_score >= 6.0:
        optimizations.append("Memory-efficient attention via xformers")
    
    if available_vram < 6:
        limitations.append("Batch size limited to 1")
        warnings.append("Very limited VRAM - 4-bit quantization required")
    
    if compute_score < 6.0:
        limitations.append("Tensor cores not available")
    
    return UniversalNecromancerStrategy(
        gpu_specs=asdict(gpu_specs),
        recommended_backend="llama-cpp",
        recommended_compute_api="CUDA",
        recommended_cuda_version=cuda_version,
        attention_mechanism=attention,
        max_quantization_bits=max_bits,
        max_context_length=max_context,
        max_batch_size=max_batch,
        kv_cache_dtype=kv_cache,
        use_gradient_checkpointing=(available_vram < 12),
        offload_strategy="none" if available_vram >= 6 else "cpu",
        limitations=limitations,
        optimizations=optimizations,
        estimated_max_model_size_b=float(max_model),
        warnings=warnings,
    )


def _generate_intel_integrated_strategy(gpu_specs: UniversalGPUSpecs) -> UniversalNecromancerStrategy:
    """Strategy for Intel integrated GPUs (UHD Graphics, Iris Xe)"""
    
    # Available system RAM (large for integrated)
    available_ram = gpu_specs.shared_system_ram_gb * 0.80
    
    # Determine attention mechanism based on generation
    if gpu_specs.architecture_generation >= 5:
        attention = "sdpa"
    elif gpu_specs.architecture_generation >= 3:
        attention = "xformers"
    else:
        attention = "eager"
    
    # Quantization based on available RAM
    if available_ram >= 20:
        max_bits = 16
        max_model = 13
    elif available_ram >= 12:
        max_bits = 8
        max_model = 13
    elif available_ram >= 8:
        max_bits = 6
        max_model = 8
    else:
        max_bits = 4
        max_model = 7
    
    warnings = [
        "Integrated GPU: Performance may be limited by system RAM bandwidth"
    ]
    optimizations = [
        "Using unified memory (GPU + System RAM)",
        "OpenVINO or llama.cpp recommended for integrated GPU",
        "CPU cores available for parallel inference"
    ]
    
    limitations = [
        "Lower memory bandwidth than discrete GPU",
        "Shared system RAM with CPU",
        "Best for single-request inference"
    ]
    
    return UniversalNecromancerStrategy(
        gpu_specs=asdict(gpu_specs),
        recommended_backend="llama-cpp",
        recommended_compute_api="ONEAPI",
        recommended_cuda_version="N/A",
        attention_mechanism=attention,
        max_quantization_bits=max_bits,
        max_context_length=4096,
        max_batch_size=1,
        kv_cache_dtype="int4",
        use_gradient_checkpointing=False,
        offload_strategy="none",
        in_vram_estimate_gb=0,
        in_system_ram_estimate_gb=available_ram * 0.5,
        limitations=limitations,
        optimizations=optimizations,
        estimated_max_model_size_b=float(max_model),
        warnings=warnings,
    )


def _generate_intel_arc_strategy(gpu_specs: UniversalGPUSpecs) -> UniversalNecromancerStrategy:
    """Strategy for Intel Arc discrete GPUs"""
    
    available_vram = gpu_specs.available_memory_gb * 0.80
    
    # Quantization based on VRAM
    if available_vram >= 16:
        max_bits = 16
        max_model = 13
    elif available_vram >= 8:
        max_bits = 8
        max_model = 8
    elif available_vram >= 6:
        max_bits = 6
        max_model = 7
    elif available_vram >= 4:
        max_bits = 5
        max_model = 7
    else:
        max_bits = 4
        max_model = 3
    
    warnings = []
    optimizations = [
        "Intel Arc: Consider using OpenVINO for optimal performance"
    ]
    limitations = []
    
    return UniversalNecromancerStrategy(
        gpu_specs=asdict(gpu_specs),
        recommended_backend="llama-cpp",
        recommended_compute_api="SYCL",
        recommended_cuda_version="N/A",
        attention_mechanism="sdpa",
        max_quantization_bits=max_bits,
        max_context_length=4096,
        max_batch_size=2,
        kv_cache_dtype="int8" if available_vram >= 8 else "int4",
        use_gradient_checkpointing=available_vram < 12,
        offload_strategy="none",
        limitations=limitations,
        optimizations=optimizations,
        estimated_max_model_size_b=float(max_model),
        warnings=warnings,
    )


def _generate_amd_strategy(gpu_specs: UniversalGPUSpecs) -> UniversalNecromancerStrategy:
    """Strategy for AMD RDNA GPUs"""
    
    available_vram = gpu_specs.available_memory_gb * 0.80
    
    # Quantization
    if available_vram >= 20:
        max_bits = 16
        max_model = 13
    elif available_vram >= 12:
        max_bits = 8
        max_model = 13
    elif available_vram >= 8:
        max_bits = 6
        max_model = 8
    else:
        max_bits = 4
        max_model = 7
    
    optimizations = [
        "AMD RDNA: Consider using ROCm for optimal performance"
    ]
    limitations = []
    
    return UniversalNecromancerStrategy(
        gpu_specs=asdict(gpu_specs),
        recommended_backend="llama-cpp",
        recommended_compute_api="ROCM",
        recommended_cuda_version="N/A",
        attention_mechanism="sdpa",
        max_quantization_bits=max_bits,
        max_context_length=4096,
        max_batch_size=2,
        kv_cache_dtype="int8" if available_vram >= 8 else "int4",
        use_gradient_checkpointing=available_vram < 12,
        offload_strategy="none",
        limitations=limitations,
        optimizations=optimizations,
        estimated_max_model_size_b=float(max_model),
    )


def _generate_cpu_strategy(gpu_specs: UniversalGPUSpecs) -> UniversalNecromancerStrategy:
    """Strategy for CPU-only inference"""
    
    available_ram = gpu_specs.shared_system_ram_gb * 0.70
    
    # Can use lots of memory efficiently on CPU
    if available_ram >= 32:
        max_model = 70
        max_bits = 8
    elif available_ram >= 20:
        max_model = 13
        max_bits = 8
    elif available_ram >= 12:
        max_model = 13
        max_bits = 6
    elif available_ram >= 8:
        max_model = 8
        max_bits = 5
    else:
        max_model = 7
        max_bits = 4
    
    warnings = [
        "CPU inference is very slow - expect 5-20 tokens/sec"
    ]
    
    limitations = [
        "CPU inference is slow (10-50 tok/sec typically)",
        "Single-threaded worse than multi-threaded",
        "Better for prompt processing than token generation"
    ]
    
    optimizations = [
        "Use llama.cpp with maximum threads (e.g., -t 16)",
        "Enable avx2 / avx512 CPU features if available",
        "Quantization reduces memory bandwidth requirements",
        "Large context window possible with system RAM"
    ]
    
    return UniversalNecromancerStrategy(
        gpu_specs=asdict(gpu_specs),
        recommended_backend="llama-cpp",
        recommended_compute_api="CPU",
        recommended_cuda_version="N/A",
        attention_mechanism="eager",
        max_quantization_bits=max_bits,
        max_context_length=8192,
        max_batch_size=1,
        kv_cache_dtype="float16",
        use_gradient_checkpointing=False,
        offload_strategy="none",
        limitations=limitations,
        optimizations=optimizations,
        estimated_max_model_size_b=float(max_model),
        warnings=warnings,
    )

