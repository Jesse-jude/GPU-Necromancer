"""
GPU Necromancer - Universal Core Specifications
Vendor-agnostic data structures for all GPU types
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from necromancer_core_enums import GPUVendor, ComputeAPI, MemoryType


@dataclass
class UniversalGPUSpecs:
    """Universal GPU specification (vendor-agnostic)"""
    
    # Identification
    name: str
    vendor: GPUVendor
    device_id: int
    
    # Architecture
    compute_family: str                    # "Ampere", "Xe", "RDNA2", etc.
    compute_capability: Tuple[int, int]   # Still useful for sorting
    architecture_generation: int           # Relative performance tier (1-10)
    
    # Memory Configuration
    memory_type: MemoryType
    total_memory_gb: float
    available_memory_gb: float
    
    # Memory Hierarchy (for integrated GPUs)
    dedicated_vram_gb: float              # For discrete GPUs = total_memory
    shared_system_ram_gb: float           # For integrated GPUs
    
    # Compute Capabilities
    compute_api: ComputeAPI
    max_compute_units: int
    max_clock_freq_mhz: int
    memory_bandwidth_gbps: float
    
    # Support Flags
    supports_fp16: bool = True
    supports_fp32: bool = True
    supports_int8: bool = True
    supports_int4: bool = True
    supports_tensor_cores: bool = False
    supports_unified_memory: bool = False
    
    # Software Support
    driver_version: str = "Unknown"
    compute_runtime_version: str = "Unknown"
    
    # Performance
    estimated_tflops_fp32: float = 0.0
    estimated_tflops_int8: float = 0.0
    
    # Other
    pci_bus_id: str = ""
    is_integrated: bool = False
    is_mobile: bool = False
    power_consumption_watts: Optional[int] = None
    
    def __str__(self) -> str:
        """Pretty print GPU specs"""
        return f"""
GPU: {self.name}
Vendor: {self.vendor.value}
Generation: {self.compute_family}
Memory: {self.total_memory_gb:.1f}GB ({self.memory_type.value})
Available: {self.available_memory_gb:.1f}GB
Compute Units: {self.max_compute_units}
Bandwidth: {self.memory_bandwidth_gbps:.1f} GB/s
Performance: {self.estimated_tflops_fp32:.0f} TFLOPS (FP32)
Is Integrated: {self.is_integrated}
"""


@dataclass
class UniversalNecromancerStrategy:
    """Universal resurrection strategy for any GPU vendor"""
    gpu_specs: Dict
    recommended_backend: str                # "llama-cpp", "openvino", "onnx", etc.
    recommended_compute_api: str            # "CUDA", "SYCL", "ONEAPI", etc.
    recommended_cuda_version: str           # For backward compatibility
    attention_mechanism: str                # 'eager', 'xformers', 'sdpa', 'flash_attention_2'
    max_quantization_bits: int              # 4, 5, 6, 8, 16
    max_context_length: int
    max_batch_size: int
    kv_cache_dtype: str                    # 'float16', 'int8', 'int4'
    use_gradient_checkpointing: bool
    offload_strategy: str                  # 'none', 'cpu', 'disk'
    
    # Unified memory specific
    in_vram_estimate_gb: Optional[float] = None
    in_system_ram_estimate_gb: Optional[float] = None
    
    limitations: List[str] = field(default_factory=list)
    optimizations: List[str] = field(default_factory=list)
    estimated_max_model_size_b: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        """Pretty print strategy"""
        output = f"\n{'='*60}\nNECROMACER STRATEGY\n{'='*60}\n"
        output += f"Backend: {self.recommended_backend}\n"
        output += f"Compute API: {self.recommended_compute_api}\n"
        output += f"Attention: {self.attention_mechanism}\n"
        output += f"Quantization: {self.max_quantization_bits}-bit\n"
        output += f"Max Context: {self.max_context_length}\n"
        output += f"Max Model: ~{self.estimated_max_model_size_b:.0f}B\n"
        
        if self.optimizations:
            output += f"\nOptimizations:\n"
            for opt in self.optimizations:
                output += f"  ✓ {opt}\n"
        
        if self.limitations:
            output += f"\nLimitations:\n"
            for lim in self.limitations:
                output += f"  • {lim}\n"
        
        if self.warnings:
            output += f"\nWarnings:\n"
            for warn in self.warnings:
                output += f"  ⚠️  {warn}\n"
        
        return output
