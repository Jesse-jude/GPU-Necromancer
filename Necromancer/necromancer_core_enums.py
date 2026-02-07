# necromancer/core/enums.py
"""
Universal GPU enums for multi-vendor support
"""

from enum import Enum


class GPUVendor(Enum):
    """GPU vendor identification"""
    NVIDIA = "nvidia"
    INTEL_ARC = "intel_arc"
    INTEL_INTEGRATED = "intel_integrated"
    AMD = "amd"
    APPLE = "apple"
    CPU = "cpu"


class ComputeAPI(Enum):
    """Supported compute APIs"""
    CUDA = "cuda"           # NVIDIA
    SYCL = "sycl"           # Intel Arc
    ONEAPI = "oneapi"       # Intel integrated
    ROCM = "rocm"           # AMD
    METAL = "metal"         # Apple
    CPU = "cpu"             # Multi-threaded CPU


class MemoryType(Enum):
    """Memory hierarchy types"""
    DEDICATED = "dedicated"      # Separate VRAM (discrete GPUs)
    UNIFIED = "unified"          # Shared system RAM (integrated GPUs)
    HETEROGENEOUS = "heterogeneous"  # Mix of both


class AttentionMechanism(Enum):
    """Supported attention mechanisms"""
    FLASH_ATTENTION_2 = "flash_attention_2"
    SDPA = "sdpa"                # Scaled Dot-Product Attention
    XFORMERS = "xformers"
    EAGER = "eager"


class QuantizationBits(Enum):
    """Supported quantization bit depths"""
    FP16 = 16
    INT8 = 8
    INT6 = 6
    INT5 = 5
    INT4 = 4


class InferenceBackendType(Enum):
    """Supported inference backends"""
    LLAMA_CPP = "llama_cpp"
    OPENVINO = "openvino"
    ONNX = "onnx"
    TRANSFORMERS = "transformers"
    OLLAMA = "ollama"
