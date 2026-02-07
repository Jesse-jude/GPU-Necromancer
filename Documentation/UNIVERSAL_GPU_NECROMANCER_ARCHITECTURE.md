# 🧟 GPU Necromancer - Multi-Vendor Architecture

## 🎯 New Vision: Universal GPU Support

Support **ALL GPUs** from budget to high-end across vendors:
- **NVIDIA** (Kepler → Hopper)
- **Intel Arc** (A380 → A770)
- **Intel Integrated** (UHD Graphics, Iris Xe)
- **AMD RDNA** (RX 5500 → RX 7900)
- **CPU Fallback** (Multi-threaded inference on CPU)

---

## 🏗️ Revised System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE LAYER                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌──────────────────────┐              ┌─────────────────────────┐    │
│   │   CLI Interface      │              │  Streamlit Dashboard    │    │
│   │  (necromancer_       │              │  "War Room"             │    │
│   │   agent.py main())   │              │  (dashboard_app.py)     │    │
│   └──────────────────────┘              └─────────────────────────┘    │
│                                                                           │
└───────────────────────────────┬───────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  AGENT ORCHESTRATION LAYER (UNCHANGED)                  │
├─────────────────────────────────────────────────────────────────────────┤
│         NecromancerAgent + ModelSqueezer + SelfHealingAgent            │
└───────────────────────────────┬───────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               🆕 HARDWARE ABSTRACTION LAYER (NEW!)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  GPUDetector (Unified Interface)                │   │
│  │                                                                 │   │
│  │  • detect_all_devices() → List[GPUDevice]                     │   │
│  │  • get_optimal_device() → GPUDevice                           │   │
│  │  • is_integrated() → bool                                     │   │
│  │  • supports_api(api_name) → bool                              │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         △           △           △           △           △             │
│         │           │           │           │           │             │
│    ┌────┴─┐   ┌────┴─┐    ┌───┴──┐   ┌───┴──┐   ┌────┴─┐            │
│    │      │   │      │    │      │   │      │   │      │            │
│   ▼      ▼   ▼      ▼    ▼      ▼   ▼      ▼   ▼      ▼            │
│ ┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐        │
│ │ NVIDIA   ││ Intel    ││ Intel    ││   AMD    ││   CPU    │        │
│ │ Detector ││ Arc      ││ Integrated││ Detector ││ Detector │        │
│ │          ││ Detector ││ Detector ││          ││          │        │
│ │(pynvml) ││(Intel    ││(Intel    ││(rocm-    ││(psutil)  │        │
│ │          ││ GPU      ││ GPU      ││ smi)     ││          │        │
│ │          ││ Metrics) ││ Runtime) ││          ││          │        │
│ └──────────┘└──────────┘└──────────┘└──────────┘└──────────┘        │
│                                                                           │
│  All return: GPUSpecs (vendor-agnostic data class)                     │
│                                                                           │
└───────────────────────────────┬───────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              INFERENCE ENGINE ABSTRACTION LAYER (NEW!)                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │               InferenceEngine (Unified Interface)                │  │
│  │                                                                  │  │
│  │  • load_model(model_path, quantization) → Model               │  │
│  │  • infer(model, inputs) → outputs                             │  │
│  │  • get_available_backends() → List[str]                       │  │
│  │  • supports_feature(feature) → bool                           │  │
│  │                                                                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│    △      △      △      △      △      △      △      △      △        │
│    │      │      │      │      │      │      │      │      │        │
│ ┌──┴─┐┌──┴─┐┌──┴─┐┌──┴─┐┌──┴─┐┌──┴─┐┌──┴─┐┌──┴─┐┌──┴─┐        │
│ │    ││    ││    ││    ││    ││    ││    ││    ││    │        │
│▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼        │
│┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐   │
││Llama.cpp ││ONNX      ││Hugging   ││OpenVINO  ││Ollama    │   │
││(CPU/GPU) ││Runtime   ││Face      ││(Intel    ││(Local)   │   │
││          ││(GPU/CPU) ││Transform ││opt)      ││          │   │
││CUDA,     ││          ││ers (GPU) ││          ││          │   │
││Metal,    ││          ││          ││          ││          │   │
││CPU       ││          ││          ││          ││          │   │
│└──────────┘└──────────┘└──────────┘└──────────┘└──────────┘   │
│                                                                   │
│  Multi-backend support with automatic fallback                 │
│                                                                   │
└───────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       HARDWARE LAYER                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │ NVIDIA GPU │  │ Intel Arc  │  │ Intel UHD/ │  │ AMD GPU    │       │
│  │ (CUDA)     │  │ (Data      │  │ Iris Xe    │  │ (ROCm)     │       │
│  │            │  │ Center API)│  │ (Media     │  │            │       │
│  │ VRAM: 4-24GB  │ VRAM: 4-16GB │ Shared RAM  │  │VRAM: 4-16GB│      │
│  │            │  │            │  │ 0.5-16GB   │  │            │       │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘       │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                    CPU (Multi-core)                          │      │
│  │                  (Fallback for all)                          │      │
│  │          System RAM: 8GB - 256GB available                  │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🆕 Hardware Abstraction Layer

### New Vendor-Agnostic GPUSpecs

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

class GPUVendor(Enum):
    """GPU Vendor identification"""
    NVIDIA = "nvidia"
    INTEL_ARC = "intel_arc"
    INTEL_INTEGRATED = "intel_integrated"
    AMD = "amd"
    APPLE = "apple"
    CPU = "cpu"

class ComputeAPI(Enum):
    """Supported compute APIs"""
    CUDA = "cuda"
    SYCL = "sycl"           # Intel Arc
    ONEAPI = "oneapi"       # Intel integrated
    ROCM = "rocm"           # AMD
    METAL = "metal"         # Apple
    CPU = "cpu"

class MemoryType(Enum):
    """Memory hierarchy types"""
    DEDICATED = "dedicated"  # Separate VRAM
    UNIFIED = "unified"      # Shared system RAM (integrated GPUs)
    HETEROGENEOUS = "heterogeneous"  # Mix of both

@dataclass
class GPUSpecs:
    """Universal GPU specification (vendor-agnostic)"""
    
    # Identification
    name: str
    vendor: GPUVendor
    device_id: int
    
    # Architecture
    compute_family: str           # "Ampere", "Xe", "RDNA2", etc.
    compute_capability: Tuple[int, int]  # Still useful for sorting
    architecture_generation: int  # Relative performance tier (1-10)
    
    # Memory Configuration
    memory_type: MemoryType
    total_memory_gb: float
    available_memory_gb: float
    
    # Memory Hierarchy (for integrated GPUs)
    dedicated_vram_gb: float      # For discrete GPUs = total_memory
    shared_system_ram_gb: float   # For integrated GPUs
    
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
    driver_version: str
    compute_runtime_version: str  # CUDA, SYCL, ROCm, etc.
    
    # Performance
    estimated_tflops_fp32: float
    estimated_tflops_int8: float
    
    # Other
    pci_bus_id: str
    is_integrated: bool = False
    is_mobile: bool = False
    power_consumption_watts: Optional[int] = None
```

### New GPUDetector Interface

```python
from abc import ABC, abstractmethod
from typing import List, Optional

class GPUDetector(ABC):
    """Abstract base class for all GPU detectors"""
    
    @abstractmethod
    def detect_devices(self) -> List[GPUSpecs]:
        """Detect all available GPUs of this vendor"""
        pass
    
    @abstractmethod
    def get_device_info(self, device_id: int) -> GPUSpecs:
        """Get detailed info for specific device"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this vendor's GPUs are available"""
        pass
    
    @abstractmethod
    def get_driver_version(self) -> str:
        """Get driver/runtime version"""
        pass


class NVIDIADetector(GPUDetector):
    """NVIDIA GPU detection using pynvml"""
    # Implementation: existing code
    pass


class IntelArcDetector(GPUDetector):
    """Intel Arc GPU detection using Intel GPU Metrics"""
    
    def __init__(self):
        try:
            import intel_gpu_metrics
            self.metrics = intel_gpu_metrics
        except ImportError:
            self.metrics = None
    
    def detect_devices(self) -> List[GPUSpecs]:
        """Detect Intel Arc GPUs"""
        if not self.is_available():
            return []
        
        devices = []
        # Use intel-gpu-tools or Intel GPU Metrics API
        # For each detected Arc GPU:
        # - Query VRAM via SYCL API
        # - Identify model (A380, A750, A770)
        # - Get compute units
        # - Return GPUSpecs
        
        return devices
    
    def is_available(self) -> bool:
        # Check if Intel Arc drivers installed
        # Try to import Intel GPU libraries
        return self.metrics is not None


class IntelIntegratedDetector(GPUDetector):
    """Intel integrated GPU detection (UHD Graphics, Iris Xe)"""
    
    def __init__(self):
        try:
            import gpu_runtime  # Hypothetical Intel GPU Runtime
            self.runtime = gpu_runtime
        except ImportError:
            self.runtime = None
    
    def detect_devices(self) -> List[GPUSpecs]:
        """Detect Intel integrated GPUs"""
        if not self.is_available():
            return []
        
        devices = []
        # Query via Windows Device Manager or Linux intel-gpu-tools
        # For Intel integrated GPU:
        # - Shared system RAM as VRAM
        # - Get GPU EU count
        # - Map to UHD Graphics / Iris Xe generation
        # - Return GPUSpecs with memory_type=UNIFIED
        
        return devices
    
    def is_available(self) -> bool:
        # Check for Intel integrated GPU
        return self.runtime is not None


class AMDDetector(GPUDetector):
    """AMD GPU detection using ROCm"""
    
    def __init__(self):
        try:
            import rocm  # or rocm-smi bindings
            self.rocm = rocm
        except ImportError:
            self.rocm = None
    
    def detect_devices(self) -> List[GPUSpecs]:
        # Use rocm-smi or HIP to detect AMD GPUs
        # Query RDNA generation, VRAM, compute units
        return []


class UnifiedGPUDetector:
    """Unified detector that tries all vendors"""
    
    def __init__(self):
        self.detectors = [
            NVIDIADetector(),
            IntelArcDetector(),
            IntelIntegratedDetector(),
            AMDDetector(),
        ]
    
    def detect_all_devices(self) -> List[GPUSpecs]:
        """Detect all available GPUs across all vendors"""
        all_devices = []
        
        for detector in self.detectors:
            if detector.is_available():
                try:
                    devices = detector.detect_devices()
                    all_devices.extend(devices)
                except Exception as e:
                    logger.warning(f"Failed to detect {detector.__class__.__name__}: {e}")
        
        # Sort by performance tier (descending)
        all_devices.sort(
            key=lambda d: (d.architecture_generation, d.total_memory_gb),
            reverse=True
        )
        
        return all_devices
    
    def get_optimal_device(self) -> Optional[GPUSpecs]:
        """Get the best available device for inference"""
        devices = self.detect_all_devices()
        return devices[0] if devices else None
    
    def has_any_gpu(self) -> bool:
        """Check if any GPU is available"""
        return len(self.detect_all_devices()) > 0
```

---

## 🆕 Inference Engine Abstraction Layer

```python
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional

class InferenceBackend(ABC):
    """Abstract base for inference engines"""
    
    name: str
    supported_vendors: List[GPUVendor]
    supported_quantizations: List[int]  # [4, 5, 6, 8, 16]
    requires_conversion: bool  # Does model need format conversion?
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is installed and working"""
        pass
    
    @abstractmethod
    def supports_gpu(self, gpu: GPUSpecs) -> bool:
        """Check if backend supports this GPU"""
        pass
    
    @abstractmethod
    def load_model(
        self, 
        model_path: str, 
        quantization_bits: int,
        device: Optional[GPUSpecs] = None
    ) -> "Model":
        """Load model with specific quantization"""
        pass
    
    @abstractmethod
    def infer(self, model: "Model", inputs: Dict) -> Dict:
        """Run inference"""
        pass
    
    @abstractmethod
    def get_memory_usage(self, model: "Model") -> float:
        """Get current memory usage in GB"""
        pass


class LlamaCppBackend(InferenceBackend):
    """llama.cpp - Highly optimized CPU + GPU backend"""
    
    name = "llama-cpp-python"
    supported_vendors = [GPUVendor.CPU, GPUVendor.NVIDIA, GPUVendor.INTEL_INTEGRATED]
    supported_quantizations = [4, 5, 6, 8, 16]
    requires_conversion = False  # Works with GGUF directly
    
    def __init__(self):
        self.llama = None
    
    def is_available(self) -> bool:
        try:
            from llama_cpp import Llama
            self.llama = Llama
            return True
        except ImportError:
            return False
    
    def supports_gpu(self, gpu: GPUSpecs) -> bool:
        """llama.cpp supports most GPUs via different backends"""
        if gpu.vendor == GPUVendor.CPU:
            return True
        if gpu.vendor == GPUVendor.NVIDIA:
            return True  # CUDA backend
        if gpu.vendor == GPUVendor.INTEL_INTEGRATED:
            return True  # CPU backend (integrated GPU uses system RAM)
        if gpu.vendor == GPUVendor.AMD:
            return True  # ROCm backend
        return False
    
    def load_model(self, model_path: str, quantization_bits: int, device=None):
        # Load GGUF model with appropriate settings
        # If device is integrated GPU: use CPU backend with large n_batch
        # If device is NVIDIA: use CUDA backend
        pass
    
    def infer(self, model, inputs):
        # Run inference using llama.cpp
        pass


class OnnxBackend(InferenceBackend):
    """ONNX Runtime - Multi-vendor support"""
    
    name = "onnx-runtime"
    supported_vendors = [
        GPUVendor.CPU,
        GPUVendor.NVIDIA,
        GPUVendor.INTEL_ARC,
        GPUVendor.INTEL_INTEGRATED,
        GPUVendor.AMD,
    ]
    supported_quantizations = [8, 16]  # ONNX is less flexible with quant
    requires_conversion = True  # Need ONNX format
    
    def is_available(self) -> bool:
        try:
            import onnxruntime
            self.ort = onnxruntime
            return True
        except ImportError:
            return False
    
    def supports_gpu(self, gpu: GPUSpecs) -> bool:
        # ONNX supports different providers per vendor
        if gpu.vendor == GPUVendor.NVIDIA:
            return "CUDAExecutionProvider" in self.ort.get_available_providers()
        if gpu.vendor == GPUVendor.INTEL_ARC:
            return "DmlExecutionProvider" in self.ort.get_available_providers()
        if gpu.vendor == GPUVendor.AMD:
            return "ROCMExecutionProvider" in self.ort.get_available_providers()
        return True  # CPU always works


class OpenVINOBackend(InferenceBackend):
    """OpenVINO - Optimized for Intel (Arc + Integrated)"""
    
    name = "openvino"
    supported_vendors = [
        GPUVendor.INTEL_ARC,
        GPUVendor.INTEL_INTEGRATED,
        GPUVendor.CPU,
    ]
    supported_quantizations = [4, 8, 16]
    requires_conversion = True  # Need OpenVINO IR format
    
    def is_available(self) -> bool:
        try:
            from openvino.runtime import Core
            self.core = Core()
            return True
        except ImportError:
            return False
    
    def supports_gpu(self, gpu: GPUSpecs) -> bool:
        if gpu.vendor in [GPUVendor.INTEL_ARC, GPUVendor.INTEL_INTEGRATED]:
            return True
        return False  # OpenVINO focuses on Intel hardware
    
    def load_model(self, model_path: str, quantization_bits: int, device=None):
        # Load OpenVINO IR (.xml/.bin)
        # Optimal for Intel integrated GPUs with shared memory
        pass


class UnifiedInferenceEngine:
    """Unified inference with automatic backend selection"""
    
    def __init__(self, gpu: GPUSpecs):
        self.gpu = gpu
        self.backends = [
            LlamaCppBackend(),
            OnnxBackend(),
            OpenVINOBackend(),
            # HuggingFace, Ollama, etc.
        ]
        self.active_backend = None
    
    def select_backend(self, model_format: str = "auto") -> InferenceBackend:
        """Select best available backend for GPU"""
        
        # Priority order based on GPU vendor
        if self.gpu.vendor == GPUVendor.INTEL_INTEGRATED:
            # OpenVINO best for integrated GPUs
            # Fall back to llama.cpp
            priority = [OpenVINOBackend, LlamaCppBackend, OnnxBackend]
        
        elif self.gpu.vendor == GPUVendor.INTEL_ARC:
            # OpenVINO optimal for Arc
            # ONNX with DML provider
            # llama.cpp with SYCL
            priority = [OpenVINOBackend, OnnxBackend, LlamaCppBackend]
        
        elif self.gpu.vendor == GPUVendor.NVIDIA:
            # llama.cpp with CUDA
            # ONNX with CUDA
            priority = [LlamaCppBackend, OnnxBackend]
        
        elif self.gpu.vendor == GPUVendor.CPU:
            # llama.cpp (most optimized for CPU)
            priority = [LlamaCppBackend, OnnxBackend]
        
        else:
            # Fallback to available
            priority = self.backends
        
        for backend_class in priority:
            backend = next((b for b in self.backends if isinstance(b, backend_class)), None)
            if backend and backend.is_available() and backend.supports_gpu(self.gpu):
                self.active_backend = backend
                return backend
        
        raise RuntimeError(f"No suitable backend found for {self.gpu.vendor}")
    
    def load_model(self, model_path: str, quantization_bits: int):
        backend = self.select_backend()
        return backend.load_model(model_path, quantization_bits, self.gpu)
    
    def infer(self, model, inputs):
        if not self.active_backend:
            self.select_backend()
        return self.active_backend.infer(model, inputs)
```

---

## 🆕 Updated Squeezer for Multi-GPU

```python
class UniversalModelSqueezer:
    """Quantization calculator for any GPU vendor"""
    
    def __init__(self, gpu_specs: GPUSpecs):
        self.gpu = gpu_specs
        self.is_integrated = gpu_specs.is_integrated
        self.memory_type = gpu_specs.memory_type
    
    def calculate_optimal_quantization(self, model_name: str, context_length: int):
        """
        Key difference: Integrated GPUs use system RAM
        Need different strategy than discrete GPUs
        """
        
        if self.is_integrated:
            return self._calculate_integrated_gpu(model_name, context_length)
        else:
            return self._calculate_discrete_gpu(model_name, context_length)
    
    def _calculate_discrete_gpu(self, model_name: str, context_length: int):
        """
        Standard approach: dedicated VRAM
        Goal: maximize performance within VRAM constraints
        """
        base_size = MODEL_SIZES[model_name]
        available = self.gpu.available_memory_gb * 0.80  # 20% buffer
        
        # Try quantizations from highest quality to lowest
        for bits in [16, 8, 6, 5, 4]:
            quantized = base_size * QUANT_FACTORS[bits]
            kv_cache = self._estimate_kv_cache(context_length, bits)
            total = quantized + kv_cache
            
            if total <= available:
                return {
                    'bits': bits,
                    'total_memory': total,
                    'fits': True,
                    'strategy': 'quality_first'
                }
        
        # Nothing fits - use most aggressive
        return {
            'bits': 4,
            'total_memory': self._estimate_memory_4bit(model_name, context_length),
            'fits': False,
            'warning': 'Model may not fit - CPU offloading recommended'
        }
    
    def _calculate_integrated_gpu(self, model_name: str, context_length: int):
        """
        Integrated GPU approach: shared system RAM
        Goal: use GPU capabilities while avoiding RAM swapping
        
        Key considerations:
        - System RAM much larger than dedicated VRAM
        - Latency higher than dedicated VRAM
        - Better to use system RAM than fail
        - May want to force CPU if GPU computation slow
        """
        
        base_size = MODEL_SIZES[model_name]
        
        # Available system RAM (keep 2GB for OS + other apps)
        available_ram = (self.gpu.shared_system_ram_gb - 2.0) * 0.9
        
        # Available dedicated VRAM if any
        available_vram = self.gpu.dedicated_vram_gb * 0.80
        
        total_available = available_ram + available_vram
        
        # For integrated GPUs: can be more aggressive since using system RAM
        for bits in [16, 8, 6, 5, 4]:
            quantized = base_size * QUANT_FACTORS[bits]
            kv_cache = self._estimate_kv_cache(context_length, bits)
            total = quantized + kv_cache
            
            if total <= total_available:
                # Estimate performance based on memory type
                if total <= available_vram:
                    speed_tier = "fast"  # In dedicated VRAM
                elif total <= available_vram + (available_ram / 2):
                    speed_tier = "moderate"  # Mostly in VRAM
                else:
                    speed_tier = "slow"  # Mostly in system RAM
                
                return {
                    'bits': bits,
                    'total_memory': total,
                    'in_vram': min(quantized + kv_cache, available_vram),
                    'in_system_ram': max(0, total - available_vram),
                    'fits': True,
                    'speed_tier': speed_tier,
                    'strategy': 'space_first',  # Prioritize fitting over speed
                    'warning': f'Model will use system RAM - expect {speed_tier} inference speed'
                }
        
        # Still doesn't fit - offer CPU inference as option
        return {
            'bits': 4,
            'total_memory': self._estimate_memory_4bit(model_name, context_length),
            'fits': False,
            'recommendation': 'CPU inference only',
            'note': 'Model requires more memory than available - suggest CPU inference'
        }
    
    def _estimate_kv_cache(self, context_length: int, quantization_bits: int) -> float:
        """KV cache size depends on quantization"""
        base_kv = (context_length / 2048) * 1.5
        
        # Quantized KV cache
        if quantization_bits == 16:
            return base_kv
        elif quantization_bits == 8:
            return base_kv * 0.5
        elif quantization_bits <= 4:
            return base_kv * 0.25
        else:
            return base_kv * (quantization_bits / 16)
```

---

## 🆕 Updated Strategy Generator

```python
class UniversalNecromancerStrategy:
    """Generate strategy for any GPU vendor"""
    
    @staticmethod
    def generate_strategy(gpu_specs: GPUSpecs) -> NecromancerStrategy:
        """Generate optimal strategy based on GPU vendor"""
        
        # Dispatch to vendor-specific generator
        if gpu_specs.vendor == GPUVendor.NVIDIA:
            return NVIDIAStrategyGenerator.generate(gpu_specs)
        
        elif gpu_specs.vendor == GPUVendor.INTEL_ARC:
            return IntelArcStrategyGenerator.generate(gpu_specs)
        
        elif gpu_specs.vendor == GPUVendor.INTEL_INTEGRATED:
            return IntelIntegratedStrategyGenerator.generate(gpu_specs)
        
        elif gpu_specs.vendor == GPUVendor.AMD:
            return AMDStrategyGenerator.generate(gpu_specs)
        
        elif gpu_specs.vendor == GPUVendor.CPU:
            return CPUStrategyGenerator.generate(gpu_specs)
        
        else:
            raise ValueError(f"Unknown vendor: {gpu_specs.vendor}")


class IntelIntegratedStrategyGenerator:
    """Special handling for Intel integrated GPUs"""
    
    @staticmethod
    def generate(gpu_specs: GPUSpecs) -> NecromancerStrategy:
        """
        Intel integrated GPU strategy:
        - Unified memory (GPU + CPU RAM)
        - Lower bandwidth than discrete GPU
        - Better to use system RAM than fail
        - May want to suggest CPU-only if GPU slow
        """
        
        # Available system RAM (large for integrated)
        available_ram = gpu_specs.shared_system_ram_gb * 0.80
        
        # Determine attention mechanism
        if "Iris Xe Max" in gpu_specs.name or "Arc" in gpu_specs.name:
            attention = "sdpa"  # Modern Intel
        elif "Iris Xe" in gpu_specs.name:
            attention = "xformers"  # Xe still decent
        else:
            attention = "eager"  # UHD Graphics - just use eager
        
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
        
        # Special notes for integrated GPUs
        warnings = []
        if available_ram < 16:
            warnings.append("Integrated GPU: Performance may be limited by system RAM bandwidth")
        
        optimizations = [
            "Using unified memory (GPU + System RAM)",
            "OpenVINO or llama.cpp recommended for integrated GPU",
            "CPU cores available for parallel inference",
        ]
        
        return NecromancerStrategy(
            gpu_specs=asdict(gpu_specs),
            recommended_backend="openvino",  # Best for Intel
            recommended_cuda_version="N/A",  # Uses ONEAPI or SYCL
            attention_mechanism=attention,
            max_quantization_bits=max_bits,
            max_context_length=4096,
            max_batch_size=1,  # Integrated GPUs don't benefit from batching
            kv_cache_dtype="int4",
            use_gradient_checkpointing=False,
            offload_strategy="none",  # Already using system RAM
            limitations=[
                "Lower memory bandwidth than discrete GPU",
                "Shared system RAM with CPU",
                "Best for single-request inference",
            ],
            optimizations=optimizations,
            estimated_max_model_size_b=max_model,
            warnings=warnings,
        )


class CPUStrategyGenerator:
    """Strategy for CPU-only inference"""
    
    @staticmethod
    def generate(gpu_specs: GPUSpecs) -> NecromancerStrategy:
        """
        CPU-only strategy:
        - Use llama.cpp with multi-threaded inference
        - Large context possible (lots of RAM)
        - Trade speed for memory efficiency
        """
        
        available_ram = gpu_specs.shared_system_ram_gb * 0.70
        
        # CPU can use lots of memory efficiently
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
        
        return NecromancerStrategy(
            gpu_specs=asdict(gpu_specs),
            recommended_backend="llama_cpp",
            recommended_cuda_version="N/A",
            attention_mechanism="eager",
            max_quantization_bits=max_bits,
            max_context_length=8192,  # CPU can handle long context
            max_batch_size=1,
            kv_cache_dtype="float16",
            use_gradient_checkpointing=False,
            offload_strategy="none",
            limitations=[
                "CPU inference is slow (10-50 tok/sec typically)",
                "Single-threaded worse than multi-threaded",
                "Better for prompt processing than token generation",
            ],
            optimizations=[
                "Use llama.cpp with maximum threads (e.g., -t 16)",
                "Enable avx2 / avx512 CPU features if available",
                "Quantization reduces memory bandwidth requirements",
                "Large context window possible with system RAM",
            ],
            estimated_max_model_size_b=max_model,
            warnings=[
                "CPU inference is very slow - expect 5-20 tokens/sec",
                "Modern CPUs with many cores (16+) work best",
                "Consider quantizing to 4-bit for 70B models",
            ],
        )
```

---

## 🔄 Updated Agent Initialization

```python
class UniversalNecromancerAgent:
    """Updated agent to work with any GPU vendor"""
    
    def __init__(self, device_id: int = None, vendor_preference: str = None):
        # Detect all available devices
        self.detector = UnifiedGPUDetector()
        
        # Select device (auto or by preference)
        if device_id is not None:
            self.gpu = self.detector.detect_all_devices()[device_id]
        elif vendor_preference:
            devices = self.detector.detect_all_devices()
            self.gpu = next((d for d in devices if d.vendor.value == vendor_preference), devices[0])
        else:
            self.gpu = self.detector.get_optimal_device()
        
        if not self.gpu:
            raise RuntimeError("No compute device detected! Install GPU drivers or use CPU.")
        
        # Initialize components
        self.prober = UniversalNecromancerProber(self.gpu)
        self.squeezer = UniversalModelSqueezer(self.gpu)
        self.inference_engine = UnifiedInferenceEngine(self.gpu)
        self.healer = SelfHealingAgent(self.squeezer)
        
        self.strategy = UniversalNecromancerStrategy.generate_strategy(self.gpu)
```

---

## 📊 Vendor-Specific Performance Tiers

```python
# Estimated performance on popular models

PERFORMANCE_TIERS = {
    # NVIDIA
    "RTX 4090": {
        "vendor": "nvidia",
        "tier": "flagship",
        "llama_7b_tokens_sec": 400,
        "llama_13b_tokens_sec": 250,
        "llama_70b_tokens_sec": 50,
    },
    "RTX 3090": {
        "vendor": "nvidia",
        "tier": "enthusiast",
        "llama_7b_tokens_sec": 150,
        "llama_13b_tokens_sec": 80,
    },
    "RTX 2080 Ti": {
        "vendor": "nvidia",
        "tier": "legacy",
        "llama_7b_tokens_sec": 40,
        "llama_13b_tokens_sec": 20,
    },
    
    # Intel Arc
    "Arc A770": {
        "vendor": "intel_arc",
        "tier": "enthusiast",
        "llama_7b_tokens_sec": 60,
        "llama_13b_tokens_sec": 30,
    },
    "Arc A750": {
        "vendor": "intel_arc",
        "tier": "mid-range",
        "llama_7b_tokens_sec": 30,
    },
    
    # Intel Integrated
    "Iris Xe Max": {
        "vendor": "intel_integrated",
        "tier": "mid-range",
        "llama_7b_tokens_sec": 15,
    },
    "Iris Xe G7": {
        "vendor": "intel_integrated",
        "tier": "budget",
        "llama_7b_tokens_sec": 8,
    },
    "UHD Graphics 770": {
        "vendor": "intel_integrated",
        "tier": "budget",
        "llama_7b_tokens_sec": 5,
    },
    
    # CPU
    "Ryzen 9 5950X": {
        "vendor": "cpu",
        "tier": "high-end",
        "llama_7b_tokens_sec": 25,  # Multi-threaded
    },
    "Ryzen 5 3600": {
        "vendor": "cpu",
        "tier": "mid-range",
        "llama_7b_tokens_sec": 8,
    },
}
```

---

## 🎯 Key Architecture Changes

### 1. **Abstraction Layers**
- GPU Detection abstracted (works with any vendor)
- Inference engines abstracted (multiple backends)
- Memory types abstracted (dedicated vs unified)

### 2. **Vendor Support**
- NVIDIA GPUs (existing + enhanced)
- Intel Arc discrete GPUs
- Intel integrated GPUs (special handling for unified memory)
- AMD RDNA GPUs
- CPU fallback (always available)

### 3. **Quantization Strategy**
- Discrete GPU: maximize performance within VRAM
- Integrated GPU: use available system RAM
- CPU: can use very aggressive quantization

### 4. **Inference Backends**
- llama.cpp (best for CPU + GPU)
- ONNX Runtime (multi-vendor)
- OpenVINO (best for Intel)
- HuggingFace Transformers
- Ollama

### 5. **Memory Handling**
- Dedicated VRAM (discrete GPUs)
- Unified memory (integrated GPUs)
- System RAM (CPU inference)
- Automatic fallback between tiers

---

## 🚀 Usage Examples

### Auto-detect and use best available device

```python
agent = UniversalNecromancerAgent()  # Auto-selects best GPU
print(agent.gpu.name)  # "RTX 4090" or "Iris Xe G7" or "CPU"

result = agent.run_with_healing(
    ModelRequest(model_name="mistral-7b", task="chat")
)
```

### Prefer Intel Arc over CPU

```python
agent = UniversalNecromancerAgent(vendor_preference="intel_arc")
# Will use Arc A770 if available, fall back to CPU otherwise
```

### Force CPU even if GPU available

```python
agent = UniversalNecromancerAgent(vendor_preference="cpu")
# Good for testing, debugging, or consistency
```

### List all available devices

```python
detector = UnifiedGPUDetector()
devices = detector.detect_all_devices()

for device in devices:
    print(f"{device.name} ({device.vendor.value}): {device.total_memory_gb}GB")

# Output:
# RTX 4090 (nvidia): 24.0GB
# Intel Arc A770 (intel_arc): 8.0GB
# Intel UHD Graphics 770 (intel_integrated): Shared with 16GB System RAM
# CPU - Ryzen 9 (cpu): 64GB System RAM
```

---

## 📈 Benefits of Multi-Vendor Support

✅ Works with **any hardware** (no GPU → use CPU)
✅ **Future-proof** (adding new vendors is trivial)
✅ **Best device selection** (auto-picks optimal GPU)
✅ **Graceful degradation** (integrated GPU → CPU)
✅ **Cost optimization** (works with budget hardware)
✅ **Maximum reach** (billions of integrated GPUs)

---

## 🔌 Integration Checklist

```
✅ Unified GPUSpecs data class (vendor-agnostic)
✅ GPUDetector abstract interface
✅ Specific detectors (NVIDIA, Intel Arc, Intel Integrated, AMD, CPU)
✅ UnifiedGPUDetector coordinator
✅ InferenceBackend abstract interface
✅ Multiple inference engines (llama.cpp, ONNX, OpenVINO)
✅ UniversalModelSqueezer (handles all memory types)
✅ Vendor-specific strategy generators
✅ Special handling for integrated GPUs
✅ CPU fallback
✅ Automatic backend selection
✅ Performance tier estimates

Next: Implement each detector and backend!
```

This architecture ensures GPU Necromancer works on virtually any device! 🧟✨
