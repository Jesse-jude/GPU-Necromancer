# 🧟 GPU Necromancer - Universal Support Implementation Roadmap

## 📋 Executive Summary

Transform GPU Necromancer from **NVIDIA-only** to **universal multi-vendor support**:
- ✅ NVIDIA GPUs (all generations)
- ✅ Intel Arc discrete GPUs
- ✅ Intel integrated GPUs (UHD Graphics, Iris Xe)
- ✅ AMD RDNA GPUs
- ✅ CPU-only fallback
- ✅ Automatic best device selection
- ✅ Multi-backend inference

---

## 🔄 Architecture Transformation

### Before (NVIDIA-Only)

```
User Input
    ↓
NVIDIADetector (pynvml)
    ↓
NVIDIAStrategy
    ↓
LlamaCppBackend (CUDA)
    ↓
NVIDIA GPU
```

**Limitation**: Only works with NVIDIA hardware

---

### After (Universal Multi-Vendor)

```
User Input
    ↓
UnifiedGPUDetector ──┬─→ NVIDIADetector (pynvml)
                     ├─→ IntelArcDetector
                     ├─→ IntelIntegratedDetector
                     ├─→ AMDDetector
                     └─→ CPUDetector
    ↓
Auto-Select Best Device
    ↓
UniversalStrategyGenerator ──┬─→ NVIDIAStrategy
                              ├─→ IntelArcStrategy
                              ├─→ IntelIntegratedStrategy (unified memory)
                              ├─→ AMDStrategy
                              └─→ CPUStrategy
    ↓
UnifiedInferenceEngine ──┬─→ LlamaCppBackend (CUDA/SYCL/ROCm/CPU)
                          ├─→ OpenVINOBackend (Intel)
                          ├─→ ONNXBackend (multi-vendor)
                          ├─→ TransformersBackend (NVIDIA/CPU)
                          └─→ Fallback chain
    ↓
Any GPU (or CPU)
```

**Benefit**: Works with any hardware, auto-selects optimal backend

---

## 📂 Project Structure

```
gpu-necromancer/
├── necromancer/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── specs.py          # 🆕 Universal GPUSpecs dataclass
│   │   ├── enums.py          # 🆕 GPUVendor, ComputeAPI, MemoryType
│   │   └── exceptions.py
│   │
│   ├── detectors/            # 🆕 Multi-vendor detection
│   │   ├── __init__.py
│   │   ├── base.py           # GPUDetector abstract class
│   │   ├── nvidia.py         # 📝 Enhanced NVIDIADetector
│   │   ├── intel_arc.py      # 🆕 IntelArcDetector
│   │   ├── intel_integrated.py # 🆕 IntelIntegratedDetector
│   │   ├── amd.py            # 🆕 AMDDetector
│   │   ├── cpu.py            # 🆕 CPUDetector
│   │   └── unified.py        # 🆕 UnifiedGPUDetector (coordinator)
│   │
│   ├── backends/             # 🆕 Multi-backend inference
│   │   ├── __init__.py
│   │   ├── base.py           # InferenceBackend abstract class
│   │   ├── llama_cpp.py      # 📝 Enhanced LlamaCppBackend
│   │   ├── openvino.py       # 🆕 OpenVINOBackend
│   │   ├── onnx.py           # 🆕 ONNXBackend
│   │   ├── transformers.py   # 🆕 TransformersBackend
│   │   └── selector.py       # 🆕 UnifiedInferenceEngine (coordinator)
│   │
│   ├── strategies/           # 🆕 Multi-vendor strategies
│   │   ├── __init__.py
│   │   ├── base.py           # Base strategy generator
│   │   ├── nvidia.py         # 📝 Enhanced NVIDIAStrategyGenerator
│   │   ├── intel_arc.py      # 🆕 IntelArcStrategyGenerator
│   │   ├── intel_integrated.py # 🆕 IntelIntegratedStrategyGenerator
│   │   ├── amd.py            # 🆕 AMDStrategyGenerator
│   │   ├── cpu.py            # 🆕 CPUStrategyGenerator
│   │   └── universal.py      # 🆕 UniversalStrategyGenerator (coordinator)
│   │
│   ├── squeezer.py           # 📝 Enhanced UniversalModelSqueezer
│   ├── healer.py             # 📝 Enhanced SelfHealingAgent (mostly unchanged)
│   ├── agent.py              # 📝 Enhanced UniversalNecromancerAgent
│   └── prober.py             # 📝 Enhanced UniversalNecromancerProber
│
├── tests/
│   ├── test_detectors.py     # 🆕 Test all detectors
│   ├── test_backends.py      # 🆕 Test all backends
│   ├── test_strategies.py    # 🆕 Test all strategies
│   └── test_integration.py   # 🆕 Multi-GPU integration tests
│
├── dashboard_app.py          # 📝 Update UI for vendor display
├── quickstart.py             # 📝 Add multi-vendor examples
├── requirements.txt          # 📝 Update dependencies
├── README.md                 # 📝 Update documentation
└── ARCHITECTURE.md           # 📝 Update architecture docs
```

**Legend**: 
- 🆕 = New file
- 📝 = Modified file
- No mark = Unchanged

---

## 🎯 Phase 1: Core Architecture (Week 1)

### 1.1 Define Universal Data Structures

**File**: `necromancer/core/specs.py`

```python
# Define GPUVendor, ComputeAPI, MemoryType enums
# Define universal GPUSpecs dataclass
# Support all vendor-specific fields

class GPUVendor(Enum):
    NVIDIA = "nvidia"
    INTEL_ARC = "intel_arc"
    INTEL_INTEGRATED = "intel_integrated"
    AMD = "amd"
    APPLE = "apple"
    CPU = "cpu"

class GPUSpecs:
    # Vendor-agnostic fields
    # Vendor-specific fields
    # Performance metrics
    # Support flags
```

**Effort**: 2-3 hours
**Testing**: Unit tests for all enum values

---

### 1.2 Create Detector Abstract Base Class

**File**: `necromancer/detectors/base.py`

```python
class GPUDetector(ABC):
    @abstractmethod
    def detect_devices(self) -> List[GPUSpecs]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    @abstractmethod
    def get_driver_version(self) -> str:
        pass
```

**Effort**: 1 hour
**Testing**: ABC enforcement tests

---

### 1.3 Create Backend Abstract Base Class

**File**: `necromancer/backends/base.py`

```python
class InferenceBackend(ABC):
    name: str
    supported_vendors: List[GPUVendor]
    supported_quantizations: List[int]
    requires_conversion: bool
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    @abstractmethod
    def supports_gpu(self, gpu: GPUSpecs) -> bool:
        pass
    
    @abstractmethod
    def load_model(self, path: str, bits: int, device: GPUSpecs):
        pass
    
    @abstractmethod
    def infer(self, model, inputs: Dict):
        pass
```

**Effort**: 1 hour
**Testing**: ABC enforcement tests

---

## 🎯 Phase 2: Detectors (Week 2)

### 2.1 Enhance NVIDIA Detector

**File**: `necromancer/detectors/nvidia.py`

Changes:
- Return universal `GPUSpecs` instead of proprietary class
- Add vendor field
- Add memory type field
- Keep existing pynvml logic

**Effort**: 2 hours
**Testing**: Test with various NVIDIA GPUs

---

### 2.2 Implement Intel Arc Detector

**File**: `necromancer/detectors/intel_arc.py`

Implement:
- Intel GPU Metrics API integration
- Windows WMI fallback
- Linux lspci parsing
- VRAM detection
- Clock frequency detection
- Xe-Core counting

**Effort**: 4-5 hours
**Testing**: Test with Intel Arc A770, A750, A380

---

### 2.3 Implement Intel Integrated Detector

**File**: `necromancer/detectors/intel_integrated.py`

Implement:
- Windows Device Manager querying
- Linux intel-gpu-tools parsing
- UHD Graphics / Iris Xe detection
- Shared memory reporting
- CPU integration

**Effort**: 4-5 hours
**Testing**: Test with UHD 770, Iris Xe G7

---

### 2.4 Implement AMD Detector

**File**: `necromancer/detectors/amd.py`

Implement:
- ROCm detection
- rocm-smi parsing
- RDNA generation identification
- VRAM queries

**Effort**: 3 hours
**Testing**: Test with RX 6800 XT, RX 7900

---

### 2.5 Implement CPU Detector

**File**: `necromancer/detectors/cpu.py`

Implement:
- psutil for CPU info
- System RAM measurement
- Thread count detection
- Frequency detection

**Effort**: 2 hours
**Testing**: Test on various CPUs

---

### 2.6 Create Unified Detector

**File**: `necromancer/detectors/unified.py`

```python
class UnifiedGPUDetector:
    def __init__(self):
        self.detectors = [
            NVIDIADetector(),
            IntelArcDetector(),
            IntelIntegratedDetector(),
            AMDDetector(),
            CPUDetector(),
        ]
    
    def detect_all_devices(self) -> List[GPUSpecs]:
        # Detect from all vendors, sort by performance
        pass
    
    def get_optimal_device(self) -> GPUSpecs:
        # Return best device
        pass
    
    def has_any_gpu(self) -> bool:
        # Check for non-CPU device
        pass
```

**Effort**: 2 hours
**Testing**: Integration test with mixed GPUs

---

## 🎯 Phase 3: Inference Backends (Week 3)

### 3.1 Enhance Llama-CPP Backend

**File**: `necromancer/backends/llama_cpp.py`

Changes:
- Support multiple compute APIs (CUDA, SYCL, ROCm, CPU)
- Auto-detect GPU memory availability
- Adaptive layer offloading for integrated GPUs
- Thread pool optimization

**Effort**: 3 hours
**Testing**: Test on all GPU vendors

---

### 3.2 Implement OpenVINO Backend

**File**: `necromancer/backends/openvino.py`

Implement:
- ONNX→OpenVINO IR conversion
- Device selection (GPU, CPU)
- Model loading
- Inference execution
- Memory estimation

**Effort**: 5 hours
**Testing**: Test with Intel Arc and integrated GPUs

---

### 3.3 Implement ONNX Backend

**File**: `necromancer/backends/onnx.py`

Implement:
- Multi-provider support (CUDA, ROCm, DML, CPU)
- Model loading
- Inference execution
- Provider fallback

**Effort**: 4 hours
**Testing**: Test on all vendors

---

### 3.4 Implement Transformers Backend

**File**: `necromancer/backends/transformers.py`

Implement:
- Device mapping
- Quantization config
- Model loading
- Inference

**Effort**: 3 hours
**Testing**: Test on NVIDIA and CPU

---

### 3.5 Create Unified Backend Selector

**File**: `necromancer/backends/selector.py`

```python
class UnifiedInferenceEngine:
    def __init__(self, gpu: GPUSpecs):
        self.gpu = gpu
    
    def select_backend(self) -> InferenceBackend:
        # Pick best backend for GPU vendor
        pass
    
    def load_model(self, path: str, bits: int):
        # Load with selected backend
        pass
    
    def infer(self, model, inputs):
        # Run inference
        pass
```

**Effort**: 2 hours
**Testing**: Test automatic backend selection

---

## 🎯 Phase 4: Strategy Generators (Week 3-4)

### 4.1 Enhance NVIDIA Strategy Generator

**File**: `necromancer/strategies/nvidia.py`

**Effort**: 1 hour (mostly copy existing)

---

### 4.2 Implement Intel Arc Strategy

**File**: `necromancer/strategies/intel_arc.py`

Implement:
- Discrete GPU strategy (similar to NVIDIA)
- Attention mechanism selection
- Quantization recommendation
- SYCL/OpenVINO backend selection

**Effort**: 2 hours

---

### 4.3 Implement Intel Integrated Strategy

**File**: `necromancer/strategies/intel_integrated.py`

Implement:
- Unified memory strategy (special!)
- System RAM consideration
- ONEAPI backend selection
- Performance tiers for shared memory

**Effort**: 3 hours

---

### 4.4 Implement AMD Strategy

**File**: `necromancer/strategies/amd.py`

Implement:
- RDNA generation handling
- ROCm version mapping
- Quantization strategy

**Effort**: 2 hours

---

### 4.5 Implement CPU Strategy

**File**: `necromancer/strategies/cpu.py`

Implement:
- Multi-threaded inference
- Large context support
- Quantization priorities (speed vs quality)

**Effort**: 2 hours

---

### 4.6 Create Universal Strategy Generator

**File**: `necromancer/strategies/universal.py`

```python
class UniversalStrategyGenerator:
    @staticmethod
    def generate_strategy(gpu: GPUSpecs) -> NecromancerStrategy:
        if gpu.vendor == GPUVendor.NVIDIA:
            return NVIDIAStrategyGenerator.generate(gpu)
        elif gpu.vendor == GPUVendor.INTEL_ARC:
            return IntelArcStrategyGenerator.generate(gpu)
        # ... etc
```

**Effort**: 1 hour

---

## 🎯 Phase 5: Core Updates (Week 4)

### 5.1 Update Squeezer for Integrated GPUs

**File**: `necromancer/squeezer.py`

Changes:
- Add `_calculate_integrated_gpu()` method
- Handle unified memory differently
- Consider system RAM in calculations

**Effort**: 3 hours
**Testing**: Test on Intel integrated GPU

---

### 5.2 Update Main Agent

**File**: `necromancer/agent.py`

Changes:
- Use `UnifiedGPUDetector` instead of just NVIDIA
- Handle multi-vendor strategies
- Device selection logic

**Effort**: 2 hours

---

### 5.3 Update Prober

**File**: `necromancer/prober.py`

Changes:
- Use unified detector
- Return vendor info
- Display backend selection

**Effort**: 2 hours

---

## 🎯 Phase 6: Testing (Week 4-5)

### 6.1 Unit Tests

**File**: `tests/test_detectors.py`

Test:
- Each detector independently
- All enums
- Error handling

**Effort**: 3 hours

---

### 6.2 Backend Tests

**File**: `tests/test_backends.py`

Test:
- Each backend loads/runs correctly
- Fallback chains work
- Device detection works

**Effort**: 4 hours

---

### 6.3 Integration Tests

**File**: `tests/test_integration.py`

Test:
- Full flow on various GPUs
- Auto-device selection
- Backend auto-selection
- Fallback chains

**Effort**: 4 hours

---

### 6.4 Hardware Testing

Test on:
- NVIDIA GPU (if available)
- Intel Arc (if available)
- Intel integrated GPU (most laptops)
- CPU only

**Effort**: 5 hours (depends on hardware access)

---

## 🎯 Phase 7: Documentation & Dashboard (Week 5)

### 7.1 Update README

Changes:
- Add multi-vendor support section
- Device selection examples
- Backend selection info

**Effort**: 2 hours

---

### 7.2 Update Dashboard

**File**: `dashboard_app.py`

Changes:
- Display vendor info
- Show selected backend
- Multi-GPU display

**Effort**: 3 hours

---

### 7.3 Update Quickstart

**File**: `quickstart.py`

Add examples:
- Auto-device selection
- Manual vendor preference
- Intel integrated GPU usage
- CPU-only mode

**Effort**: 2 hours

---

### 7.4 Update ARCHITECTURE.md

Describe:
- New abstraction layers
- Vendor detection flow
- Backend selection
- Integration points

**Effort**: 3 hours

---

## 📅 Total Timeline

| Phase | Component | Effort | Week |
|-------|-----------|--------|------|
| 1 | Core Architecture | 6 hrs | 1 |
| 2 | Detectors (all) | 18 hrs | 2 |
| 3 | Backends (all) | 17 hrs | 3 |
| 4 | Strategies | 12 hrs | 3-4 |
| 5 | Core Updates | 7 hrs | 4 |
| 6 | Testing | 15-20 hrs | 4-5 |
| 7 | Docs & UI | 10 hrs | 5 |
| **TOTAL** | **All** | **85-90 hrs** | **5 weeks** |

---

## 🚀 Deployment Checklist

- [ ] All detectors working on target hardware
- [ ] All backends available
- [ ] Strategy generators complete
- [ ] Unit tests passing (95%+ coverage)
- [ ] Integration tests passing
- [ ] Hardware tested (at least CPU + 1 GPU type)
- [ ] Documentation updated
- [ ] Dashboard updated
- [ ] Quickstart updated
- [ ] README updated
- [ ] Example scripts working
- [ ] Error messages clear and helpful
- [ ] Performance benchmarks documented
- [ ] Fallback chains tested

---

## 💡 Key Design Decisions

1. **Abstraction Layers**: Enables adding new vendors easily
2. **Priority-based Selection**: Always picks best available option
3. **Fallback Chains**: Never fails if partial support available
4. **Special Handling for Integrated GPUs**: Unified memory is different
5. **Multi-backend Support**: Same GPU can use different inference engines
6. **Auto-optimization**: User just needs model path, everything else automatic

---

## 🎯 Success Metrics

After implementation:

✅ Works on **any GPU** (NVIDIA, Intel Arc, integrated, AMD)
✅ Works on **CPU** as fallback
✅ **Auto-selects** best device
✅ **Auto-selects** best backend
✅ **Handles** unified memory correctly
✅ **Graceful degradation** when features unavailable
✅ **Zero configuration** needed
✅ **Clear logging** for debugging
✅ **95%+ test coverage**
✅ **Production-ready** code quality

---

## 🎓 Lessons Learned Integration

From this implementation, future LLM projects can:
1. Reference multi-vendor detection pattern
2. Use abstraction layer approach
3. Implement fallback chains
4. Handle integrated GPU memory differently
5. Support multiple inference backends

---

## 📝 Final Notes

- **Backward compatibility**: NVIDIA-only code still works
- **Gradual rollout**: Can deploy per-detector as completed
- **Community feedback**: Encourage issues for new hardware support
- **Vendor partnerships**: Consider official support from Intel, AMD
- **Future expansion**: Apple Silicon, Qualcomm Hexagon, others

---

**This roadmap transforms GPU Necromancer from a cool NVIDIA tool into a UNIVERSAL LLM inference platform! 🧟✨**

Start with Phase 1 (data structures) → it unblocks all other work!
