"""
GPU Necromancer - Universal Inference Backends
Multi-backend support: llama-cpp, ONNX Runtime, HuggingFace Transformers
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from necromancer_core_enums import GPUVendor
from necromancer_core_specs import UniversalGPUSpecs

logger = logging.getLogger(__name__)


class InferenceBackendBase(ABC):
    """Abstract base for inference engines"""
    
    name: str
    supported_vendors: List[GPUVendor]
    supported_quantizations: List[int]
    requires_conversion: bool
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is installed and working"""
        pass
    
    @abstractmethod
    def supports_gpu(self, gpu: UniversalGPUSpecs) -> bool:
        """Check if backend supports this GPU"""
        pass
    
    @abstractmethod
    def load_model(self, model_path: str, quantization_bits: int, device: Optional[UniversalGPUSpecs] = None):
        """Load model with specific quantization"""
        pass
    
    @abstractmethod
    def infer(self, model, inputs: Dict) -> Dict:
        """Run inference"""
        pass


class LlamaCppBackend(InferenceBackendBase):
    """llama.cpp - Highly optimized for all platforms"""
    
    name = "llama-cpp-python"
    supported_vendors = [GPUVendor.CPU, GPUVendor.NVIDIA, GPUVendor.INTEL_INTEGRATED, GPUVendor.AMD]
    supported_quantizations = [4, 5, 6, 8, 16]
    requires_conversion = False
    
    def __init__(self):
        self.llama = None
        self.is_available_flag = False
        
        try:
            from llama_cpp import Llama
            self.llama = Llama
            self.is_available_flag = True
            logger.info("✅ llama-cpp-python available")
        except ImportError as e:
            logger.debug(f"llama-cpp-python not available: {e}")
    
    def is_available(self) -> bool:
        return self.is_available_flag
    
    def supports_gpu(self, gpu: UniversalGPUSpecs) -> bool:
        """Check if GPU is supported by llama-cpp"""
        if gpu.vendor == GPUVendor.NVIDIA:
            return True  # CUDA support
        elif gpu.vendor == GPUVendor.INTEL_INTEGRATED:
            return True  # CPU backend
        elif gpu.vendor == GPUVendor.CPU:
            return True
        elif gpu.vendor == GPUVendor.AMD:
            return True  # ROCm support
        return False
    
    def load_model(self, model_path: str, quantization_bits: int, device: Optional[UniversalGPUSpecs] = None):
        """Load GGUF model"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if device:
            n_gpu_layers = self._calculate_gpu_layers(device, quantization_bits)
        else:
            n_gpu_layers = 0
        
        import psutil
        n_threads = min(psutil.cpu_count(logical=False) or 4, 16)
        
        logger.info(f"Loading GGUF model: {model_path}")
        logger.info(f"  GPU layers: {n_gpu_layers}, Threads: {n_threads}")
        
        try:
            model = self.llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                n_ctx=4096,
                verbose=False,
            )
            return LlamaCppModel(model, self, device)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _calculate_gpu_layers(self, device: UniversalGPUSpecs, quant_bits: int) -> int:
        """Calculate how many layers to offload to GPU"""
        if device.is_integrated:
            return 10  # Integrated: fewer layers
        else:
            available_vram = device.available_memory_gb * 0.80
            bytes_per_layer = 200 * (quant_bits / 8)
            max_layers = int(available_vram * 1024 / bytes_per_layer)
            return min(max_layers, 80)
    
    def infer(self, model, inputs: Dict[str, Any]) -> Dict:
        """Run inference"""
        prompt = inputs.get('prompt', '')
        max_tokens = inputs.get('max_tokens', 256)
        temperature = inputs.get('temperature', 0.7)
        top_p = inputs.get('top_p', 0.9)
        
        try:
            output = model.llama_model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            return {
                'text': output['choices'][0]['text'],
                'tokens': output['usage']['completion_tokens'],
                'prompt_tokens': output['usage']['prompt_tokens'],
            }
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise


class LlamaCppModel:
    """Wrapper for loaded llama-cpp model"""
    def __init__(self, llama_model, backend, device):
        self.llama_model = llama_model
        self.backend = backend
        self.device = device


class ONNXBackend(InferenceBackendBase):
    """ONNX Runtime - Multi-vendor support"""
    
    name = "onnx-runtime"
    supported_vendors = [GPUVendor.CPU, GPUVendor.NVIDIA, GPUVendor.INTEL_INTEGRATED, GPUVendor.AMD]
    supported_quantizations = [8, 16]
    requires_conversion = True
    
    def __init__(self):
        self.ort = None
        self.is_available_flag = False
        self.available_providers = []
        
        try:
            import onnxruntime
            self.ort = onnxruntime
            self.available_providers = onnxruntime.get_available_providers()
            self.is_available_flag = True
            logger.info(f"✅ ONNX Runtime available. Providers: {self.available_providers}")
        except ImportError:
            logger.debug("ONNX Runtime not installed")
    
    def is_available(self) -> bool:
        return self.is_available_flag
    
    def supports_gpu(self, gpu: UniversalGPUSpecs) -> bool:
        """Check if GPU is supported by ONNX Runtime"""
        if gpu.vendor == GPUVendor.NVIDIA:
            return "CUDAExecutionProvider" in self.available_providers
        elif gpu.vendor == GPUVendor.INTEL_INTEGRATED:
            return "CPUExecutionProvider" in self.available_providers
        elif gpu.vendor == GPUVendor.CPU:
            return "CPUExecutionProvider" in self.available_providers
        return False
    
    def load_model(self, model_path: str, quantization_bits: int, device: Optional[UniversalGPUSpecs] = None):
        """Load ONNX model"""
        if not self.is_available():
            raise RuntimeError("ONNX Runtime not available")
        
        providers = self._select_providers(device)
        logger.info(f"Using ONNX providers: {providers}")
        
        try:
            session = self.ort.InferenceSession(model_path, providers=providers)
            return ONNXModel(session, self, device)
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def _select_providers(self, gpu: Optional[UniversalGPUSpecs]) -> list:
        """Select execution providers based on GPU"""
        if gpu:
            if gpu.vendor == GPUVendor.NVIDIA:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif gpu.vendor == GPUVendor.INTEL_INTEGRATED:
                return ["CPUExecutionProvider"]
            elif gpu.vendor == GPUVendor.CPU:
                return ["CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    
    def infer(self, model, inputs: Dict[str, Any]) -> Dict:
        """Run inference"""
        import numpy as np
        
        input_names = [input.name for input in model.session.get_inputs()]
        input_data = {}
        
        for name in input_names:
            if name in inputs:
                input_data[name] = np.array(inputs[name])
        
        try:
            outputs = model.session.run(None, input_data)
            return {'output': outputs[0]}
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            raise


class ONNXModel:
    """Wrapper for ONNX model"""
    def __init__(self, session, backend, device):
        self.session = session
        self.backend = backend
        self.device = device


class UnifiedInferenceEngine:
    """Unified inference with automatic backend selection"""
    
    def __init__(self, gpu: UniversalGPUSpecs):
        self.gpu = gpu
        self.backends = {
            'llama-cpp': LlamaCppBackend(),
            'onnx': ONNXBackend(),
        }
        self.active_backend = None
        logger.info(f"Initialized InferenceEngine for {gpu.name}")
    
    def select_backend(self, model_format: str = "auto") -> InferenceBackendBase:
        """Select best available backend for GPU"""
        
        priority = self._get_backend_priority(self.gpu)
        
        for backend_name in priority:
            if backend_name not in self.backends:
                continue
            
            backend = self.backends[backend_name]
            
            if backend.is_available() and backend.supports_gpu(self.gpu):
                logger.info(f"✅ Selected backend: {backend_name} for {self.gpu.name}")
                self.active_backend = backend
                return backend
        
        raise RuntimeError(f"No suitable backend found for {self.gpu.vendor}")
    
    def _get_backend_priority(self, gpu: UniversalGPUSpecs) -> list:
        """Get backend priority list based on GPU vendor"""
        if gpu.vendor == GPUVendor.NVIDIA:
            return ['llama-cpp', 'onnx']
        elif gpu.vendor == GPUVendor.INTEL_INTEGRATED:
            return ['llama-cpp', 'onnx']
        elif gpu.vendor == GPUVendor.CPU:
            return ['llama-cpp', 'onnx']
        elif gpu.vendor == GPUVendor.AMD:
            return ['llama-cpp', 'onnx']
        else:
            return ['llama-cpp', 'onnx']
    
    def load_model(self, model_path: str, quantization_bits: int = 8):
        """Load model with selected backend"""
        if not self.active_backend:
            self.select_backend()
        
        return self.active_backend.load_model(model_path, quantization_bits, self.gpu)
    
    def infer(self, model, inputs: Dict) -> Dict:
        """Run inference"""
        if not self.active_backend:
            self.select_backend()
        
        return self.active_backend.infer(model, inputs)

