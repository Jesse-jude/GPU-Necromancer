"""
GPU Necromancer - Universal GPU Detectors
Multi-vendor GPU detection: NVIDIA, Intel Arc, Intel Integrated, AMD, CPU
"""

import pynvml
import psutil
import platform
import subprocess
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from necromancer_core_enums import GPUVendor, ComputeAPI, MemoryType
from necromancer_core_specs import UniversalGPUSpecs

logger = logging.getLogger(__name__)


class GPUDetectorBase(ABC):
    """Abstract base class for all GPU detectors"""
    
    @abstractmethod
    def detect_devices(self) -> List[UniversalGPUSpecs]:
        """Detect all available GPUs of this vendor"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this vendor's GPUs are available"""
        pass
    
    @abstractmethod
    def get_driver_version(self) -> str:
        """Get driver/runtime version"""
        pass


class NVIDIADetector(GPUDetectorBase):
    """Enhanced NVIDIA GPU detection using pynvml"""
    
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.initialized = True
            self.device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"✅ NVIDIA GPU detected: {self.device_count} device(s)")
        except Exception as e:
            self.initialized = False
            self.device_count = 0
            logger.debug(f"NVIDIA GPU not available: {e}")
        
        # Mapping: compute capability → (generation, tier)
        self.compute_to_generation = {
            (3, 0): ("Kepler", 1), (3, 5): ("Kepler", 1),
            (5, 0): ("Maxwell", 2), (5, 2): ("Maxwell", 2),
            (6, 0): ("Pascal", 3), (6, 1): ("Pascal", 3),
            (7, 0): ("Volta", 4),
            (7, 5): ("Turing", 5),
            (8, 0): ("Ampere", 7), (8, 6): ("Ampere", 7),
            (8, 9): ("Ada", 8),
            (9, 0): ("Hopper", 9),
        }
    
    def is_available(self) -> bool:
        return self.initialized
    
    def detect_devices(self) -> List[UniversalGPUSpecs]:
        """Detect all NVIDIA GPUs"""
        if not self.initialized:
            return []
        
        devices = []
        for i in range(self.device_count):
            try:
                spec = self._get_device_info(i)
                devices.append(spec)
            except Exception as e:
                logger.warning(f"Failed to detect NVIDIA GPU {i}: {e}")
        
        return devices
    
    def _get_device_info(self, device_id: int) -> UniversalGPUSpecs:
        """Get detailed info for specific NVIDIA GPU"""
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        
        # Basic info
        name = pynvml.nvmlDeviceGetName(handle)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        compute_capability = (major, minor)
        
        # Memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory_gb = mem_info.total / (1024**3)
        available_memory_gb = mem_info.free / (1024**3)
        
        # Architecture generation
        gen_name, gen_tier = self.compute_to_generation.get(
            compute_capability, ("Unknown", 0)
        )
        
        # Driver/CUDA version
        driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
        cuda_version_int = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        cuda_version = f"{cuda_version_int // 1000}.{(cuda_version_int % 1000) // 10}"
        
        # PCI info
        pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
        pci_bus_id = pci_info.busId.decode('utf-8')
        
        # Compute and memory info
        try:
            cuda_cores = pynvml.nvmlDeviceGetNumGpuCores(handle)
        except:
            cuda_cores = self._estimate_cuda_cores(name)
        
        # Memory bandwidth
        try:
            mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            mem_bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
            memory_bandwidth_gbps = (mem_clock * mem_bus_width * 2) / 8000
        except:
            memory_bandwidth_gbps = self._estimate_bandwidth(gen_name)
        
        # GPU clock and TFLOPS
        try:
            gpu_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        except:
            gpu_clock = 1500
        
        tflops_fp32 = (cuda_cores * gpu_clock) / 1000
        tflops_int8 = tflops_fp32 * 4
        
        return UniversalGPUSpecs(
            name=name,
            vendor=GPUVendor.NVIDIA,
            device_id=device_id,
            compute_family=gen_name,
            compute_capability=compute_capability,
            architecture_generation=gen_tier,
            memory_type=MemoryType.DEDICATED,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            dedicated_vram_gb=total_memory_gb,
            shared_system_ram_gb=0,
            compute_api=ComputeAPI.CUDA,
            max_compute_units=cuda_cores,
            max_clock_freq_mhz=gpu_clock,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            supports_fp16=True,
            supports_fp32=True,
            supports_int8=True,
            supports_int4=(compute_capability[0] >= 6),
            supports_tensor_cores=(compute_capability[0] >= 7),
            supports_unified_memory=(compute_capability[0] >= 6),
            driver_version=driver_version,
            compute_runtime_version=f"CUDA {cuda_version}",
            estimated_tflops_fp32=tflops_fp32,
            estimated_tflops_int8=tflops_int8,
            pci_bus_id=pci_bus_id,
            is_integrated=False,
        )
    
    def _estimate_cuda_cores(self, name: str) -> int:
        """Estimate CUDA cores from GPU name"""
        cuda_core_map = {
            "RTX 4090": 16384, "RTX 3090": 10496, "RTX 2080 Ti": 4352,
            "GTX 1080 Ti": 3584, "GTX 1080": 2560, "P100": 3584,
            "P40": 3840, "V100": 5120, "T4": 2560,
        }
        
        for key, cores in cuda_core_map.items():
            if key.lower() in name.lower():
                return cores
        return 2048
    
    def _estimate_bandwidth(self, generation: str) -> float:
        """Estimate memory bandwidth"""
        bandwidth_map = {
            "Pascal": 480, "Volta": 900, "Turing": 616,
            "Ampere": 1555, "Ada": 1008, "Hopper": 3350,
        }
        return bandwidth_map.get(generation, 300)
    
    def get_driver_version(self) -> str:
        """Get NVIDIA driver version"""
        if self.initialized:
            try:
                return pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            except:
                pass
        return "Unknown"
    
    def __del__(self):
        try:
            if self.initialized:
                pynvml.nvmlShutdown()
        except:
            pass


class IntelIntegratedDetector(GPUDetectorBase):
    """Intel integrated GPU detection (UHD Graphics, Iris Xe)"""
    
    def __init__(self):
        self.available = self._check_availability()
        self.system = platform.system()
        if self.available:
            logger.info("✅ Intel integrated GPU detected")
        else:
            logger.debug("Intel integrated GPU not found")
    
    def _check_availability(self) -> bool:
        """Check if Intel integrated GPU is available"""
        if platform.system() == "Windows":
            return self._check_windows_integrated_gpu()
        elif platform.system() == "Linux":
            return self._check_linux_integrated_gpu()
        return False
    
    def _check_windows_integrated_gpu(self) -> bool:
        """Check Windows for Intel integrated GPU"""
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True, text=True, timeout=5
            )
            output = result.stdout.lower()
            return any(x in output for x in ["intel uhd", "intel iris", "intel graphics"])
        except:
            return False
    
    def _check_linux_integrated_gpu(self) -> bool:
        """Check Linux for Intel integrated GPU"""
        try:
            result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5)
            output = result.stdout.lower()
            return any(x in output for x in ["intel graphics", "intel iris", "intel uhd"])
        except:
            return False
    
    def is_available(self) -> bool:
        return self.available
    
    def detect_devices(self) -> List[UniversalGPUSpecs]:
        """Detect Intel integrated GPU"""
        if not self.available:
            return []
        
        device = self._detect_integrated_gpu()
        return [device] if device else []
    
    def _detect_integrated_gpu(self) -> Optional[UniversalGPUSpecs]:
        """Detect single integrated GPU"""
        if self.system == "Windows":
            return self._detect_windows_igpu()
        elif self.system == "Linux":
            return self._detect_linux_igpu()
        return None
    
    def _detect_windows_igpu(self) -> Optional[UniversalGPUSpecs]:
        """Windows: Get integrated GPU info from WMI"""
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True, text=True, timeout=5
            )
            
            gpu_name = None
            for line in result.stdout.split('\n'):
                if any(x in line.lower() for x in ["uhd", "iris", "graphics"]):
                    gpu_name = line.strip()
                    break
            
            if not gpu_name:
                return None
            
            gen_info = self._parse_intel_igpu_name(gpu_name)
            system_ram_gb = psutil.virtual_memory().total / (1024**3)
            gpu_ram_allocation = min(system_ram_gb / 2, 8)
            
            return UniversalGPUSpecs(
                name=gpu_name,
                vendor=GPUVendor.INTEL_INTEGRATED,
                device_id=0,
                compute_family=gen_info['family'],
                compute_capability=(7, 5),
                architecture_generation=gen_info['tier'],
                memory_type=MemoryType.UNIFIED,
                total_memory_gb=gpu_ram_allocation,
                available_memory_gb=gpu_ram_allocation * 0.9,
                dedicated_vram_gb=0,
                shared_system_ram_gb=system_ram_gb,
                compute_api=ComputeAPI.ONEAPI,
                max_compute_units=gen_info['eus'] * 8,
                max_clock_freq_mhz=1200,
                memory_bandwidth_gbps=gen_info['bandwidth'],
                supports_fp16=True,
                supports_fp32=True,
                supports_int8=(gen_info['tier'] >= 2),
                supports_int4=(gen_info['tier'] >= 2),
                supports_unified_memory=True,
                driver_version=self._get_igpu_driver_version(),
                compute_runtime_version="OneAPI 2024",
                estimated_tflops_fp32=gen_info['tflops_fp32'],
                estimated_tflops_int8=gen_info['tflops_fp32'] * 2,
                is_integrated=True,
                is_mobile=True,
            )
        except Exception as e:
            logger.debug(f"Windows iGPU detection failed: {e}")
            return None
    
    def _detect_linux_igpu(self) -> Optional[UniversalGPUSpecs]:
        """Linux: Get integrated GPU info"""
        try:
            result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5)
            
            gpu_name = self._parse_lspci_for_igpu(result.stdout)
            if not gpu_name:
                gpu_name = "Intel Integrated Graphics"
            
            gen_info = self._parse_intel_igpu_name(gpu_name)
            system_ram_gb = psutil.virtual_memory().total / (1024**3)
            
            return UniversalGPUSpecs(
                name=gpu_name,
                vendor=GPUVendor.INTEL_INTEGRATED,
                device_id=0,
                compute_family=gen_info['family'],
                compute_capability=(7, 5),
                architecture_generation=gen_info['tier'],
                memory_type=MemoryType.UNIFIED,
                total_memory_gb=min(system_ram_gb / 2, 8),
                available_memory_gb=min(system_ram_gb / 2, 8) * 0.9,
                dedicated_vram_gb=0,
                shared_system_ram_gb=system_ram_gb,
                compute_api=ComputeAPI.ONEAPI,
                max_compute_units=gen_info['eus'] * 8,
                max_clock_freq_mhz=1200,
                memory_bandwidth_gbps=gen_info['bandwidth'],
                supports_fp16=True,
                supports_fp32=True,
                supports_int8=(gen_info['tier'] >= 2),
                supports_int4=(gen_info['tier'] >= 2),
                supports_unified_memory=True,
                driver_version=self._get_igpu_driver_version(),
                compute_runtime_version="OneAPI 2024",
                estimated_tflops_fp32=gen_info['tflops_fp32'],
                estimated_tflops_int8=gen_info['tflops_fp32'] * 2,
                is_integrated=True,
            )
        except Exception as e:
            logger.debug(f"Linux iGPU detection failed: {e}")
            return None
    
    def _parse_intel_igpu_name(self, name: str) -> Dict[str, Any]:
        """Parse Intel iGPU name to get specs"""
        iris_map = {
            "Iris Xe Max": {"family": "Iris Xe Max", "tier": 5, "eus": 96, "bandwidth": 96, "tflops_fp32": 60},
            "Iris Xe G7": {"family": "Iris Xe", "tier": 4, "eus": 80, "bandwidth": 80, "tflops_fp32": 40},
            "Iris Xe": {"family": "Iris Xe", "tier": 4, "eus": 80, "bandwidth": 80, "tflops_fp32": 40},
        }
        
        uhd_map = {
            "UHD Graphics 770": {"family": "UHD 770", "tier": 3, "eus": 32, "bandwidth": 32, "tflops_fp32": 10},
            "UHD Graphics 750": {"family": "UHD 750", "tier": 3, "eus": 32, "bandwidth": 32, "tflops_fp32": 10},
            "UHD Graphics 730": {"family": "UHD 730", "tier": 3, "eus": 32, "bandwidth": 32, "tflops_fp32": 10},
        }
        
        name_lower = name.lower()
        
        for key, spec in iris_map.items():
            if key.lower() in name_lower:
                return spec
        
        for key, spec in uhd_map.items():
            if key.lower() in name_lower:
                return spec
        
        return {"family": "Intel Integrated", "tier": 1, "eus": 24, "bandwidth": 16, "tflops_fp32": 5}
    
    def _parse_lspci_for_igpu(self, lspci_output: str) -> Optional[str]:
        """Extract Intel iGPU name from lspci output"""
        for line in lspci_output.split('\n'):
            if 'intel' in line.lower() and any(x in line.lower() for x in ['uhd', 'iris', 'graphics']):
                return line.strip()
        return None
    
    def _get_igpu_driver_version(self) -> str:
        """Get Intel iGPU driver version"""
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["powershell", "-c", "Get-WmiObject Win32_VideoController | Select-Object DriverVersion"],
                    capture_output=True, text=True, timeout=5
                )
                return result.stdout.strip()[:20]
            except:
                return "Unknown"
        return "Unknown"
    
    def get_driver_version(self) -> str:
        return self._get_igpu_driver_version()


class CPUDetector(GPUDetectorBase):
    """CPU fallback detector"""
    
    def is_available(self) -> bool:
        return True  # CPU always available
    
    def detect_devices(self) -> List[UniversalGPUSpecs]:
        """Always return single CPU device"""
        return [self._get_device_info()]
    
    def _get_device_info(self) -> UniversalGPUSpecs:
        """Get CPU specs"""
        cpu_count = psutil.cpu_count(logical=False) or 4
        total_ram = psutil.virtual_memory().total / (1024**3)
        available_ram = psutil.virtual_memory().available / (1024**3)
        
        freq = psutil.cpu_freq()
        max_freq = freq.max if freq else 3000
        
        tflops_fp32 = (cpu_count * max_freq / 1000) * 4
        cpu_name = platform.processor() or "CPU"
        
        return UniversalGPUSpecs(
            name=f"{cpu_name} ({cpu_count} cores)",
            vendor=GPUVendor.CPU,
            device_id=0,
            compute_family="Multi-core CPU",
            compute_capability=(0, 0),
            architecture_generation=3,
            memory_type=MemoryType.UNIFIED,
            total_memory_gb=total_ram,
            available_memory_gb=available_ram,
            dedicated_vram_gb=0,
            shared_system_ram_gb=total_ram,
            compute_api=ComputeAPI.CPU,
            max_compute_units=cpu_count,
            max_clock_freq_mhz=int(max_freq),
            memory_bandwidth_gbps=50,
            supports_fp16=False,
            supports_fp32=True,
            supports_int8=True,
            supports_int4=True,
            supports_unified_memory=False,
            driver_version="N/A",
            compute_runtime_version=platform.python_version(),
            estimated_tflops_fp32=tflops_fp32,
            estimated_tflops_int8=tflops_fp32 * 2,
            is_integrated=True,
        )
    
    def get_driver_version(self) -> str:
        return "N/A"


class UnifiedGPUDetector:
    """Unified detector that tries all vendors"""
    
    def __init__(self):
        self.detectors = [
            NVIDIADetector(),
            IntelIntegratedDetector(),
            CPUDetector(),
        ]
        logger.info("✅ Initialized UnifiedGPUDetector")
    
    def detect_all_devices(self) -> List[UniversalGPUSpecs]:
        """Detect all available GPUs across all vendors"""
        all_devices = []
        
        for detector in self.detectors:
            if detector.is_available():
                try:
                    devices = detector.detect_devices()
                    all_devices.extend(devices)
                    logger.info(f"Detected {len(devices)} {detector.__class__.__name__} device(s)")
                except Exception as e:
                    logger.warning(f"Failed to detect {detector.__class__.__name__}: {e}")
        
        # Sort by performance tier (descending)
        all_devices.sort(
            key=lambda d: (d.architecture_generation, d.total_memory_gb),
            reverse=True
        )
        
        return all_devices
    
    def get_optimal_device(self) -> Optional[UniversalGPUSpecs]:
        """Get the best available device for inference"""
        devices = self.detect_all_devices()
        if devices:
            logger.info(f"✅ Selected optimal device: {devices[0].name}")
            return devices[0]
        logger.warning("No GPU detected!")
        return None
    
    def has_any_gpu(self) -> bool:
        """Check if any GPU is available"""
        devices = self.detect_all_devices()
        return len(devices) > 0 and devices[0].vendor != GPUVendor.CPU

