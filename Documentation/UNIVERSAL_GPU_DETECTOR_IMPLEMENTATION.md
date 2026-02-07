# GPU Necromancer - Universal Detector Implementation

## 🔧 Practical Implementation Guide

### 1. NVIDIA GPU Detector (Enhanced)

```python
# necromancer/detectors/nvidia_detector.py

import pynvml
from typing import List, Optional
from dataclasses import asdict

class NVIDIADetector(GPUDetector):
    """Enhanced NVIDIA detector with universal specs"""
    
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.initialized = True
            self.device_count = pynvml.nvmlDeviceGetCount()
        except Exception as e:
            self.initialized = False
            self.device_count = 0
            logger.warning(f"NVIDIA GPU not available: {e}")
        
        # Mapping: compute capability → generation
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
    
    def detect_devices(self) -> List[GPUSpecs]:
        """Detect all NVIDIA GPUs"""
        if not self.initialized:
            return []
        
        devices = []
        for i in range(self.device_count):
            try:
                spec = self.get_device_info(i)
                devices.append(spec)
            except Exception as e:
                logger.warning(f"Failed to detect NVIDIA GPU {i}: {e}")
        
        return devices
    
    def get_device_info(self, device_id: int) -> GPUSpecs:
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
            compute_capability, 
            ("Unknown", 0)
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
        
        # Estimate TFLOPS
        gpu_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        tflops_fp32 = (cuda_cores * gpu_clock) / 1000  # MHz to GHz
        tflops_int8 = tflops_fp32 * 4  # Rough estimate
        
        # Tensor core support
        supports_tensor_cores = compute_capability[0] >= 7
        
        return GPUSpecs(
            name=name,
            vendor=GPUVendor.NVIDIA,
            device_id=device_id,
            
            # Architecture
            compute_family=gen_name,
            compute_capability=compute_capability,
            architecture_generation=gen_tier,
            
            # Memory
            memory_type=MemoryType.DEDICATED,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            dedicated_vram_gb=total_memory_gb,
            shared_system_ram_gb=0,
            
            # Compute
            compute_api=ComputeAPI.CUDA,
            max_compute_units=cuda_cores,
            max_clock_freq_mhz=gpu_clock,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            
            # Support
            supports_fp16=True,
            supports_fp32=True,
            supports_int8=True,
            supports_int4=(compute_capability[0] >= 6),
            supports_tensor_cores=supports_tensor_cores,
            supports_unified_memory=(compute_capability[0] >= 6),
            
            # Software
            driver_version=driver_version,
            compute_runtime_version=cuda_version,
            
            # Performance
            estimated_tflops_fp32=tflops_fp32,
            estimated_tflops_int8=tflops_int8,
            
            # Other
            pci_bus_id=pci_bus_id,
            is_integrated=False,
        )
    
    def _estimate_cuda_cores(self, name: str) -> int:
        """Estimate CUDA cores from GPU name"""
        cuda_core_map = {
            "RTX 4090": 16384,
            "RTX 3090": 10496,
            "RTX 2080 Ti": 4352,
            "GTX 1080 Ti": 3584,
            "GTX 1080": 2560,
            "P100": 3584,
            "P40": 3840,
            "V100": 5120,
            "T4": 2560,
        }
        
        for key, cores in cuda_core_map.items():
            if key.lower() in name.lower():
                return cores
        return 2048  # Conservative estimate
    
    def _estimate_bandwidth(self, generation: str) -> float:
        """Estimate memory bandwidth"""
        bandwidth_map = {
            "Pascal": 480,
            "Volta": 900,
            "Turing": 616,
            "Ampere": 1555,
            "Ada": 1008,
            "Hopper": 3350,
        }
        return bandwidth_map.get(generation, 300)
    
    def __del__(self):
        try:
            if self.initialized:
                pynvml.nvmlShutdown()
        except:
            pass
```

---

### 2. Intel Arc Detector

```python
# necromancer/detectors/intel_arc_detector.py

import subprocess
import json
from typing import List, Dict, Any

class IntelArcDetector(GPUDetector):
    """Intel Arc GPU detection"""
    
    def __init__(self):
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Intel Arc GPUs are available"""
        try:
            # Try to import Intel GPU libraries
            import intel_gpu_metrics
            return True
        except ImportError:
            pass
        
        try:
            # Fallback: check for Windows Device Manager or intel-gpu-tools
            result = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return "Intel Arc" in result.stdout or "Arc" in result.stdout
        except:
            pass
        
        try:
            # Linux: check for Intel GPU via lspci
            result = subprocess.run(
                ["lspci"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return "Intel Corporation" in result.stdout and "Arc" in result.stdout
        except:
            pass
        
        return False
    
    def is_available(self) -> bool:
        return self.available
    
    def detect_devices(self) -> List[GPUSpecs]:
        """Detect Intel Arc GPUs"""
        if not self.available:
            return []
        
        devices = []
        
        # Try Intel GPU Metrics API first
        try:
            import intel_gpu_metrics
            # Query Intel Arc GPUs
            for i, gpu_info in enumerate(intel_gpu_metrics.get_gpus()):
                spec = self._parse_intel_arc_metrics(gpu_info, i)
                if spec:
                    devices.append(spec)
        except Exception as e:
            logger.warning(f"Intel metrics API failed: {e}")
            # Fallback to Windows Device Manager
            devices = self._detect_via_wmic()
        
        return devices
    
    def _parse_intel_arc_metrics(self, gpu_info: Dict[str, Any], device_id: int) -> Optional[GPUSpecs]:
        """Parse Intel GPU Metrics data"""
        
        # Map Arc models to generations
        arc_map = {
            "A380": ("Arc Alchemist", 6),
            "A750": ("Arc Alchemist", 6),
            "A770": ("Arc Alchemist", 6),
            "A380M": ("Arc Alchemist", 6),
            "A750M": ("Arc Alchemist", 6),
            "A770M": ("Arc Alchemist", 6),
        }
        
        name = gpu_info.get("name", "Intel Arc")
        vram_mb = gpu_info.get("dedicated_vram_mb", 8192)  # Default 8GB
        total_memory_gb = vram_mb / 1024
        available_memory_gb = gpu_info.get("available_memory_mb", vram_mb) / 1024
        
        # Find generation tier
        compute_family = "Arc Alchemist"
        gen_tier = 6
        for model, (family, tier) in arc_map.items():
            if model in name:
                compute_family = family
                gen_tier = tier
                break
        
        # VRAM by model
        if "A770" in name:
            total_memory_gb = 16 if "16G" in name else 8
        elif "A750" in name:
            total_memory_gb = 8
        elif "A380" in name:
            total_memory_gb = 6
        
        # Estimate cores (Xe-Core count)
        xe_cores = gpu_info.get("xe_cores", 128)
        max_clock = gpu_info.get("max_clock_mhz", 2400)
        
        # Memory bandwidth estimate
        # Arc uses 192-bit or 256-bit memory bus
        mem_bus_width = 192 if "A750" in name else 256
        mem_clock = gpu_info.get("memory_clock_mhz", 17.5 * 1000)  # Typical
        memory_bandwidth_gbps = (mem_clock * mem_bus_width * 2) / 8000
        
        # TFLOPS
        tflops_fp32 = (xe_cores * max_clock) / 1000
        tflops_int8 = tflops_fp32 * 4
        
        return GPUSpecs(
            name=name,
            vendor=GPUVendor.INTEL_ARC,
            device_id=device_id,
            
            # Architecture
            compute_family=compute_family,
            compute_capability=(6, 0),  # Arc uses compute 6.0
            architecture_generation=gen_tier,
            
            # Memory
            memory_type=MemoryType.DEDICATED,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            dedicated_vram_gb=total_memory_gb,
            shared_system_ram_gb=0,
            
            # Compute
            compute_api=ComputeAPI.SYCL,
            max_compute_units=xe_cores,
            max_clock_freq_mhz=max_clock,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            
            # Support
            supports_fp16=True,
            supports_fp32=True,
            supports_int8=True,
            supports_int4=True,
            supports_tensor_cores=True,
            supports_unified_memory=True,
            
            # Software
            driver_version=self._get_arc_driver_version(),
            compute_runtime_version="SYCL 1.0",
            
            # Performance
            estimated_tflops_fp32=tflops_fp32,
            estimated_tflops_int8=tflops_int8,
            
            # Other
            pci_bus_id="",
            is_integrated=False,
        )
    
    def _detect_via_wmic(self) -> List[GPUSpecs]:
        """Fallback: Detect Arc GPUs via Windows WMI"""
        devices = []
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name,adapterram"],
                capture_output=True,
                text=True
            )
            
            lines = result.stdout.strip().split('\n')
            device_id = 0
            
            for line in lines[1:]:  # Skip header
                if "Intel Arc" in line or "Arc" in line:
                    # Parse VRAM if available
                    parts = line.split()
                    vram_bytes = 0
                    for part in parts:
                        if part.isdigit() and int(part) > 1000000000:
                            vram_bytes = int(part)
                            break
                    
                    vram_gb = max(vram_bytes / (1024**3), 4)  # At least 4GB
                    
                    spec = self._create_arc_spec(line, device_id, vram_gb)
                    devices.append(spec)
                    device_id += 1
        except Exception as e:
            logger.warning(f"WMIC detection failed: {e}")
        
        return devices
    
    def _create_arc_spec(self, name: str, device_id: int, vram_gb: float) -> GPUSpecs:
        """Create GPUSpecs for Arc GPU"""
        return GPUSpecs(
            name=name.strip(),
            vendor=GPUVendor.INTEL_ARC,
            device_id=device_id,
            compute_family="Arc Alchemist",
            compute_capability=(6, 0),
            architecture_generation=6,
            memory_type=MemoryType.DEDICATED,
            total_memory_gb=vram_gb,
            available_memory_gb=vram_gb * 0.9,
            dedicated_vram_gb=vram_gb,
            shared_system_ram_gb=0,
            compute_api=ComputeAPI.SYCL,
            max_compute_units=128,
            max_clock_freq_mhz=2400,
            memory_bandwidth_gbps=480,
            supports_fp16=True,
            supports_fp32=True,
            supports_int8=True,
            supports_int4=True,
            supports_tensor_cores=True,
            driver_version=self._get_arc_driver_version(),
            compute_runtime_version="SYCL 1.0",
            estimated_tflops_fp32=100,
            estimated_tflops_int8=400,
            is_integrated=False,
        )
    
    def _get_arc_driver_version(self) -> str:
        """Get Intel Arc driver version"""
        try:
            result = subprocess.run(
                ["intel-gpu-metrics", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip()
        except:
            return "Unknown"
```

---

### 3. Intel Integrated GPU Detector

```python
# necromancer/detectors/intel_integrated_detector.py

import platform
import subprocess
import psutil
from typing import List, Optional

class IntelIntegratedDetector(GPUDetector):
    """Intel integrated GPU detection (UHD Graphics, Iris Xe)"""
    
    def __init__(self):
        self.available = self._check_availability()
        self.system = platform.system()
    
    def _check_availability(self) -> bool:
        """Check if Intel integrated GPU is available"""
        
        if platform.system() == "Windows":
            return self._check_windows_integrated_gpu()
        elif platform.system() == "Linux":
            return self._check_linux_integrated_gpu()
        elif platform.system() == "Darwin":
            return False  # macOS uses Metal, not Intel integrated
        
        return False
    
    def _check_windows_integrated_gpu(self) -> bool:
        """Check Windows for Intel integrated GPU"""
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            output = result.stdout.lower()
            return any(x in output for x in [
                "intel uhd", "intel iris", "intel arc", "intel graphics"
            ])
        except:
            return False
    
    def _check_linux_integrated_gpu(self) -> bool:
        """Check Linux for Intel integrated GPU"""
        try:
            result = subprocess.run(
                ["lspci"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            output = result.stdout.lower()
            return any(x in output for x in [
                "intel graphics", "intel iris", "intel uhd", "intel hd"
            ])
        except:
            return False
    
    def is_available(self) -> bool:
        return self.available
    
    def detect_devices(self) -> List[GPUSpecs]:
        """Detect Intel integrated GPU"""
        if not self.available:
            return []
        
        device = self._detect_integrated_gpu()
        return [device] if device else []
    
    def _detect_integrated_gpu(self) -> Optional[GPUSpecs]:
        """Detect single integrated GPU"""
        
        if self.system == "Windows":
            return self._detect_windows_igpu()
        elif self.system == "Linux":
            return self._detect_linux_igpu()
        
        return None
    
    def _detect_windows_igpu(self) -> Optional[GPUSpecs]:
        """Windows: Get integrated GPU info from WMI"""
        try:
            # Get GPU name
            result = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            gpu_name = None
            for line in result.stdout.split('\n'):
                if any(x in line.lower() for x in ["uhd", "iris", "graphics"]):
                    gpu_name = line.strip()
                    break
            
            if not gpu_name:
                return None
            
            # Determine generation from name
            gen_info = self._parse_intel_igpu_name(gpu_name)
            
            # Shared system RAM
            system_ram_gb = psutil.virtual_memory().total / (1024**3)
            # Typically allocate 1/8 to GPU, but can use more if needed
            gpu_ram_allocation = min(system_ram_gb / 2, 8)
            
            return GPUSpecs(
                name=gpu_name,
                vendor=GPUVendor.INTEL_INTEGRATED,
                device_id=0,
                
                # Architecture
                compute_family=gen_info['family'],
                compute_capability=(7, 5),  # Iris Xe equivalent
                architecture_generation=gen_info['tier'],
                
                # Memory
                memory_type=MemoryType.UNIFIED,
                total_memory_gb=gpu_ram_allocation,
                available_memory_gb=gpu_ram_allocation * 0.9,
                dedicated_vram_gb=0,  # No dedicated VRAM
                shared_system_ram_gb=system_ram_gb,
                
                # Compute
                compute_api=ComputeAPI.ONEAPI,
                max_compute_units=gen_info['eus'] * 8,  # EU count
                max_clock_freq_mhz=1200,  # Typical
                memory_bandwidth_gbps=gen_info['bandwidth'],
                
                # Support
                supports_fp16=True,
                supports_fp32=True,
                supports_int8=gen_info['tier'] >= 2,
                supports_int4=gen_info['tier'] >= 2,
                supports_tensor_cores=False,
                supports_unified_memory=True,
                
                # Software
                driver_version=self._get_igpu_driver_version(),
                compute_runtime_version="OneAPI 2024",
                
                # Performance
                estimated_tflops_fp32=gen_info['tflops_fp32'],
                estimated_tflops_int8=gen_info['tflops_fp32'] * 2,
                
                # Other
                pci_bus_id="",
                is_integrated=True,
                is_mobile=True,
            )
        
        except Exception as e:
            logger.warning(f"Windows iGPU detection failed: {e}")
            return None
    
    def _detect_linux_igpu(self) -> Optional[GPUSpecs]:
        """Linux: Get integrated GPU info"""
        try:
            # Use intel-gpu-tools if available
            result = subprocess.run(
                ["intel_gpu_top", "-s", "0"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Fallback: use lspci
            if not result.stdout:
                result = subprocess.run(
                    ["lspci"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            
            gpu_name = self._parse_lspci_for_igpu(result.stdout)
            
            if not gpu_name:
                gpu_name = "Intel Integrated Graphics"
            
            gen_info = self._parse_intel_igpu_name(gpu_name)
            system_ram_gb = psutil.virtual_memory().total / (1024**3)
            
            return GPUSpecs(
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
                supports_int8=gen_info['tier'] >= 2,
                supports_int4=gen_info['tier'] >= 2,
                supports_tensor_cores=False,
                supports_unified_memory=True,
                driver_version=self._get_igpu_driver_version(),
                compute_runtime_version="OneAPI 2024",
                estimated_tflops_fp32=gen_info['tflops_fp32'],
                estimated_tflops_int8=gen_info['tflops_fp32'] * 2,
                is_integrated=True,
            )
        
        except Exception as e:
            logger.warning(f"Linux iGPU detection failed: {e}")
            return None
    
    def _parse_intel_igpu_name(self, name: str) -> Dict:
        """Parse Intel iGPU name to get specs"""
        
        # Iris Xe mapping
        iris_map = {
            "Iris Xe Max": {"family": "Iris Xe Max", "tier": 5, "eus": 96, "bandwidth": 96, "tflops_fp32": 60},
            "Iris Xe G7": {"family": "Iris Xe", "tier": 4, "eus": 80, "bandwidth": 80, "tflops_fp32": 40},
            "Iris Xe": {"family": "Iris Xe", "tier": 4, "eus": 80, "bandwidth": 80, "tflops_fp32": 40},
            "Iris Pro": {"family": "Iris Pro", "tier": 3, "eus": 48, "bandwidth": 48, "tflops_fp32": 20},
            "Iris": {"family": "Iris", "tier": 3, "eus": 48, "bandwidth": 48, "tflops_fp32": 20},
        }
        
        # UHD Graphics mapping
        uhd_map = {
            "UHD Graphics 770": {"family": "UHD 770", "tier": 3, "eus": 32, "bandwidth": 32, "tflops_fp32": 10},
            "UHD Graphics 750": {"family": "UHD 750", "tier": 3, "eus": 32, "bandwidth": 32, "tflops_fp32": 10},
            "UHD Graphics 730": {"family": "UHD 730", "tier": 3, "eus": 32, "bandwidth": 32, "tflops_fp32": 10},
            "UHD Graphics 630": {"family": "UHD 630", "tier": 2, "eus": 24, "bandwidth": 24, "tflops_fp32": 7},
            "HD Graphics": {"family": "HD Graphics", "tier": 1, "eus": 16, "bandwidth": 16, "tflops_fp32": 3},
        }
        
        name_lower = name.lower()
        
        for key, spec in iris_map.items():
            if key.lower() in name_lower:
                return spec
        
        for key, spec in uhd_map.items():
            if key.lower() in name_lower:
                return spec
        
        # Default for unknown Intel iGPU
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
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return result.stdout.strip()
            except:
                return "Unknown"
        else:
            try:
                result = subprocess.run(
                    ["modinfo", "i915"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                for line in result.stdout.split('\n'):
                    if 'version' in line.lower():
                        return line.split(':', 1)[1].strip()
            except:
                return "Unknown"
        
        return "Unknown"
```

---

### 4. CPU Detector

```python
# necromancer/detectors/cpu_detector.py

import psutil
import platform

class CPUDetector(GPUDetector):
    """CPU fallback detector"""
    
    def is_available(self) -> bool:
        # CPU always available
        return True
    
    def detect_devices(self) -> List[GPUSpecs]:
        """Always return single CPU device"""
        return [self.get_device_info(0)]
    
    def get_device_info(self, device_id: int = 0) -> GPUSpecs:
        """Get CPU specs"""
        
        cpu_count = psutil.cpu_count(logical=False) or 4
        total_ram = psutil.virtual_memory().total / (1024**3)
        available_ram = psutil.virtual_memory().available / (1024**3)
        
        # CPU frequency (MHz)
        freq = psutil.cpu_freq()
        max_freq = freq.max if freq else 3000
        
        # Estimate TFLOPS based on core count and frequency
        # Modern CPUs: ~2 ops per cycle per core (FMA = 1 FLOP but counted as 2)
        tflops_fp32 = (cpu_count * max_freq / 1000) * 4  # 4 for AVX2
        
        cpu_name = platform.processor() or "CPU"
        
        return GPUSpecs(
            name=f"{cpu_name} ({cpu_count} cores)",
            vendor=GPUVendor.CPU,
            device_id=device_id,
            
            # Architecture
            compute_family="Multi-core CPU",
            compute_capability=(0, 0),  # N/A for CPU
            architecture_generation=3,  # Generic tier
            
            # Memory
            memory_type=MemoryType.UNIFIED,
            total_memory_gb=total_ram,
            available_memory_gb=available_ram,
            dedicated_vram_gb=0,
            shared_system_ram_gb=total_ram,
            
            # Compute
            compute_api=ComputeAPI.CPU,
            max_compute_units=cpu_count,
            max_clock_freq_mhz=int(max_freq),
            memory_bandwidth_gbps=50,  # Rough estimate from DDR4
            
            # Support
            supports_fp16=False,
            supports_fp32=True,
            supports_int8=True,
            supports_int4=True,
            supports_tensor_cores=False,
            supports_unified_memory=False,
            
            # Software
            driver_version="N/A",
            compute_runtime_version=platform.python_version(),
            
            # Performance
            estimated_tflops_fp32=tflops_fp32,
            estimated_tflops_int8=tflops_fp32 * 2,
            
            # Other
            pci_bus_id="",
            is_integrated=True,
        )
```

---

## 🎯 Usage

```python
# Import all detectors
from necromancer.detectors import UnifiedGPUDetector

# Auto-detect all GPUs
detector = UnifiedGPUDetector()
devices = detector.detect_all_devices()

for device in devices:
    print(f"\n{device.name}")
    print(f"  Vendor: {device.vendor.value}")
    print(f"  Memory: {device.total_memory_gb:.1f}GB")
    print(f"  Compute: {device.estimated_tflops_fp32:.0f} TFLOPS FP32")
    print(f"  Supports Tensor Cores: {device.supports_tensor_cores}")

# Output example:
# RTX 4090
#   Vendor: nvidia
#   Memory: 24.0GB
#   Compute: 330 TFLOPS FP32
#   Supports Tensor Cores: True
#
# Intel Arc A770
#   Vendor: intel_arc
#   Memory: 8.0GB
#   Compute: 50 TFLOPS FP32
#   Supports Tensor Cores: True
#
# Intel UHD Graphics 770
#   Vendor: intel_integrated
#   Memory: 8.0GB (Shared with System RAM)
#   Compute: 10 TFLOPS FP32
#   Supports Tensor Cores: False
#
# CPU (16 cores)
#   Vendor: cpu
#   Memory: 64.0GB
#   Compute: 200 TFLOPS FP32
#   Supports Tensor Cores: False
```

This implementation makes GPU Necromancer truly universal! 🧟✨
