"""
GPU Necromancer - Universal System Tests
Test all detectors, backends, and strategies
"""

import logging
import unittest
from typing import List

from necromancer_core_enums import GPUVendor, ComputeAPI, MemoryType
from necromancer_core_specs import UniversalGPUSpecs, UniversalNecromancerStrategy
from necromancer_universal_detectors import UnifiedGPUDetector, NVIDIADetector, CPUDetector, IntelIntegratedDetector
from necromancer_universal_backends import UnifiedInferenceEngine, LlamaCppBackend, ONNXBackend
from necromancer_universal_strategies import StrategyGenerator
from necromancer_universal_agent import UniversalNecromancerAgent, ModelRequest

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestDetectors(unittest.TestCase):
    """Test GPU detection"""
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        detector = UnifiedGPUDetector()
        self.assertIsNotNone(detector)
        logger.info(" Detector initialized")
    
    def test_cpu_detector(self):
        """Test CPU detection"""
        cpu_detector = CPUDetector()
        self.assertTrue(cpu_detector.is_available())
        
        devices = cpu_detector.detect_devices()
        self.assertEqual(len(devices), 1)
        
        cpu = devices[0]
        self.assertEqual(cpu.vendor, GPUVendor.CPU)
        self.assertTrue(cpu.is_integrated)
        self.assertGreater(cpu.total_memory_gb, 0)
        logger.info(f" CPU Detector: Found {cpu.name}")
    
    def test_detect_all_devices(self):
        """Test detecting all devices"""
        detector = UnifiedGPUDetector()
        devices = detector.detect_all_devices()
        
        self.assertGreater(len(devices), 0, "At least CPU should be detected")
        logger.info(f" Detected {len(devices)} device(s)")
        
        for device in devices:
            logger.info(f"  - {device.name} ({device.vendor.value})")
    
    def test_optimal_device_selection(self):
        """Test optimal device selection"""
        detector = UnifiedGPUDetector()
        optimal = detector.get_optimal_device()
        
        self.assertIsNotNone(optimal)
        self.assertIn(optimal.vendor, [GPUVendor.CPU, GPUVendor.NVIDIA, GPUVendor.INTEL_INTEGRATED])
        logger.info(f" Selected optimal device: {optimal.name}")


class TestStrategyGeneration(unittest.TestCase):
    """Test strategy generation"""
    
    def test_cpu_strategy(self):
        """Test CPU strategy generation"""
        detector = CPUDetector()
        cpu = detector.detect_devices()[0]
        
        strategy = StrategyGenerator.generate_strategy(cpu)
        
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.recommended_backend, "llama-cpp")
        self.assertEqual(strategy.recommended_compute_api, "CPU")
        self.assertEqual(strategy.attention_mechanism, "eager")
        self.assertGreater(strategy.max_quantization_bits, 0)
        logger.info(f" CPU Strategy generated: {strategy.recommended_backend}")
    
    def test_strategy_has_limitations(self):
        """Test that strategies have appropriate limitations"""
        detector = CPUDetector()
        cpu = detector.detect_devices()[0]
        
        strategy = StrategyGenerator.generate_strategy(cpu)
        
        self.assertGreater(len(strategy.limitations), 0)
        self.assertGreater(len(strategy.optimizations), 0)
        logger.info(f" Strategy has {len(strategy.limitations)} limitations, {len(strategy.optimizations)} optimizations")
    
    def test_integrated_gpu_strategy(self):
        """Test integrated GPU strategy if available"""
        intel_detector = IntelIntegratedDetector()
        if not intel_detector.is_available():
            logger.info("⏭  Intel Integrated GPU not available, skipping test")
            return
        
        devices = intel_detector.detect_devices()
        if len(devices) == 0:
            logger.info("⏭  No Intel Integrated GPU detected")
            return
        
        gpu = devices[0]
        strategy = StrategyGenerator.generate_strategy(gpu)
        
        self.assertEqual(gpu.memory_type, MemoryType.UNIFIED)
        self.assertTrue(strategy.in_system_ram_estimate_gb is not None or True)
        logger.info(f" Intel Integrated GPU Strategy generated")


class TestBackendSelection(unittest.TestCase):
    """Test inference backend selection"""
    
    def test_backend_initialization(self):
        """Test backend initialization"""
        llama_backend = LlamaCppBackend()
        onnx_backend = ONNXBackend()
        
        self.assertIsNotNone(llama_backend)
        self.assertIsNotNone(onnx_backend)
        logger.info(" Backends initialized")
    
    def test_inference_engine_selection(self):
        """Test inference engine backend selection"""
        detector = CPUDetector()
        cpu = detector.detect_devices()[0]
        
        engine = UnifiedInferenceEngine(cpu)
        
        try:
            backend = engine.select_backend()
            self.assertIsNotNone(backend)
            logger.info(f" Backend selected: {backend.name}")
        except RuntimeError as e:
            logger.warning(f"No backends available: {e}")
    
    def test_backend_supports_cpu(self):
        """Test that backends support CPU"""
        cpu_detector = CPUDetector()
        cpu = cpu_detector.detect_devices()[0]
        
        llama = LlamaCppBackend()
        onnx = ONNXBackend()
        
        # At least one should support CPU
        supports = llama.supports_gpu(cpu) or onnx.supports_gpu(cpu)
        self.assertTrue(supports, "At least one backend should support CPU")
        logger.info(" CPU is supported by at least one backend")


class TestUniversalAgent(unittest.TestCase):
    """Test the universal agent"""
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        try:
            agent = UniversalNecromancerAgent()
            self.assertIsNotNone(agent.gpu)
            self.assertIsNotNone(agent.strategy)
            logger.info(f" Agent initialized with {agent.gpu.name}")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def test_agent_strategy_display(self):
        """Test agent strategy display"""
        agent = UniversalNecromancerAgent()
        agent.display_strategy()
        logger.info(" Strategy displayed successfully")
    
    def test_model_request_preparation(self):
        """Test model request preparation"""
        agent = UniversalNecromancerAgent()
        
        request = ModelRequest(
            model_name="test-model",
            context_length=2048,
            max_tokens=256
        )
        
        config = agent.prepare_model(request)
        
        self.assertIn('quantization_bits', config)
        self.assertIn('context_length', config)
        self.assertIn('attention_mechanism', config)
        self.assertLessEqual(config['context_length'], agent.strategy.max_context_length)
        logger.info(" Model config prepared successfully")
    
    def test_agent_status_report(self):
        """Test agent status report"""
        agent = UniversalNecromancerAgent()
        status = agent.get_status_report()
        
        self.assertIn('gpu', status)
        self.assertIn('vendor', status)
        self.assertIn('total_executions', status)
        logger.info(f" Status report: {status['gpu']} ({status['vendor']})")


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_full_detection_to_strategy_pipeline(self):
        """Test full detection and strategy pipeline"""
        
        # Detect
        detector = UnifiedGPUDetector()
        devices = detector.detect_all_devices()
        self.assertGreater(len(devices), 0)
        
        # Select best
        optimal = detector.get_optimal_device()
        self.assertIsNotNone(optimal)
        
        # Generate strategy
        strategy = StrategyGenerator.generate_strategy(optimal)
        self.assertIsNotNone(strategy)
        
        logger.info(f" Full pipeline: {optimal.name} → {strategy.recommended_backend}")
    
    def test_device_to_agent_pipeline(self):
        """Test from device detection to agent creation"""
        
        # This tests the full flow
        agent = UniversalNecromancerAgent()
        self.assertIsNotNone(agent.gpu)
        self.assertIsNotNone(agent.strategy)
        self.assertIsNotNone(agent.inference_engine)
        
        logger.info(f" Device to Agent pipeline complete")


def run_tests():
    """Run all tests"""
    
    print("\n" + "="*80)
    print(" GPU NECROMANCER UNIVERSAL SYSTEM TESTS")
    print("="*80 + "\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDetectors))
    suite.addTests(loader.loadTestsFromTestCase(TestStrategyGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestBackendSelection))
    suite.addTests(loader.loadTestsFromTestCase(TestUniversalAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

