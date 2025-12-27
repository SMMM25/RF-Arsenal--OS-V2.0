#!/usr/bin/env python3
"""
RF Arsenal OS - FPGA Acceleration Integration Tests
Comprehensive tests for FPGA acceleration modules

Tests:
- FPGA Controller functionality
- DSP Accelerator operations
- Stealth mode features
- LTE/5G acceleration
- Image management
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.fpga import (
    FPGAController,
    FPGAConfig,
    FPGAStatus,
    FPGAMode,
    AcceleratorType,
    FPGAImageManager,
    FPGAImage,
    ImageType,
    ImageMetadata,
    DSPAccelerator,
    FFTConfig,
    FilterConfig,
    ModulatorConfig,
    DSPOperation,
    WindowType,
    StealthFPGAController,
    StealthProfile,
    FrequencyHopConfig,
    PowerRampConfig,
    LTEAccelerator,
    OFDMConfig,
    ChannelConfig,
    LTEFrameConfig,
)


class TestFPGAController(unittest.TestCase):
    """Test FPGAController functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.controller = FPGAController()
        self.events = []
        
    def tearDown(self):
        """Clean up"""
        self.controller = None
        
    def test_initialization(self):
        """Test controller initialization"""
        self.assertEqual(self.controller.status, FPGAStatus.UNINITIALIZED)
        self.assertIsNotNone(self.controller.config)
        self.assertFalse(self.controller.is_configured)
        
    def test_config_creation(self):
        """Test FPGAConfig creation and validation"""
        config = FPGAConfig()
        
        self.assertEqual(config.mode, FPGAMode.STANDARD)
        self.assertEqual(config.sample_rate, 30_720_000)
        self.assertEqual(config.fft_size, 2048)
        self.assertFalse(config.stealth_enabled)
        
    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = FPGAConfig(
            mode=FPGAMode.DSP_ACCEL,
            fft_size=4096,
            stealth_enabled=True,
        )
        
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict['mode'], 'dsp_accel')
        self.assertEqual(config_dict['fft_size'], 4096)
        self.assertTrue(config_dict['stealth_enabled'])
        
    def test_connect_without_device(self):
        """Test connect with mock device"""
        mock_device = MagicMock()
        
        result = self.controller.connect(mock_device)
        
        self.assertTrue(result)
        self.assertIsNotNone(self.controller._device)
        
    def test_disconnect(self):
        """Test disconnect"""
        mock_device = MagicMock()
        self.controller.connect(mock_device)
        
        result = self.controller.disconnect()
        
        self.assertTrue(result)
        self.assertIsNone(self.controller._device)
        self.assertEqual(self.controller.status, FPGAStatus.UNINITIALIZED)
        
    def test_configure_standard_mode(self):
        """Test standard mode configuration"""
        mock_device = MagicMock()
        self.controller.connect(mock_device)
        
        config = FPGAConfig(mode=FPGAMode.STANDARD)
        result = self.controller.configure(config)
        
        # Should succeed even without hardware
        self.assertTrue(result)
        
    def test_configure_dsp_mode(self):
        """Test DSP acceleration mode configuration"""
        mock_device = MagicMock()
        self.controller.connect(mock_device)
        
        config = FPGAConfig(
            mode=FPGAMode.DSP_ACCEL,
            enable_fft_accel=True,
            enable_filter_accel=True,
        )
        result = self.controller.configure(config)
        
        self.assertTrue(result)
        self.assertEqual(self.controller.config.mode, FPGAMode.DSP_ACCEL)
        
    def test_configure_stealth_mode(self):
        """Test stealth mode configuration"""
        mock_device = MagicMock()
        self.controller.connect(mock_device)
        
        config = FPGAConfig(
            mode=FPGAMode.STEALTH,
            stealth_enabled=True,
            frequency_hopping=True,
            power_ramping=True,
        )
        result = self.controller.configure(config)
        
        self.assertTrue(result)
        
    def test_configure_lte_mode(self):
        """Test LTE acceleration mode configuration"""
        mock_device = MagicMock()
        self.controller.connect(mock_device)
        
        config = FPGAConfig(
            mode=FPGAMode.LTE_ACCEL,
            lte_enabled=True,
            ofdm_nfft=2048,
            num_prb=100,
        )
        result = self.controller.configure(config)
        
        self.assertTrue(result)
        
    def test_start_stop(self):
        """Test start/stop processing"""
        mock_device = MagicMock()
        self.controller.connect(mock_device)
        self.controller.configure(FPGAConfig())
        
        result_start = self.controller.start()
        self.assertTrue(result_start)
        self.assertEqual(self.controller.status, FPGAStatus.RUNNING)
        
        result_stop = self.controller.stop()
        self.assertTrue(result_stop)
        self.assertEqual(self.controller.status, FPGAStatus.CONFIGURED)
        
    def test_emergency_stop(self):
        """Test emergency stop"""
        mock_device = MagicMock()
        self.controller.connect(mock_device)
        self.controller.configure(FPGAConfig())
        self.controller.start()
        
        result = self.controller.emergency_stop()
        
        self.assertTrue(result)
        self.assertEqual(self.controller.status, FPGAStatus.EMERGENCY_STOP)
        
    def test_fft_configuration(self):
        """Test FFT engine configuration"""
        mock_device = MagicMock()
        self.controller.connect(mock_device)
        self.controller.configure(FPGAConfig())
        
        # Valid FFT sizes
        for size in [64, 128, 256, 512, 1024, 2048, 4096]:
            result = self.controller.configure_fft(size=size)
            self.assertTrue(result)
            
        # Invalid FFT size
        result = self.controller.configure_fft(size=100)
        self.assertFalse(result)
        
    def test_filter_configuration(self):
        """Test filter configuration"""
        mock_device = MagicMock()
        self.controller.connect(mock_device)
        self.controller.configure(FPGAConfig())
        
        coefficients = [0.1, 0.2, 0.4, 0.2, 0.1]
        result = self.controller.configure_filter(coefficients)
        
        self.assertTrue(result)
        
    def test_stealth_activation(self):
        """Test stealth mode activation"""
        mock_device = MagicMock()
        self.controller.connect(mock_device)
        self.controller.configure(FPGAConfig())
        
        hop_freqs = [915e6, 916e6, 917e6, 918e6]
        result = self.controller.activate_stealth(
            hop_frequencies=hop_freqs,
            hop_rate_ms=10.0,
            enable_power_ramping=True,
        )
        
        self.assertTrue(result)
        self.assertTrue(self.controller._stealth_active)
        self.assertEqual(self.controller.status, FPGAStatus.STEALTH_ACTIVE)
        
    def test_stealth_deactivation(self):
        """Test stealth mode deactivation"""
        mock_device = MagicMock()
        self.controller.connect(mock_device)
        self.controller.configure(FPGAConfig())
        self.controller.activate_stealth([915e6, 916e6])
        
        result = self.controller.deactivate_stealth()
        
        self.assertTrue(result)
        self.assertFalse(self.controller._stealth_active)
        
    def test_lte_ofdm_configuration(self):
        """Test LTE OFDM configuration"""
        mock_device = MagicMock()
        self.controller.connect(mock_device)
        self.controller.configure(FPGAConfig())
        
        result = self.controller.configure_lte_ofdm(
            nfft=2048,
            cp_type="normal",
            num_prb=100,
            subcarrier_spacing_khz=15,
        )
        
        self.assertTrue(result)
        
    def test_get_status(self):
        """Test status retrieval"""
        mock_device = MagicMock()
        self.controller.connect(mock_device)
        self.controller.configure(FPGAConfig(mode=FPGAMode.DSP_ACCEL))
        
        status = self.controller.get_status()
        
        self.assertIn('status', status)
        self.assertIn('mode', status)
        self.assertIn('capabilities', status)
        self.assertEqual(status['mode'], 'dsp_accel')
        
    def test_ai_command_configure(self):
        """Test AI command interface - configure"""
        mock_device = MagicMock()
        self.controller.connect(mock_device)
        
        async def run_test():
            result = await self.controller.ai_command(
                'configure',
                {'mode': 'dsp_accel', 'fft_size': 4096}
            )
            return result
            
        result = asyncio.run(run_test())
        
        self.assertTrue(result['success'])
        
    def test_ai_command_stealth(self):
        """Test AI command interface - stealth"""
        mock_device = MagicMock()
        self.controller.connect(mock_device)
        self.controller.configure(FPGAConfig())
        
        async def run_test():
            result = await self.controller.ai_command(
                'activate_stealth',
                {'hop_frequencies': [915e6, 916e6], 'max_power_dbm': 5.0}
            )
            return result
            
        result = asyncio.run(run_test())
        
        self.assertTrue(result['success'])
        self.assertTrue(result['stealth_active'])


class TestFPGAImageManager(unittest.TestCase):
    """Test FPGAImageManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.temp_dir) / "images"
        self.image_dir.mkdir()
        
        self.manager = FPGAImageManager(
            image_dir=self.image_dir,
            hdl_dir=Path(self.temp_dir) / "hdl",
            build_dir=Path(self.temp_dir) / "build",
        )
        
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """Test manager initialization"""
        self.assertIsNotNone(self.manager)
        self.assertTrue(self.manager.image_dir.exists())
        
    def test_metadata_creation(self):
        """Test ImageMetadata creation"""
        metadata = ImageMetadata(
            name="test_image",
            version="1.0.0",
            image_type=ImageType.DSP_ACCEL,
            description="Test FPGA image",
            features=['fft', 'filter'],
        )
        
        self.assertEqual(metadata.name, "test_image")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.image_type, ImageType.DSP_ACCEL)
        
    def test_metadata_to_dict(self):
        """Test metadata serialization"""
        metadata = ImageMetadata(
            name="test",
            version="1.0.0",
            image_type=ImageType.STEALTH,
            features=['stealth', 'hopping'],
        )
        
        data = metadata.to_dict()
        
        self.assertEqual(data['name'], 'test')
        self.assertEqual(data['type'], 'stealth')
        self.assertIn('stealth', data['features'])
        
    def test_metadata_from_dict(self):
        """Test metadata deserialization"""
        data = {
            'name': 'test',
            'version': '2.0.0',
            'type': 'lte',
            'features': ['ofdm'],
        }
        
        metadata = ImageMetadata.from_dict(data)
        
        self.assertEqual(metadata.name, 'test')
        self.assertEqual(metadata.version, '2.0.0')
        self.assertEqual(metadata.image_type, ImageType.LTE)
        
    def test_add_image(self):
        """Test adding image to inventory"""
        # Create test image file
        test_image = self.temp_dir + "/test.rbf"
        with open(test_image, 'wb') as f:
            f.write(b'test image data' * 100)
            
        metadata = ImageMetadata(
            name="test_image",
            version="1.0.0",
            image_type=ImageType.CUSTOM,
        )
        
        result = self.manager.add_image(Path(test_image), metadata, verify=False)
        
        self.assertTrue(result)
        self.assertIn('test_image', self.manager.available_images)
        
    def test_get_image(self):
        """Test retrieving image"""
        # Create and add test image
        test_image = self.temp_dir + "/test.rbf"
        with open(test_image, 'wb') as f:
            f.write(b'data' * 100)
            
        metadata = ImageMetadata(
            name="retrieve_test",
            version="1.0.0",
            image_type=ImageType.STANDARD,
        )
        self.manager.add_image(Path(test_image), metadata, verify=False)
        
        image = self.manager.get_image("retrieve_test")
        
        self.assertIsNotNone(image)
        self.assertEqual(image.metadata.name, "retrieve_test")
        
    def test_get_image_by_type(self):
        """Test filtering images by type"""
        # Create multiple test images
        for name, img_type in [("img1", ImageType.STANDARD), 
                               ("img2", ImageType.STEALTH),
                               ("img3", ImageType.STEALTH)]:
            test_image = self.temp_dir + f"/{name}.rbf"
            with open(test_image, 'wb') as f:
                f.write(b'data' * 100)
                
            metadata = ImageMetadata(name=name, version="1.0.0", image_type=img_type)
            self.manager.add_image(Path(test_image), metadata, verify=False)
            
        stealth_images = self.manager.get_image_by_type(ImageType.STEALTH)
        
        self.assertEqual(len(stealth_images), 2)
        
    def test_remove_image(self):
        """Test removing image"""
        test_image = self.temp_dir + "/remove_test.rbf"
        with open(test_image, 'wb') as f:
            f.write(b'data' * 100)
            
        metadata = ImageMetadata(name="remove_test", version="1.0.0", image_type=ImageType.CUSTOM)
        self.manager.add_image(Path(test_image), metadata, verify=False)
        
        result = self.manager.remove_image("remove_test")
        
        self.assertTrue(result)
        self.assertNotIn("remove_test", self.manager.available_images)
        
    def test_get_inventory(self):
        """Test getting complete inventory"""
        test_image = self.temp_dir + "/inventory_test.rbf"
        with open(test_image, 'wb') as f:
            f.write(b'data' * 100)
            
        metadata = ImageMetadata(name="inventory_test", version="1.0.0", image_type=ImageType.CUSTOM)
        self.manager.add_image(Path(test_image), metadata, verify=False)
        
        inventory = self.manager.get_inventory()
        
        self.assertIn("inventory_test", inventory)
        self.assertIn('metadata', inventory["inventory_test"])


class TestDSPAccelerator(unittest.TestCase):
    """Test DSPAccelerator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_fpga = MagicMock()
        self.mock_fpga.is_configured = False
        
        self.dsp = DSPAccelerator(
            fpga_controller=self.mock_fpga,
            use_hardware=False,
            fallback_to_software=True,
        )
        
    def test_initialization(self):
        """Test DSP accelerator initialization"""
        self.assertIsNotNone(self.dsp)
        self.assertFalse(self.dsp.hardware_available)
        
    def test_fft_config_validation(self):
        """Test FFT configuration validation"""
        # Valid config
        config = FFTConfig(size=2048)
        self.assertTrue(config.validate())
        
        # Invalid size
        config = FFTConfig(size=100)
        self.assertFalse(config.validate())
        
        # Invalid overlap
        config = FFTConfig(size=1024, overlap=2000)
        self.assertFalse(config.validate())
        
    def test_filter_config_validation(self):
        """Test filter configuration validation"""
        # Valid config
        config = FilterConfig(order=64, cutoff_freq=0.25)
        self.assertTrue(config.validate())
        
        # Invalid order
        config = FilterConfig(order=500)
        self.assertFalse(config.validate())
        
        # Invalid cutoff
        config = FilterConfig(cutoff_freq=0.7)
        self.assertFalse(config.validate())
        
    def test_software_fft(self):
        """Test software FFT fallback"""
        data = np.random.randn(1024) + 1j * np.random.randn(1024)
        
        # Use config with matching size to avoid padding
        config = FFTConfig(size=1024, window=WindowType.NONE)
        result = self.dsp.fft(data, config)
        
        self.assertEqual(len(result), 1024)
        self.assertTrue(np.iscomplexobj(result))
        
    def test_software_ifft(self):
        """Test software IFFT"""
        data = np.random.randn(1024) + 1j * np.random.randn(1024)
        
        # Use config with matching size to avoid padding
        config = FFTConfig(size=1024, window=WindowType.NONE)
        result = self.dsp.ifft(data, config)
        
        self.assertEqual(len(result), 1024)
        
    def test_fft_roundtrip(self):
        """Test FFT -> IFFT roundtrip"""
        original = np.random.randn(1024) + 1j * np.random.randn(1024)
        
        # Use config with correct window type enum and matching size
        config = FFTConfig(size=1024, shift=False, window=WindowType.NONE)
        freq_domain = self.dsp.fft(original, config)
        result = self.dsp.ifft(freq_domain, config)
        
        np.testing.assert_allclose(original, result[:len(original)], rtol=1e-5)
        
    def test_lowpass_filter(self):
        """Test lowpass filter"""
        # Create test signal with low and high frequency components
        t = np.linspace(0, 1, 1000)
        low_freq = np.sin(2 * np.pi * 10 * t)
        high_freq = np.sin(2 * np.pi * 100 * t)
        signal = low_freq + high_freq
        
        filtered = self.dsp.lowpass_filter(signal, cutoff=0.05, order=64)
        
        # High frequency should be attenuated
        self.assertEqual(len(filtered), len(signal))
        
    def test_bandpass_filter(self):
        """Test bandpass filter"""
        signal = np.random.randn(1000)
        
        filtered = self.dsp.bandpass_filter(signal, 0.1, 0.3, order=64)
        
        self.assertEqual(len(filtered), len(signal))
        
    def test_custom_filter(self):
        """Test custom filter with provided coefficients"""
        signal = np.random.randn(100)
        coefficients = np.ones(5) / 5  # Moving average
        
        filtered = self.dsp.custom_filter(signal, list(coefficients))
        
        self.assertEqual(len(filtered), len(signal))
        
    def test_qpsk_modulation(self):
        """Test QPSK modulation"""
        bits = np.array([0, 1, 1, 0, 0, 0, 1, 1])
        
        self.dsp.configure_modulator(ModulatorConfig(modulation_type='qpsk'))
        symbols = self.dsp.modulate_qam(bits)
        
        self.assertEqual(len(symbols), 4)  # 2 bits per symbol
        
    def test_qpsk_demodulation(self):
        """Test QPSK demodulation"""
        # Create QPSK symbols
        symbols = np.array([
            (1 + 1j) / np.sqrt(2),
            (-1 + 1j) / np.sqrt(2),
            (-1 - 1j) / np.sqrt(2),
            (1 - 1j) / np.sqrt(2),
        ])
        
        self.dsp.configure_modulator(ModulatorConfig(modulation_type='qpsk'))
        bits = self.dsp.demodulate_qam(symbols)
        
        self.assertEqual(len(bits), 8)  # 2 bits per symbol
        
    def test_tone_generation(self):
        """Test tone signal generation"""
        tone = self.dsp.generate_tone(
            frequency=1000,
            duration=0.01,
            sample_rate=48000,
        )
        
        self.assertEqual(len(tone), 480)  # 0.01s * 48000
        
    def test_chirp_generation(self):
        """Test chirp signal generation"""
        chirp = self.dsp.generate_chirp(
            start_freq=100,
            end_freq=1000,
            duration=0.01,
            sample_rate=48000,
        )
        
        self.assertEqual(len(chirp), 480)
        
    def test_noise_generation(self):
        """Test noise generation"""
        noise = self.dsp.generate_noise(1000, power_db=0.0)
        
        self.assertEqual(len(noise), 1000)
        self.assertTrue(np.iscomplexobj(noise))
        
    def test_preamble_generation(self):
        """Test preamble generation"""
        preamble = self.dsp.generate_preamble('zadoff_chu', length=64, root=25)
        
        self.assertEqual(len(preamble), 64)
        
    def test_statistics(self):
        """Test statistics tracking"""
        data = np.random.randn(1024) + 1j * np.random.randn(1024)
        
        self.dsp.fft(data)
        self.dsp.fft(data)
        
        stats = self.dsp.stats
        
        self.assertEqual(stats['fft_count'], 2)
        self.assertEqual(stats['sw_fallbacks'], 2)


class TestStealthFPGAController(unittest.TestCase):
    """Test StealthFPGAController functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_fpga = MagicMock()
        self.stealth = StealthFPGAController(fpga_controller=self.mock_fpga)
        
    def test_initialization(self):
        """Test stealth controller initialization"""
        self.assertIsNotNone(self.stealth)
        self.assertFalse(self.stealth.is_active)
        
    def test_predefined_profiles(self):
        """Test predefined stealth profiles"""
        profiles = self.stealth.list_profiles()
        
        self.assertIn('low_observable', profiles)
        self.assertIn('moderate_stealth', profiles)
        self.assertIn('spread_spectrum', profiles)
        self.assertIn('adaptive', profiles)
        
    def test_get_profile(self):
        """Test getting predefined profile"""
        profile = self.stealth.get_profile('low_observable')
        
        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, 'low_observable')
        
    def test_create_custom_profile(self):
        """Test creating custom profile"""
        profile = self.stealth.create_custom_profile(
            name="my_profile",
            base_profile="moderate_stealth",
            hop_rate_hz=500.0,
        )
        
        self.assertEqual(profile.name, "my_profile")
        
    def test_frequency_hop_config_validation(self):
        """Test FrequencyHopConfig validation"""
        config = FrequencyHopConfig(
            frequencies=[915e6, 916e6, 917e6],
            hop_rate_hz=100.0,
        )
        
        self.assertTrue(config.validate())
        
        # Invalid config (no frequencies)
        config = FrequencyHopConfig(frequencies=[])
        self.assertFalse(config.validate())
        
    def test_power_ramp_config_validation(self):
        """Test PowerRampConfig validation"""
        config = PowerRampConfig(
            min_power_dbm=-30.0,
            max_power_dbm=10.0,
        )
        
        self.assertTrue(config.validate())
        
        # Invalid (min > max)
        config = PowerRampConfig(min_power_dbm=10.0, max_power_dbm=-30.0)
        self.assertFalse(config.validate())
        
    def test_activation_deactivation(self):
        """Test stealth activation and deactivation"""
        async def run_test():
            # Activate
            result = await self.stealth.activate(profile_name='moderate_stealth')
            self.assertTrue(result)
            self.assertTrue(self.stealth.is_active)
            
            # Deactivate
            result = await self.stealth.deactivate()
            self.assertTrue(result)
            self.assertFalse(self.stealth.is_active)
            
        asyncio.run(run_test())
        
    def test_set_hop_frequencies(self):
        """Test setting hop frequencies"""
        async def run_test():
            await self.stealth.activate(profile_name='moderate_stealth')
            
            frequencies = [915e6 + i * 1e6 for i in range(10)]
            result = self.stealth.set_hop_frequencies(frequencies)
            
            self.assertTrue(result)
            
        asyncio.run(run_test())
        
    def test_generate_frequency_band(self):
        """Test frequency band generation"""
        frequencies = self.stealth.generate_frequency_band(
            start_freq=915_000_000,
            end_freq=916_000_000,
            channel_spacing=100_000,
        )
        
        self.assertEqual(len(frequencies), 11)  # 915.0, 915.1, ..., 916.0 MHz
        self.assertEqual(frequencies[0], 915_000_000)
        self.assertEqual(frequencies[-1], 916_000_000)
        
    def test_get_status(self):
        """Test status retrieval"""
        status = self.stealth.get_status()
        
        self.assertIn('active', status)
        self.assertIn('profile', status)
        self.assertIn('stats', status)
        
    def test_ai_command(self):
        """Test AI command interface"""
        async def run_test():
            # Activate via AI command
            result = await self.stealth.ai_command(
                'activate',
                {'profile': 'moderate_stealth'}
            )
            self.assertTrue(result['success'])
            
            # Get status via AI command
            result = await self.stealth.ai_command('get_status', {})
            self.assertTrue(result['success'])
            self.assertIn('status', result)
            
            # Deactivate via AI command
            result = await self.stealth.ai_command('deactivate', {})
            self.assertTrue(result['success'])
            
        asyncio.run(run_test())


class TestLTEAccelerator(unittest.TestCase):
    """Test LTEAccelerator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_fpga = MagicMock()
        self.mock_dsp = MagicMock()
        
        self.lte = LTEAccelerator(
            fpga_controller=self.mock_fpga,
            dsp_accelerator=None,  # Use software fallback
        )
        
    def test_initialization(self):
        """Test LTE accelerator initialization"""
        self.assertIsNotNone(self.lte)
        self.assertIsNotNone(self.lte.ofdm_config)
        self.assertIsNotNone(self.lte.frame_config)
        
    def test_ofdm_config_properties(self):
        """Test OFDMConfig computed properties"""
        config = OFDMConfig(nfft=2048, numerology=0)  # 15 kHz SCS
        
        self.assertEqual(config.subcarrier_spacing_hz, 15000)
        self.assertEqual(config.sample_rate, 30_720_000)
        self.assertEqual(config.symbols_per_slot, 14)
        self.assertEqual(config.slots_per_subframe, 1)
        
    def test_ofdm_config_nr_numerology(self):
        """Test OFDM config for NR numerologies"""
        # 30 kHz SCS
        config = OFDMConfig(nfft=4096, numerology=1)
        self.assertEqual(config.subcarrier_spacing_hz, 30000)
        self.assertEqual(config.slots_per_subframe, 2)
        
        # 120 kHz SCS
        config = OFDMConfig(nfft=4096, numerology=3)
        self.assertEqual(config.subcarrier_spacing_hz, 120000)
        self.assertEqual(config.slots_per_subframe, 8)
        
    def test_cp_length_calculation(self):
        """Test cyclic prefix length calculation"""
        config = OFDMConfig(nfft=2048, cp_type='normal')
        
        # First symbol has longer CP
        cp_first = config.get_cp_length(0)
        cp_other = config.get_cp_length(1)
        
        self.assertGreater(cp_first, cp_other)
        
    def test_frame_config_properties(self):
        """Test LTEFrameConfig properties"""
        config = LTEFrameConfig(bandwidth_mhz=20.0, num_prb=100)
        
        self.assertEqual(config.nfft, 2048)
        self.assertEqual(config.subcarriers_per_prb, 12)
        
    def test_ofdm_symbol_generation(self):
        """Test OFDM symbol generation"""
        # Configure for 20 MHz
        self.lte.configure_frame(LTEFrameConfig(bandwidth_mhz=20.0, num_prb=100))
        
        # Create frequency domain data
        freq_data = np.random.randn(1200) + 1j * np.random.randn(1200)
        
        ofdm_symbol = self.lte.generate_ofdm_symbol(freq_data, symbol_index=0)
        
        # Should include CP
        expected_length = 2048 + 160  # NFFT + CP for first symbol
        self.assertGreater(len(ofdm_symbol), 2048)
        
    def test_ofdm_demodulation(self):
        """Test OFDM symbol demodulation"""
        self.lte.configure_frame(LTEFrameConfig(bandwidth_mhz=20.0, num_prb=100))
        
        # Generate and then demodulate
        freq_data_orig = np.random.randn(1200) + 1j * np.random.randn(1200)
        ofdm_symbol = self.lte.generate_ofdm_symbol(freq_data_orig, symbol_index=1)
        
        freq_data_demod = self.lte.demodulate_ofdm_symbol(ofdm_symbol, symbol_index=1)
        
        self.assertEqual(len(freq_data_demod), 1200)
        
    def test_pss_generation(self):
        """Test PSS generation"""
        for n_id_2 in range(3):
            pss = self.lte.generate_pss(n_id_2)
            
            self.assertEqual(len(pss), 127)
            self.assertTrue(np.iscomplexobj(pss))
            
    def test_sss_generation(self):
        """Test SSS generation"""
        sss = self.lte.generate_sss(n_id_1=0, n_id_2=0)
        
        self.assertEqual(len(sss), 127)
        
    def test_dmrs_generation(self):
        """Test DMRS generation"""
        config = ChannelConfig(num_prb=50)
        
        dmrs = self.lte.generate_dmrs(config, slot_number=0)
        
        self.assertEqual(len(dmrs), 300)  # 50 PRB * 12 / 2 (every other SC)
        
    def test_resource_element_mapping(self):
        """Test RE mapping"""
        self.lte.configure_frame(LTEFrameConfig(bandwidth_mhz=20.0, num_prb=100))
        
        config = ChannelConfig(
            start_prb=0,
            num_prb=10,
            start_symbol=0,
            num_symbols=14,
        )
        
        # Create data
        num_re = config.get_re_count() - 2 * 10 * 12  # Minus DMRS symbols
        data = np.random.randn(num_re) + 1j * np.random.randn(num_re)
        
        grid = self.lte.map_to_resource_elements(data, config)
        
        self.assertIsNotNone(grid)
        
    def test_channel_estimation(self):
        """Test channel estimation"""
        self.lte.configure_frame(LTEFrameConfig(bandwidth_mhz=20.0, num_prb=100))
        
        config = ChannelConfig(num_prb=10)
        
        # Create mock received grid
        rx_grid = np.random.randn(14, 1200) + 1j * np.random.randn(14, 1200)
        
        h_estimate = self.lte.estimate_channel(rx_grid, config)
        
        self.assertIsNotNone(h_estimate)
        
    def test_equalization(self):
        """Test channel equalization"""
        rx_data = np.array([1+1j, 2+2j, 3+3j, 4+4j])
        h_estimate = np.array([0.5+0.5j, 1+0j, 0.5-0.5j, 1+0j])
        
        eq_data = self.lte.equalize(rx_data, h_estimate)
        
        self.assertEqual(len(eq_data), 4)
        
    def test_get_status(self):
        """Test status retrieval"""
        status = self.lte.get_status()
        
        self.assertIn('ofdm_config', status)
        self.assertIn('frame_config', status)
        self.assertIn('stats', status)


class TestFPGAIntegration(unittest.TestCase):
    """Integration tests for FPGA components"""
    
    def test_full_signal_chain(self):
        """Test complete signal processing chain"""
        # Create components
        mock_fpga = MagicMock()
        mock_fpga.is_configured = False
        
        dsp = DSPAccelerator(
            fpga_controller=mock_fpga,
            use_hardware=False,
            fallback_to_software=True,
        )
        
        lte = LTEAccelerator(
            fpga_controller=mock_fpga,
            dsp_accelerator=dsp,
        )
        
        # Configure
        lte.configure_frame(LTEFrameConfig(bandwidth_mhz=10.0, num_prb=50))
        
        # Generate bits -> modulate -> map -> OFDM -> demod -> equalize -> demod
        bits = np.random.randint(0, 2, 2400)
        
        # Modulate
        symbols = dsp.modulate_qam(bits)
        
        # Map to resource grid
        config = ChannelConfig(num_prb=10)
        grid = lte.map_to_resource_elements(symbols, config)
        
        self.assertIsNotNone(grid)
        
    def test_stealth_with_lte(self):
        """Test stealth mode with LTE processing"""
        mock_fpga = MagicMock()
        
        stealth = StealthFPGAController(fpga_controller=mock_fpga)
        lte = LTEAccelerator(fpga_controller=mock_fpga)
        
        async def run_test():
            # Activate stealth
            await stealth.activate(profile_name='moderate_stealth')
            self.assertTrue(stealth.is_active)
            
            # LTE should still work
            lte.configure_frame(LTEFrameConfig(bandwidth_mhz=20.0))
            pss = lte.generate_pss(0)
            self.assertEqual(len(pss), 127)
            
            # Deactivate
            await stealth.deactivate()
            
        asyncio.run(run_test())


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFPGAController))
    suite.addTests(loader.loadTestsFromTestCase(TestFPGAImageManager))
    suite.addTests(loader.loadTestsFromTestCase(TestDSPAccelerator))
    suite.addTests(loader.loadTestsFromTestCase(TestStealthFPGAController))
    suite.addTests(loader.loadTestsFromTestCase(TestLTEAccelerator))
    suite.addTests(loader.loadTestsFromTestCase(TestFPGAIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
